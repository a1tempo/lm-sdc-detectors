"""TFLite (LiteRT) conversion and inference.

`convert_fp32` and `convert_int8` are ported from Reem's notebook. Conversion
runs on a workstation (needs litert-torch); inference runs on the Pi
(needs ai-edge-litert). Install with `pip install -e .[tflite]`.

Everything below the converters needs only numpy, Pillow and ai-edge-litert, not torch,
so the Pi runner (scripts/11_run_faulty_tflite.py) works without PyTorch installed.
`preprocess_image`, `label_logits` and `blackbox_stats_np` match EVAL_TRANSFORM,
label_run and blackbox_stats (checked in tests/test_tflite.py).
"""

from __future__ import annotations

import copy
import math

import numpy as np

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sample_inputs():
    import torch

    return (torch.randn(1, 3, 224, 224),)


def convert_fp32(model, out_path: str) -> None:
    import litert_torch

    model = model.eval().cpu()
    edge_model = litert_torch.convert(copy.deepcopy(model), _sample_inputs())
    edge_model.export(out_path)


def convert_int8(model, out_path: str) -> None:
    """PT2E dynamic per-channel INT8 quantization (stretch goal in the proposal)."""
    import litert_torch
    import torch
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    SAMPLE_INPUTS = _sample_inputs()
    model_int8 = copy.deepcopy(model).eval().cpu()
    exported_program = torch.export.export(model_int8, SAMPLE_INPUTS).module()
    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    prepared_model = prepare_pt2e(exported_program, pt2e_quantizer)
    with torch.no_grad():
        prepared_model(*SAMPLE_INPUTS)
    quantized_torch_model = convert_pt2e(prepared_model, fold_quantize=False)
    edge_model = litert_torch.convert(
        quantized_torch_model,
        SAMPLE_INPUTS,
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )
    edge_model.export(out_path)


def load_interpreter(path: str | None = None, num_threads: int | None = None,
                     model_content: bytes | None = None):
    """Return an allocated LiteRT interpreter for a file `path` or bytes `model_content`."""
    from ai_edge_litert.interpreter import Interpreter

    if (path is None) == (model_content is None):
        raise ValueError("give exactly one of path and model_content")
    if path is not None:
        interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    else:
        interpreter = Interpreter(model_content=model_content, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(path) -> np.ndarray:
    """Load an image as a [1, 3, 224, 224] float32 array, the same as EVAL_TRANSFORM.

    torchvision's Resize on a PIL image calls PIL's bilinear resize, so this matches it.
    """
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def label_logits(clean_pred: int, logits: np.ndarray | None) -> str:
    """masked / sdc / crash, the same rule as sdc.injection.labeling.label_run."""
    if logits is None or not np.isfinite(logits).all():
        return "crash"
    return "sdc" if int(np.argmax(logits)) != clean_pred else "masked"


def blackbox_stats_np(logits: np.ndarray) -> dict[str, float]:
    """top-1 probability and entropy (nats), the same as observations.stats.blackbox_stats."""
    logits = np.asarray(logits, dtype=np.float64).flatten()
    if not np.isfinite(logits).all():
        return {"top1_prob": math.nan, "entropy": math.nan}
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.exp(shifted).sum())
    probs = np.exp(log_probs)
    return {"top1_prob": float(probs.max()), "entropy": float(-(probs * log_probs).sum())}


def run_tflite(interpreter, x: np.ndarray) -> np.ndarray:
    """Run one NCHW float input through the interpreter and return float32 logits.

    Ported from the notebook's `run_tflite`: a quantized (int8/uint8) input or output
    is converted with the tensor's scale and zero point.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    if input_dtype in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        info = np.iinfo(input_dtype)
        x = np.clip(np.round(x / scale + zero_point), info.min, info.max).astype(input_dtype)
    else:
        x = np.asarray(x, dtype=np.float32)

    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_details["quantization"]
        return scale * (out.astype(np.float32) - zero_point)
    return out.astype(np.float32)


def check_layer_alignment(torch_model, tflite_path: str) -> dict:
    """Verify weight faults stay attributable to their original layer after conversion.

    TFLite may fuse or restructure layers (proposal §6.1). TODO: map each PyTorch
    weight tensor to its TFLite tensor and report anything fused or missing.
    """
    raise NotImplementedError


def monitored_module(model):
    """Wrap a ResNet-18 so one forward pass returns (logits, stats) with stats computed
    in the graph: shape [6, 5] = (stem, layer1..4, fc) x (min, max, mean, std, nonfinite).

    This is the white-box model as it would be deployed: the converted .tflite computes
    the observation itself, so the Pi pays for it inside the same invoke. Stats are over
    finite values, like observations.stats.layer_stats, but in float32 (fine for timing
    and for clean runs; faulty runs near 1e38 would overflow the variance).
    """
    import torch

    def point_stats(x):
        x = x.flatten()
        finite = (x - x) == 0                      # False for NaN and +-inf
        n = finite.sum().to(x.dtype)
        values = torch.where(finite, x, torch.zeros_like(x))
        mean = values.sum() / n
        var = torch.where(finite, (x - mean) ** 2, torch.zeros_like(x)).sum() / n
        big = torch.full_like(x, 3.0e38)
        return torch.stack([
            torch.where(finite, x, big).min(), torch.where(finite, x, -big).max(),
            mean, torch.sqrt(var), x.numel() - n,
        ])

    class Monitored(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            n = self.net
            stem = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l1 = n.layer1(stem)
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            l4 = n.layer4(l3)
            logits = n.fc(torch.flatten(n.avgpool(l4), 1))
            stats = torch.stack([point_stats(a) for a in (stem, l1, l2, l3, l4, logits)])
            return logits, stats

    return Monitored(copy.deepcopy(model).eval().cpu()).eval()


def run_tflite_outputs(interpreter, x: np.ndarray) -> list[np.ndarray]:
    """Run one float input and return every output, in the interpreter's output order."""
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], np.asarray(x, dtype=np.float32))
    interpreter.invoke()
    return [interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()]
