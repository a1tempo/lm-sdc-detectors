"""Build the kit the Pi overhead run needs (runs on the Mac, in the LiteRT environment).

Action item: Part I.3c (overhead of the classical detectors on the Pi)

Writes <out> (default runs/overhead_kit/), which is copied to the Pi:
    classifier.tflite            the FP32 classifier (logits only): the no-detector baseline
    classifier_monitored.tflite  the same model also returning the white-box observation
                                 (min, max, mean, std, nonfinite at 6 points), computed in
                                 the graph, i.e. what a white-box deployment would run
    fit_features.npz             fit-split features and labels (SDC vs. masked), so the Pi
                                 trains the logistic-regression and boosted-trees detectors
                                 itself (no pickles across library versions)
    eval_features.npz            a sample of eval-run features to time the detectors on
    images.json                  eval images to time the classifier on
    reference_ranges.json        the range check's bounds
    kit.json                     provenance and the scikit-learn version to install

The monitored model is checked against PyTorch's ActivationMonitor on a few images first.
Run with the LiteRT environment:
    ~/.venvs/lm-sdc-tflite/bin/python scripts/09_prepare_overhead.py
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import sklearn
import torch

from sdc.config import load_config
from sdc.data.eurosat import load_splits, load_test_split
from sdc.injection.campaign import current_git_commit
from sdc.model import resnet18
from sdc.model.monitor import ActivationMonitor
from sdc.model.tflite import (
    convert_fp32,
    load_interpreter,
    monitored_module,
    preprocess_image,
    run_tflite_outputs,
)
from sdc.observations.stats import feature_matrix, whitebox_stats
from sdc.records.io import load_runs


def convert_monitored(model, out_path: Path) -> None:
    import litert_torch

    litert_torch.convert(monitored_module(model), (torch.randn(1, 3, 224, 224),)).export(str(out_path))


def check_monitored(model, path: Path, image_paths: list[Path]) -> float:
    """Largest relative difference between in-graph stats and PyTorch's, over a few images."""
    interpreter = load_interpreter(path)
    worst = 0.0
    for image_path in image_paths:
        x = preprocess_image(image_path)
        outputs = run_tflite_outputs(interpreter, x)
        stats = next(o for o in outputs if o.shape == (6, 5))
        logits = next(o for o in outputs if o.shape == (1, 10))
        with torch.no_grad(), ActivationMonitor(model) as monitor:
            reference_logits = model(torch.from_numpy(x)).numpy()
        reference = whitebox_stats(monitor.activations)
        expected = np.array([[s.min, s.max, s.mean, s.std, s.nonfinite] for s in reference.values()])
        worst = max(worst, float(np.max(np.abs(stats - expected) / (np.abs(expected) + 1e-3))))
        if int(np.argmax(logits)) != int(np.argmax(reference_logits)):
            raise RuntimeError(f"monitored model predicts a different class on {image_path}")
    if worst > 1e-3:
        raise RuntimeError(f"monitored model's stats differ from PyTorch's by {worst:.2g}")
    return worst


def main(args) -> None:
    config = load_config(args.config)
    paths, data = config["paths"], config["data"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = resnet18.load_checkpoint(paths["checkpoint"])
    checkpoint_hash = resnet18.checkpoint_sha256(paths["checkpoint"])
    splits = load_splits(paths["splits"])
    test_ds = load_test_split(paths["eurosat_dir"], data["split_fractions"], data["split_seed"],
                              indices_path=paths["test_indices"] if splits["split_source"] == "test_indices" else None)
    eval_images = [entry["image_path"] for entry in splits["eval"]][:args.images]
    data_dir = Path(paths["eurosat_dir"])

    print("converting the classifier and the monitored classifier ...", flush=True)
    convert_fp32(model, str(out / "classifier.tflite"))
    convert_monitored(model, out / "classifier_monitored.tflite")
    worst = check_monitored(model, out / "classifier_monitored.tflite",
                            [data_dir / p for p in eval_images[:10]])
    print(f"monitored model matches PyTorch's stats (worst relative difference {worst:.1e})")

    features = {}
    for split in ("fit", "eval"):
        runs = load_runs(Path(paths["fault_runs"]) / split)
        runs = runs[runs["label"] != "crash"]
        if split == "eval":
            runs = runs.sample(n=min(args.eval_runs, len(runs)), random_state=0)
        features[split] = {
            "blackbox": feature_matrix(runs, "blackbox").astype(np.float32),
            "whitebox": feature_matrix(runs, "whitebox").astype(np.float32),
            "raw_whitebox": runs[[f"{p}_{s}" for p in ("stem", "layer1", "layer2", "layer3", "layer4", "fc")
                                  for s in ("min", "max")]].to_numpy(dtype=np.float64),
            "is_sdc": (runs["label"] == "sdc").to_numpy(),
        }
    np.savez_compressed(out / "fit_features.npz", **features["fit"])
    np.savez_compressed(out / "eval_features.npz", **features["eval"])

    (out / "images.json").write_text(json.dumps(eval_images, indent=1))
    shutil.copy(paths["reference_ranges"], out / "reference_ranges.json")
    kit = {
        "checkpoint_sha256": checkpoint_hash, "git_commit": current_git_commit(),
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "n_images": len(eval_images), "n_fit_runs": int(len(features["fit"]["is_sdc"])),
        "n_eval_runs": int(len(features["eval"]["is_sdc"])),
        "monitored_max_rel_diff": worst,
    }
    (out / "kit.json").write_text(json.dumps(kit, indent=2))
    size = sum(p.stat().st_size for p in out.iterdir()) / 1024**2
    print(f"wrote {out} ({size:.0f} MB); on the Pi install scikit-learn=={sklearn.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--out", default="runs/overhead_kit")
    parser.add_argument("--images", type=int, default=200, help="eval images to time the classifier on")
    parser.add_argument("--eval-runs", type=int, default=2000, help="eval runs to time the detectors on")
    main(parser.parse_args())
