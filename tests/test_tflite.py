"""The torch-free helpers the Pi runner uses must match the PyTorch pipeline."""

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from sdc.data.eurosat import EVAL_TRANSFORM, IMAGE_SIZE, MEAN, STD
from sdc.injection.labeling import label_run
from sdc.model import tflite
from sdc.observations.stats import blackbox_stats


def test_constants_match_eurosat():
    assert tflite.IMAGE_SIZE == IMAGE_SIZE
    assert tflite.MEAN.tolist() == pytest.approx(MEAN)
    assert tflite.STD.tolist() == pytest.approx(STD)


def test_preprocess_matches_eval_transform(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "image.jpg"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    expected = EVAL_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).numpy()
    got = tflite.preprocess_image(path)
    assert got.shape == (1, 3, 224, 224) and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("logits", [
    [0.1, 2.0, -1.0], [3.0, 2.0, -1.0], [math.nan, 1.0, 0.0], [math.inf, 1.0, 0.0],
    [1e30, -1e30, 0.0],
])
def test_label_and_stats_match_pytorch(logits):
    array = np.array(logits, dtype=np.float32)
    tensor = torch.tensor(logits, dtype=torch.float32)
    assert tflite.label_logits(1, array) == label_run(1, tensor, crashed=False).value
    expected, got = blackbox_stats(tensor), tflite.blackbox_stats_np(array)
    for key in expected:
        assert got[key] == pytest.approx(expected[key], nan_ok=True, abs=1e-12)


def test_crash_without_logits():
    assert tflite.label_logits(0, None) == "crash"


def test_tflite_helpers_import_without_torch():
    code = ("import sys; sys.modules['torch'] = None; import sdc.model.tflite as t; "
            "import sdc.config; print(t.label_logits(0, None))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "crash"
