import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch

from sdc.config import REPO_ROOT
from sdc.detectors.classical.range_check import RangeCheck
from sdc.model.monitor import MONITORED_POINTS
from sdc.observations.stats import _signed_log, blackbox_stats
from sdc.overhead.measure import peak_rss_bytes, time_per_call

spec = importlib.util.spec_from_file_location("overhead", REPO_ROOT / "scripts" / "09_overhead.py")
overhead = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overhead)


def test_time_per_call():
    calls = []
    t = time_per_call(lambda: calls.append(1), warmup=3, iterations=20)
    assert len(calls) == 23 and t["iterations"] == 20
    assert 0 <= t["min_s"] <= t["p50_s"] <= t["p95_s"]
    assert peak_rss_bytes() > 0


def test_pi_math_matches_the_detectors():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(size=5) * 1e30, np.inf, -np.inf, np.nan, 0.0]
    np.testing.assert_allclose(overhead.signed_log(x), _signed_log(x))
    logits = rng.normal(size=10).astype(np.float32)
    expected = blackbox_stats(torch.from_numpy(logits))
    np.testing.assert_allclose(overhead.blackbox_from_logits(logits),
                               [expected["top1_prob"], expected["entropy"]], rtol=1e-12)
    ranges = {p: {"min": -1.0, "max": 4.0} for p in MONITORED_POINTS}
    row = {}
    for p in MONITORED_POINTS:
        row.update({f"{p}_min": rng.uniform(-3, 1), f"{p}_max": rng.uniform(2, 9), f"{p}_nonfinite": 0})
    min_max = np.array([v for p in MONITORED_POINTS for v in (row[f"{p}_min"], row[f"{p}_max"])])
    expected = RangeCheck().fit(None, ranges).score(pd.DataFrame([row]))[0]
    got = overhead.range_score(min_max, np.full(6, -1.0), np.full(6, 4.0))
    assert got == pytest.approx(expected)


def test_monitored_module_matches_activation_monitor():
    from sdc.model.monitor import ActivationMonitor
    from sdc.model.resnet18 import build_resnet18
    from sdc.model.tflite import monitored_module
    from sdc.observations.stats import whitebox_stats

    torch.manual_seed(0)
    model = build_resnet18().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits, stats = monitored_module(model)(x)
        with ActivationMonitor(model) as monitor:
            expected_logits = model(x)
    expected = np.array([[s.min, s.max, s.mean, s.std, s.nonfinite]
                         for s in whitebox_stats(monitor.activations).values()])
    np.testing.assert_allclose(logits.numpy(), expected_logits.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(stats.numpy(), expected, rtol=1e-4, atol=1e-5)
