import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sdc.detectors.base import threshold_for_fpr
from sdc.detectors.classical.confidence import ConfidenceCheck
from sdc.detectors.classical.learned import LearnedDetector
from sdc.detectors.classical.quantile import QuantileMonitor
from sdc.detectors.classical.range_check import RangeCheck
from sdc.detectors.classical.sanity import SanityCheck
from sdc.eval.metrics import (
    auroc,
    crash_rate,
    fpr_at_threshold,
    recall_at_threshold,
    sdc_vs_masked,
)
from sdc.model.monitor import MONITORED_POINTS
from sdc.observations.stats import feature_matrix, feature_names, feature_vector

RANGES = {point: {"min": 0.0, "max": 10.0, "quantile_low": 1.0, "quantile_high": 8.0}
          for point in MONITORED_POINTS}


def make_runs(rows: list[dict]) -> pd.DataFrame:
    """Runs with every point in [0, 5] unless a row overrides a column."""
    base = {"label": "masked", "top1_prob": 0.9, "entropy": 0.3}
    for point in MONITORED_POINTS:
        base.update({f"{point}_min": 2.0, f"{point}_max": 5.0, f"{point}_mean": 3.0,
                     f"{point}_std": 1.0, f"{point}_nonfinite": 0})
    return pd.DataFrame([{**base, "run_id": f"r{i}", **row} for i, row in enumerate(rows)])


def test_range_check_matches_eq1():
    runs = make_runs([{}, {"layer3_max": 15.0}, {"stem_min": -2.0, "fc_max": 13.0},
                      {"layer2_nonfinite": 4}, {"layer4_min": math.nan, "layer4_max": math.nan}])
    scores = RangeCheck().fit(None, RANGES).score(runs)
    np.testing.assert_allclose(scores[:3], [0.0, 0.5, 0.3])
    assert np.isinf(scores[3:]).all()


def test_quantile_uses_quantile_bounds():
    runs = make_runs([{"layer1_max": 9.0}, {"layer1_min": 0.5}])
    np.testing.assert_allclose(QuantileMonitor().fit(None, RANGES).score(runs), [1 / 7, 0.5 / 7])
    np.testing.assert_allclose(RangeCheck().fit(None, RANGES).score(runs), [0.0, 0.0])


def test_range_check_needs_whitebox_and_ranges():
    with pytest.raises(ValueError):
        RangeCheck("blackbox")
    with pytest.raises(ValueError):
        RangeCheck().fit(None, None)


def test_sanity_uses_observations_not_labels():
    runs = make_runs([{}, {"label": "sdc"}, {"top1_prob": math.nan}, {"top1_prob": None},
                      {"layer1_nonfinite": 2}])
    np.testing.assert_array_equal(SanityCheck("whitebox").score(runs), [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(SanityCheck("blackbox").score(runs), [0, 0, 1, 1, 0])


def test_confidence_is_entropy_and_nan_is_worst():
    runs = make_runs([{"entropy": 0.1}, {"entropy": 1.5}, {"entropy": math.nan}])
    np.testing.assert_array_equal(ConfidenceCheck().score(runs), [0.1, 1.5, np.inf])


def test_learned_separates_and_rejects_crashes():
    rng = np.random.default_rng(0)
    rows = [{"label": "sdc" if i % 2 else "masked",
             "layer3_max": (20.0 if i % 2 else 5.0) + rng.normal(), "entropy": rng.random()}
            for i in range(200)]
    runs = make_runs(rows)
    detector = LearnedDetector("whitebox").fit(runs)
    assert auroc(runs["label"] == "sdc", detector.score(runs)) > 0.99
    with pytest.raises(ValueError):
        LearnedDetector().fit(make_runs([{"label": "crash"}, {"label": "sdc"}]))
    with pytest.raises(RuntimeError):
        LearnedDetector().score(runs)


def test_threshold_for_fpr():
    assert threshold_for_fpr(np.arange(100.0), 0.05) == 94.0      # 95..99 above: 5%
    assert threshold_for_fpr(np.r_[np.zeros(97), np.ones(3)], 0.05) == 0.0
    assert threshold_for_fpr(np.r_[np.zeros(90), np.ones(10)], 0.05) == 1.0
    assert threshold_for_fpr(np.array([0.0, np.inf]), 0.05) == np.inf


def test_freeze_threshold_and_alarms():
    runs = make_runs([{"entropy": e} for e in np.linspace(0, 1, 100)] + [{"label": "sdc", "entropy": 5.0}])
    detector = ConfidenceCheck()
    threshold = detector.freeze_threshold(runs, 0.05)
    alarms = detector.alarms(runs)
    assert fpr_at_threshold(runs["label"] == "sdc", detector.score(runs), threshold) <= 0.05
    assert alarms[-1]


def test_auroc_matches_sklearn_and_handles_inf():
    rng = np.random.default_rng(1)
    y = rng.random(300) < 0.3
    s = np.round(rng.random(300) + y * 0.4, 1)                     # ties included
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auroc([0, 0, 1], [0.1, 0.2, np.inf]) == 1.0
    assert math.isnan(auroc([1, 1], [0.1, 0.2]))


def test_recall_fpr_and_labels():
    y = np.array([1, 1, 0, 0, 0, 0], bool)
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    assert recall_at_threshold(y, s, 0.5) == 0.5
    assert fpr_at_threshold(y, s, 0.5) == 0.25
    runs = pd.DataFrame({"label": ["sdc", "masked", "crash", "masked"]})
    kept = sdc_vs_masked(runs)
    assert kept["is_sdc"].tolist() == [True, False, False]
    assert crash_rate(runs) == 0.25


def test_feature_vector_order_and_transform():
    assert feature_names("blackbox") == ["top1_prob", "entropy"]
    assert len(feature_names("whitebox")) == 2 + 5 * len(MONITORED_POINTS)
    runs = make_runs([{"layer4_max": 1e38, "fc_min": -1e30, "stem_mean": math.nan}])
    matrix = feature_matrix(runs, "whitebox")
    row = feature_vector(runs.iloc[0].to_dict(), "whitebox")
    np.testing.assert_allclose(matrix[0], row)
    names = feature_names("whitebox")
    assert row[names.index("layer4_max")] == pytest.approx(math.log1p(1e38))
    assert row[names.index("fc_min")] == pytest.approx(-math.log1p(1e30))
    assert row[names.index("stem_mean")] == 0.0
    assert np.isfinite(feature_matrix(make_runs([{"layer1_max": np.inf}]), "whitebox")).all()
    with pytest.raises(ValueError):
        feature_names("greybox")
