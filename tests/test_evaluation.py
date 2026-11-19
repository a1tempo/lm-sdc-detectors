import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sdc.detectors.classical.learned import LearnedDetector, out_of_fold_scores
from sdc.eval.bootstrap import bootstrap_ci, image_resamples
from sdc.eval.metrics import auroc, weighted_auroc, weighted_recall
from sdc.eval.stratify import stratified_metrics
from test_classical import make_runs


def test_image_resamples_keep_whole_images():
    image_ids = np.array([3, 3, 1, 2, 2, 2])
    for idx in image_resamples(image_ids, 20, seed=1):
        drawn = image_ids[idx]
        for image in np.unique(drawn):
            # every copy of an image brings all of its runs
            assert (drawn == image).sum() % (image_ids == image).sum() == 0
        assert len(np.unique(drawn)) <= 3


def test_bootstrap_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"image_id": np.repeat(np.arange(50), 4), "x": rng.normal(size=200)})
    point, low, high = bootstrap_ci(df, lambda d: d["x"].mean(), n_resamples=300, seed=3)
    assert low < point < high
    assert bootstrap_ci(df, lambda d: d["x"].mean(), n_resamples=300, seed=3) == (point, low, high)


def test_weighted_metrics():
    y = np.array([1, 1, 0, 0, 0], bool)
    s = np.array([0.9, 0.1, 0.5, 0.2, 0.3])
    w = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
    assert weighted_auroc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert weighted_auroc(y, s, np.ones(5)) == pytest.approx(auroc(y, s))
    assert weighted_recall(y, s, 0.4, w) == 0.75
    assert weighted_auroc(y, np.r_[np.inf, s[1:]], w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_stratified_metrics_per_stratum():
    rows = []
    for i in range(80):
        sdc = i % 4 == 0
        rows.append({"name": "d", "label": "sdc" if sdc else "masked", "image_id": i // 2,
                     "bit_group": "exponent" if i < 40 else "mantissa",
                     "score": (1.0 if sdc else 0.0) if i < 40 else 0.5, "threshold": 0.5})
    rows += [{"name": "d", "label": "crash", "image_id": 0, "bit_group": "sign", "score": 9.0,
              "threshold": 0.5}]
    table = stratified_metrics(pd.DataFrame(rows), "bit_group", n_resamples=50)
    by = table.set_index("bit_group")
    assert "sign" not in by.index                      # crashes dropped
    assert by.loc["exponent", "auroc"] == 1.0 and by.loc["exponent", "recall"] == 1.0
    assert by.loc["mantissa", "auroc"] == 0.5 and by.loc["mantissa", "recall"] == 0.0


def test_out_of_fold_scores_never_score_the_training_image():
    rng = np.random.default_rng(0)
    runs = make_runs([{"label": "sdc" if i % 3 == 0 else "masked", "image_id": i // 3,
                       "layer3_max": (20.0 if i % 3 == 0 else 5.0) + rng.normal()} for i in range(150)])
    scores = out_of_fold_scores(lambda: LearnedDetector("whitebox"), runs)
    assert scores.shape == (150,)
    assert auroc(runs["label"] == "sdc", scores) > 0.95
