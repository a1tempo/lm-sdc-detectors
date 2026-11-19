"""Detection-quality metrics (proposal §7.1).

Main task: SDC vs. masked runs. Crash runs are excluded and reported
separately as the share of all faulty runs that crash.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def sdc_vs_masked(runs: pd.DataFrame) -> pd.DataFrame:
    """Drop crash runs; add a binary `is_sdc` column."""
    unknown = set(runs["label"]) - {"sdc", "masked", "crash"}
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    kept = runs[runs["label"] != "crash"].copy()
    kept["is_sdc"] = kept["label"] == "sdc"
    return kept


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, ties counted as half (the Mann-Whitney statistic).

    Computed from ranks, so +inf scores are fine. NaN if only one class is present.
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """SDC recall at a threshold frozen on the fitting set (alarm: score > threshold)."""
    y_true = np.asarray(y_true, dtype=bool)
    if not y_true.any():
        return float("nan")
    return float((np.asarray(scores)[y_true] > threshold).mean())


def fpr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Share of masked runs that raise an alarm at the frozen threshold."""
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.all():
        return float("nan")
    return float((np.asarray(scores)[~y_true] > threshold).mean())


def crash_rate(runs: pd.DataFrame) -> float:
    """Share of all faulty runs labelled crash."""
    return float((runs["label"] == "crash").mean())
