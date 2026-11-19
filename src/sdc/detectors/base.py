"""Common interface for every detector: higher score = more likely SDC."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    name: str = "detector"
    supervised: bool = False   # True if fit() uses SDC labels

    def __init__(self, access_level: str = "whitebox"):
        self.access_level = access_level   # "blackbox" or "whitebox"
        self.threshold: float | None = None

    def fit(self, fit_runs: pd.DataFrame, reference_ranges: dict | None = None) -> "Detector":
        """Fit on the fitting set only. Default: nothing to fit."""
        return self

    @abstractmethod
    def score(self, runs: pd.DataFrame) -> np.ndarray:
        """Return one score s_i per run."""

    def freeze_threshold(self, fit_runs: pd.DataFrame, target_fpr: float = 0.05) -> float:
        """Choose the threshold giving at most `target_fpr` on masked fitting runs, and store it.

        A run raises an alarm when its score is strictly above the threshold. The threshold
        is the smallest masked score that leaves at most target_fpr of masked runs above
        it, so ties (common for 0/1 detectors) never push the fitting FPR over the target.
        """
        masked = fit_runs[fit_runs["label"] == "masked"]
        if masked.empty:
            raise ValueError("no masked fitting runs to set a threshold on")
        self.threshold = threshold_for_fpr(self.score(masked), target_fpr)
        return self.threshold

    def alarms(self, runs: pd.DataFrame, scores: np.ndarray | None = None) -> np.ndarray:
        """True where the score is above the frozen threshold."""
        if self.threshold is None:
            raise RuntimeError("call freeze_threshold first")
        scores = self.score(runs) if scores is None else scores
        return np.asarray(scores) > self.threshold


def threshold_for_fpr(masked_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest t among the scores with mean(masked_scores > t) <= target_fpr."""
    scores = np.sort(np.asarray(masked_scores, dtype=np.float64))
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    candidates = np.unique(scores)
    fpr = (len(scores) - np.searchsorted(scores, candidates, side="right")) / len(scores)
    return float(candidates[np.argmax(fpr <= target_fpr)])   # fpr only falls; the last is 0
