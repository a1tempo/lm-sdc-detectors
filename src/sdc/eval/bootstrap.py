"""95% bootstrap confidence intervals, resampling over images (not runs)."""

from typing import Callable

import numpy as np
import pandas as pd


def image_resamples(image_ids: np.ndarray, n_resamples: int, seed: int = 0) -> list[np.ndarray]:
    """Row indices for each resample: images drawn with replacement, all runs of each kept.

    Runs of the same image share a clean prediction and image content, so they are not
    independent; resampling whole images keeps that dependence in the interval.
    """
    image_ids = np.asarray(image_ids)
    order = np.argsort(image_ids, kind="stable")
    images, starts = np.unique(image_ids[order], return_index=True)
    groups = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(len(images), size=len(images))])
            for _ in range(n_resamples)]


def bootstrap_ci(runs: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                 n_resamples: int = 1000, ci: float = 0.95, seed: int = 0,
                 resamples: list[np.ndarray] | None = None) -> tuple[float, float, float]:
    """Return (point estimate, lower, upper), resampling image_id with replacement.

    Pass `resamples` (from image_resamples on the same rows) to reuse draws across
    metrics, so intervals for different detectors come from the same resampled sets.
    NaN resamples (e.g. no SDC drawn) are dropped.
    """
    if resamples is None:
        resamples = image_resamples(runs["image_id"].to_numpy(), n_resamples, seed)
    point = metric(runs)
    values = np.array([metric(runs.iloc[idx]) for idx in resamples], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return point, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return point, float(low), float(high)
