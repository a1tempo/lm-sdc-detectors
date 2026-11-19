"""Build the observation vector o_i at both access levels (proposal §6.2).

black-box: top-1 softmax probability and entropy of the output distribution.
white-box: black-box + per monitored point (min, max, mean, std, nonfinite count).

Everything is computed on the CPU in float64: a corrupted weight can push many
values in a layer near 1e38, and summing those overflows the mean and std to inf
in float32, which would make finite activations look non-finite.
"""

import math

import numpy as np
import pandas as pd
import torch

from sdc.model.monitor import MONITORED_POINTS
from sdc.records.schema import LayerStats

QUANTILE_LOW = 0.001
QUANTILE_HIGH = 0.999


def blackbox_stats(logits: torch.Tensor) -> dict[str, float]:
    """Return {"top1_prob": ..., "entropy": ...}, with entropy in nats.

    `logits` holds one image (shape [10] or [1, 10]). Both values are NaN if any
    logit is non-finite.
    """
    logits = logits.detach().cpu().to(torch.float64).flatten()
    if not torch.isfinite(logits).all():
        return {"top1_prob": math.nan, "entropy": math.nan}
    log_probs = torch.log_softmax(logits, dim=0)
    probs = log_probs.exp()
    return {
        "top1_prob": probs.max().item(),
        "entropy": -(probs * log_probs).sum().item(),
    }


def layer_stats(activation: torch.Tensor) -> LayerStats:
    """Summary statistics over finite values plus the non-finite count.

    If no value is finite, min/max/mean/std are NaN. std is the population std,
    since the activations are the whole layer rather than a sample of it.
    """
    values = activation.detach().cpu().to(torch.float64)
    finite = torch.isfinite(values)
    nonfinite = int((~finite).sum())
    values = values[finite]
    if values.numel() == 0:
        return LayerStats(math.nan, math.nan, math.nan, math.nan, nonfinite)
    return LayerStats(
        min=values.min().item(),
        max=values.max().item(),
        mean=values.mean().item(),
        std=values.std(unbiased=False).item(),
        nonfinite=nonfinite,
    )


def whitebox_stats(activations: dict[str, torch.Tensor]) -> dict[str, LayerStats]:
    """Apply layer_stats to every monitored point."""
    return {point: layer_stats(activation) for point, activation in activations.items()}


BLACKBOX_FEATURES = ["top1_prob", "entropy"]
WHITEBOX_STATS = ["min", "max", "mean", "std", "nonfinite"]


def feature_names(access_level: str) -> list[str]:
    """Column names of the observation o_i, in the fixed order feature_vector uses."""
    if access_level == "blackbox":
        return list(BLACKBOX_FEATURES)
    if access_level == "whitebox":
        return BLACKBOX_FEATURES + [f"{p}_{s}" for p in MONITORED_POINTS for s in WHITEBOX_STATS]
    raise ValueError(f"access_level must be blackbox or whitebox, got {access_level!r}")


def _signed_log(values: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|), with NaN -> 0 and +-inf -> +-log(max float64).

    A corrupted weight can push a layer's values near 1e38, which would swamp any
    linear model on raw numbers. The transform keeps order and sign, and a NaN (a layer
    with no finite values) is still visible through that layer's nonfinite count.
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(np.float64).max,
                           neginf=-np.finfo(np.float64).max)
    return np.sign(values) * np.log1p(np.abs(values))


def feature_matrix(runs: pd.DataFrame, access_level: str) -> np.ndarray:
    """feature_vector for every run: shape [n_runs, n_features]."""
    return _signed_log(runs[feature_names(access_level)].to_numpy(dtype=np.float64))


def feature_vector(record_row: dict, access_level: str) -> np.ndarray:
    """Flatten an observation into a fixed-order numeric vector.

    Shared by the learned detector and anything else that needs o_i as numbers,
    so every detector sees identical information: black-box is (top1_prob, entropy);
    white-box adds (min, max, mean, std, nonfinite) at each monitored point. Values go
    through a signed log (see _signed_log).
    """
    raw = [record_row[name] for name in feature_names(access_level)]
    return _signed_log(np.array([np.nan if v is None else v for v in raw], dtype=np.float64))


def reference_ranges(fit_clean_runs: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-layer normal ranges over fault-free runs in the fitting set.

    "min"/"max" are (a_l, b_l) for the range check and the LM "reference" prompt.
    "quantile_low"/"quantile_high" are the quantile monitor's bounds: the 0.1% quantile
    of per-run mins and the 99.9% quantile of per-run maxes.

    Raises on no runs or on non-finite values: a clean run should never have them.
    """
    if len(fit_clean_runs) == 0:
        raise ValueError("no clean runs to compute ranges from")
    ranges = {}
    for point in MONITORED_POINTS:
        if fit_clean_runs[f"{point}_nonfinite"].sum() > 0:
            raise ValueError(f"clean runs have non-finite values at {point}")
        run_mins = fit_clean_runs[f"{point}_min"].to_numpy(dtype=float)
        run_maxes = fit_clean_runs[f"{point}_max"].to_numpy(dtype=float)
        ranges[point] = {
            "min": float(run_mins.min()),
            "max": float(run_maxes.max()),
            "quantile_low": float(np.quantile(run_mins, QUANTILE_LOW)),
            "quantile_high": float(np.quantile(run_maxes, QUANTILE_HIGH)),
        }
    return ranges
