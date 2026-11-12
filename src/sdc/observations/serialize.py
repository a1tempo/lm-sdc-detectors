"""Serialize o_i as plain text for the language-model detectors (proposal §6.2).

Example white-box line:
    layer3: min=0 max=5.727 mean=0.4076 std=0.3552 nonfinite=0
followed by the output statistics.

`observation_columns` is the single list of what a detector may see. The serializer
reads a row only through it, so label, fault, and clean-run columns can't reach a prompt.
"""

import math

from sdc.model.monitor import MONITORED_POINTS

BLACKBOX_COLUMNS = ["top1_prob", "entropy"]
LAYER_STATS = ["min", "max", "mean", "std", "nonfinite"]


def observation_columns(access_level: str) -> list[str]:
    """Columns of a run record that a detector at `access_level` is allowed to see."""
    if access_level == "blackbox":
        return list(BLACKBOX_COLUMNS)
    if access_level == "whitebox":
        return BLACKBOX_COLUMNS + [f"{p}_{s}" for p in MONITORED_POINTS for s in LAYER_STATS]
    raise ValueError(f"access_level must be 'blackbox' or 'whitebox', got {access_level!r}")


def fmt(x: float) -> str:
    """One fixed number format for every prompt: 4 significant digits.

    Covers 0.4076 and 1.103e+38 alike. -0.0 prints as "0".
    """
    x = float(x)
    if x == 0 or math.isnan(x):
        return "0" if x == 0 else "nan"
    return f"{x:.4g}"


def serialize_observation(record_row: dict, access_level: str) -> str:
    """One line per monitored point (white-box only), then top-1 prob and entropy."""
    obs = {c: record_row[c] for c in observation_columns(access_level)}
    lines = []
    if access_level == "whitebox":
        for p in MONITORED_POINTS:
            lines.append(
                f"{p}: min={fmt(obs[f'{p}_min'])} max={fmt(obs[f'{p}_max'])} "
                f"mean={fmt(obs[f'{p}_mean'])} std={fmt(obs[f'{p}_std'])} "
                f"nonfinite={int(obs[f'{p}_nonfinite'])}"
            )
    lines.append(f"output: top1_prob={fmt(obs['top1_prob'])} entropy={fmt(obs['entropy'])}")
    return "\n".join(lines)


def output_ranges(fit_clean_runs) -> dict[str, dict[str, float]]:
    """Normal [min, max] of top1_prob and entropy over clean fitting runs.

    The black-box counterpart of runs/reference_ranges.json, from the same clean fit
    table and the same min/max rule. Fit data only, never eval.
    """
    return {
        col: {"min": float(fit_clean_runs[col].to_numpy(dtype=float).min()),
              "max": float(fit_clean_runs[col].to_numpy(dtype=float).max())}
        for col in BLACKBOX_COLUMNS
    }


def serialize_reference(layer_ranges: dict[str, dict[str, float]],
                        out_ranges: dict[str, dict[str, float]], access_level: str) -> str:
    """Normal ranges from fault-free fitting runs, for the "reference" condition.

    Mirrors serialize_observation: white-box gets one line per monitored point
    (a_l, b_l from runs/reference_ranges.json "points"), and both levels get the
    output line (from output_ranges).
    """
    observation_columns(access_level)   # validates access_level
    lines = []
    if access_level == "whitebox":
        lines = [f"{p}: min={fmt(layer_ranges[p]['min'])} max={fmt(layer_ranges[p]['max'])}"
                 for p in MONITORED_POINTS]
    for col in BLACKBOX_COLUMNS:
        lines.append(f"{col}: min={fmt(out_ranges[col]['min'])} max={fmt(out_ranges[col]['max'])}")
    return "\n".join(lines)
