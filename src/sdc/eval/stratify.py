"""Breakdowns for RQ3 (fault site, bit group, layer) and RQ4 ablations."""

import numpy as np
import pandas as pd

from sdc.eval.bootstrap import bootstrap_ci
from sdc.eval.metrics import auroc, recall_at_threshold

STRATA = ["fault_site", "bit_group", "layer"]

ABLATIONS = {
    "reasoning": ("lm_generative", "lm_single_pass"),
    "decision_head": ("lm_probe", "learned"),
    "access": ("blackbox", "whitebox"),
    "context": ("zero_shot", "reference"),
}


def _auroc(df: pd.DataFrame) -> float:
    return auroc(df["label"] == "sdc", df["score"].to_numpy())


def stratified_metrics(scored_runs: pd.DataFrame, by: str, n_resamples: int = 1000,
                       ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """AUROC and recall@threshold (with CIs) per detector per stratum.

    `scored_runs`: one row per (run, detector) with name, score, threshold, label
    (sdc / masked; crashes are dropped), image_id and the `by` column. A stratum with no
    SDC gets NaN metrics: it says how many runs there were, not how well anything did.
    """
    runs = scored_runs[scored_runs["label"] != "crash"]
    rows = []
    for (name, stratum), group in runs.groupby(["name", by], sort=True):
        threshold = group["threshold"].iloc[0]
        n_sdc = int((group["label"] == "sdc").sum())
        a = r = (float("nan"),) * 3
        if n_sdc and n_sdc < len(group):
            a = bootstrap_ci(group, _auroc, n_resamples, ci, seed)
            r = bootstrap_ci(group, lambda df: recall_at_threshold(
                df["label"] == "sdc", df["score"].to_numpy(), threshold), n_resamples, ci, seed)
        rows.append({"detector": name, by: stratum, "n_sdc": n_sdc,
                     "n_masked": int((group["label"] == "masked").sum()),
                     "auroc": a[0], "auroc_low": a[1], "auroc_high": a[2],
                     "recall": r[0], "recall_low": r[1], "recall_high": r[2]})
    return pd.DataFrame(rows)


def ablation_table(scored_runs: pd.DataFrame, ablation: str) -> pd.DataFrame:
    """Side-by-side comparison for one entry of ABLATIONS. TODO: implement once the LM
    detectors' per-run scores are merged into the same table."""
    raise NotImplementedError
