"""Fit classical detectors on the fitting set, freeze thresholds, score both sets.

Action item: Part I.3b Implement the classical detectors

Detectors (higher score = more likely SDC):
    sanity      black-box and white-box: 1 if anything is NaN/inf (or the run crashed)
    confidence  black-box: output entropy
    range       white-box: proposal Eq. 1 with the fault-free min/max per layer
    quantile    white-box: Eq. 1 with the 0.1% / 99.9% quantile bounds
    learned     black-box and white-box: logistic regression on the observation, the same
                classifier as the LM probe
    boosted     black-box and white-box: gradient-boosted trees on the same features
                (non-linear baseline)

Every detector is fitted, and its threshold frozen at target_fpr on masked runs, using the
fitting split only (crashes excluded). A supervised detector's threshold comes from
out-of-fold scores (5 folds grouped by image), since scores on its own training runs are too
clean; scores_fit.parquet keeps those in `fit_score` for later threshold choices. With
--fit-subset, the trained detectors (learned, boosted) are also trained on just those fit
runs (variant "subset"), for a like-for-like comparison with an LM probe trained on the
same runs.

Writes <paths.detector_scores>/classical/:
    scores_fit.parquet, scores_eval.parquet   one row per (run, detector): score, alarm
    summary.csv                               eval metrics: SDC vs. masked, crashes excluded
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdc.config import load_config
from sdc.detectors.classical.boosted import BoostedDetector
from sdc.detectors.classical.confidence import ConfidenceCheck
from sdc.detectors.classical.learned import LearnedDetector, out_of_fold_scores
from sdc.detectors.classical.quantile import QuantileMonitor
from sdc.detectors.classical.range_check import RangeCheck
from sdc.detectors.classical.sanity import SanityCheck
from sdc.eval.metrics import auroc, fpr_at_threshold, recall_at_threshold, sdc_vs_masked
from sdc.records.io import load_runs

KEEP_COLUMNS = ["run_id", "plan_idx", "image_id", "label", "fault_site", "bit_group",
                "bit_position", "layer"]


def detectors() -> list:
    return [
        SanityCheck("blackbox"), SanityCheck("whitebox"), ConfidenceCheck("blackbox"),
        RangeCheck("whitebox"), QuantileMonitor("whitebox"),
        LearnedDetector("blackbox"), LearnedDetector("whitebox"),
        BoostedDetector("blackbox"), BoostedDetector("whitebox"),
    ]


def read_run_ids(path: str) -> list[str]:
    """A JSON list of run ids, or a JSON object with a "run_ids" list (an LM manifest)."""
    data = json.loads(Path(path).read_text())
    return data["run_ids"] if isinstance(data, dict) else data


def subset(runs: pd.DataFrame, run_ids: list[str], what: str) -> pd.DataFrame:
    missing = set(run_ids) - set(runs["run_id"])
    if missing:
        raise ValueError(f"{len(missing)} {what} run ids are not in the runs, e.g. {sorted(missing)[0]}")
    return runs[runs["run_id"].isin(set(run_ids))]


def metrics(runs: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    kept = sdc_vs_masked(runs.assign(score=scores))
    y, s = kept["is_sdc"].to_numpy(), kept["score"].to_numpy()
    in_range = kept["in_range"].to_numpy()
    hard = ~kept["is_sdc"].to_numpy() | in_range        # masked plus in-range SDCs
    return {
        "n_sdc": int(y.sum()), "n_masked": int((~y).sum()),
        "auroc": auroc(y, s),
        "recall": recall_at_threshold(y, s, threshold),
        "fpr": fpr_at_threshold(y, s, threshold),
        "n_sdc_in_range": int((y & in_range).sum()),
        "auroc_in_range": auroc(y[hard], s[hard]),
        "recall_in_range": recall_at_threshold(y[hard], s[hard], threshold),
    }


def main(args) -> None:
    config = load_config(args.config)
    paths = config["paths"]
    target_fpr = config["eval"]["target_fpr"]
    out_dir = Path(paths["detector_scores"]) / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranges = json.loads(Path(paths["reference_ranges"]).read_text())
    runs = {split: load_runs(Path(paths["fault_runs"]) / split) for split in ("fit", "eval")}
    for split, df in runs.items():
        if set(df["checkpoint_sha256"]) != {ranges["checkpoint_sha256"]}:
            raise ValueError(f"{split} runs and reference_ranges.json come from different checkpoints")

    # "In range": an SDC whose white-box stats all stay inside the fault-free min/max, so a
    # range check can't see it. Same definition as the LM detectors' sdc_in_range.
    range_check = RangeCheck().fit(None, ranges["points"])
    for df in runs.values():
        df["in_range"] = range_check.score(df) == 0

    fit_train = runs["fit"][runs["fit"]["label"] != "crash"]
    variants = [("full", fit_train)]
    if args.fit_subset:
        variants.append(("subset", sdc_vs_masked(subset(runs["fit"], read_run_ids(args.fit_subset), "fit"))
                         .drop(columns="is_sdc")))
    eval_sets = {"eval_full": runs["eval"]}
    if args.eval_subset:
        eval_sets["eval_subset"] = subset(runs["eval"], read_run_ids(args.eval_subset), "eval")

    scores = {split: [] for split in runs}
    summary = []
    for variant, train in variants:
        for detector in detectors():
            if variant == "subset" and not detector.supervised:
                continue
            detector.fit(train, ranges["points"])
            if detector.supervised:
                level = detector.access_level
                kind = type(detector)
                train_scores = out_of_fold_scores(lambda: kind(level), train)
            else:
                train_scores = detector.score(train)
            threshold = detector.freeze_threshold(train, target_fpr, scores=train_scores)
            name = f"{detector.name}_{detector.access_level}" + ("" if variant == "full" else "_subset")
            fit_fpr = fpr_at_threshold(train["label"] == "sdc", train_scores, threshold)
            for split, df in runs.items():
                s = detector.score(df)
                part = df[KEEP_COLUMNS].assign(
                    detector=detector.name, access_level=detector.access_level, variant=variant,
                    name=name, score=s, threshold=threshold, alarm=s > threshold)
                if split == "fit":
                    oof = pd.Series(train_scores, index=train["run_id"])
                    part["fit_score"] = df["run_id"].map(oof).fillna(pd.Series(s, index=df.index))
                scores[split].append(part)
                if split == "eval":
                    by_run = pd.Series(s, index=df["run_id"])
                    for set_name, eval_runs in eval_sets.items():
                        summary.append({
                            "detector": name, "eval_set": set_name, "threshold": threshold,
                            "fit_fpr": fit_fpr, "n_train": len(train),
                            **metrics(eval_runs, by_run.loc[eval_runs["run_id"]].to_numpy(), threshold),
                        })
            print(f"{name:28s} threshold {threshold:.4g}", flush=True)

    for split, parts in scores.items():
        pd.concat(parts, ignore_index=True).to_parquet(out_dir / f"scores_{split}.parquet", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    for set_name, group in table.groupby("eval_set", sort=False):
        first = group.iloc[0]
        print(f"\n{set_name}: {first['n_sdc']} SDC ({first['n_sdc_in_range']} in range) vs "
              f"{first['n_masked']} masked; threshold frozen at {target_fpr:.0%} FPR on fit masked runs")
        print(group[["detector", "auroc", "recall", "fpr", "auroc_in_range", "recall_in_range"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--fit-subset", default=None,
                        help="JSON run ids (or an LM manifest) to also train the trained detectors on")
    parser.add_argument("--eval-subset", default=None,
                        help="JSON run ids (or an LM manifest) to also report metrics on")
    main(parser.parse_args())
