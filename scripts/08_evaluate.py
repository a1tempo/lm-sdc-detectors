"""Evaluate the detectors: confidence intervals, breakdowns, data size, and classical vs. LM.

Action item: Part I.3d / Part III evaluation

Reads the classical scores from 06_score_classical.py (runs/scores/classical/) and the fault
runs. The task is SDC vs. masked (crashes excluded); thresholds are the ones frozen at
target_fpr on the fitting split. 95% intervals come from bootstrapping eval images.

Writes <paths.results>/evaluation/:
    overall.csv              each classical detector on the full eval split
    strata_<by>.csv          the same per fault_site, bit_group and layer
    data_size.csv            trained detectors (learned, boosted) on 1, 2, 4, 8, 10 faults per image
    lm_comparison.csv        classical and LM detectors on the LM eval subset (needs the
                             subset manifests, and the LM results via --lm-prefix)

LM comparison, following the LM long-run analysis so the rows are comparable: every detector
gets its threshold from the fit subset's masked runs (out-of-fold scores for trained
detectors), and AUROC is reported for the 31 in-range SDCs, the other SDCs, and both
reweighted to the full eval split's proportions (the subset keeps every in-range SDC but
only a sample of the others). LM single-pass scores come from <prefix>-{fit,eval}/results.csv;
the LM probe needs <prefix>-{fit,eval}/hidden.npz too and is skipped without it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sdc.config import load_config
from sdc.detectors.base import threshold_for_fpr
from sdc.detectors.classical.boosted import BoostedDetector
from sdc.detectors.classical.learned import LearnedDetector, out_of_fold_scores
from sdc.detectors.classical.range_check import RangeCheck
from sdc.eval.bootstrap import bootstrap_ci, image_resamples
from sdc.eval.metrics import (
    auroc,
    fpr_at_threshold,
    recall_at_threshold,
    weighted_auroc,
    weighted_recall,
)
from sdc.eval.stratify import STRATA, stratified_metrics
from sdc.records.io import load_runs

SITES_PER_INJECTION = 6        # 2 fault sites x 3 bit groups per (image, injection) in the plan
DATA_SIZES = [1, 2, 4, 8, 10]


def fmt(table: pd.DataFrame) -> str:
    return table.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def overall(scores: pd.DataFrame, n_resamples: int, ci: float) -> pd.DataFrame:
    """Each detector on the full eval split, with image-bootstrap intervals."""
    rows = []
    first = scores[scores["name"] == scores["name"].iloc[0]]
    resamples = image_resamples(first["image_id"].to_numpy(), n_resamples)
    for name, group in scores.groupby("name", sort=False):
        group = group.reset_index(drop=True)
        y = (group["label"] == "sdc").to_numpy()
        threshold = group["threshold"].iloc[0]
        a = bootstrap_ci(group, lambda d: auroc(d["label"] == "sdc", d["score"].to_numpy()),
                         ci=ci, resamples=resamples)
        r = bootstrap_ci(group, lambda d: recall_at_threshold(
            d["label"] == "sdc", d["score"].to_numpy(), threshold), ci=ci, resamples=resamples)
        hard = ~y | group["in_range"].to_numpy()
        rows.append({
            "detector": name, "n_sdc": int(y.sum()), "n_masked": int((~y).sum()),
            "auroc": a[0], "auroc_low": a[1], "auroc_high": a[2],
            "recall": r[0], "recall_low": r[1], "recall_high": r[2],
            "fpr": fpr_at_threshold(y, group["score"].to_numpy(), threshold),
            "auroc_in_range": auroc(y[hard], group["score"].to_numpy()[hard]),
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def data_size(fit: pd.DataFrame, ev: pd.DataFrame, target_fpr: float, n_resamples: int,
              ci: float) -> pd.DataFrame:
    """Trained detectors fitted on the first k faults per image (k = 1, 2, 4, 8, 10).

    The plan is deterministic, so the first k faults per image are exactly what a campaign
    with injections_per_image = k would have produced.
    """
    rows = []
    resamples = image_resamples(ev["image_id"].to_numpy(), n_resamples)
    y = (ev["label"] == "sdc").to_numpy()
    hard = ~y | ev["in_range"].to_numpy()
    for k in DATA_SIZES:
        train = fit[fit["fault_number"] < k]
        for kind, level in [(k_, l_) for k_ in (LearnedDetector, BoostedDetector)
                            for l_ in ("blackbox", "whitebox")]:
            detector = kind(level).fit(train)
            threshold = detector.freeze_threshold(
                train, target_fpr, scores=out_of_fold_scores(lambda: kind(level), train))
            s = detector.score(ev)
            scored = ev[["image_id", "label"]].assign(score=s)
            a = bootstrap_ci(scored, lambda d: auroc(d["label"] == "sdc", d["score"].to_numpy()),
                             ci=ci, resamples=resamples)
            rows.append({
                "faults_per_image": k, "detector": f"{detector.name}_{level}",
                "n_train_sdc": int((train["label"] == "sdc").sum()),
                "n_train_masked": int((train["label"] == "masked").sum()),
                "auroc": a[0], "auroc_low": a[1], "auroc_high": a[2],
                "recall": recall_at_threshold(y, s, threshold),
                "fpr": fpr_at_threshold(y, s, threshold),
                "auroc_in_range": auroc(y[hard], s[hard]),
            })
            print(f"  k={k:2d} {detector.name}_{level}: auroc {a[0]:.3f}", flush=True)
    return pd.DataFrame(rows)


def read_manifest(path: str) -> dict:
    manifest = json.loads(Path(path).read_text())
    for key in ("run_ids", "population", "chosen"):
        if key not in manifest:
            raise ValueError(f"{path} has no {key!r}: pass an LM subset manifest")
    return manifest


def stratum(runs: pd.DataFrame) -> np.ndarray:
    return np.where(runs["label"] == "masked", "masked",
                    np.where(runs["in_range"], "sdc_in_range", "sdc_out_of_range"))


def probe_oof(features: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return cross_val_predict(model, features, y, groups=groups, cv=GroupKFold(5),
                             method="predict_proba")[:, 1]


def lm_comparison(fit_runs, ev_runs, fit_scores, ev_scores, fit_manifest, ev_manifest,
                  lm_prefix: str | None, target_fpr: float, n_resamples: int, ci: float):
    fit_sub = fit_runs.set_index("run_id").loc[fit_manifest["run_ids"]].reset_index()
    ev_sub = ev_runs.set_index("run_id").loc[ev_manifest["run_ids"]].reset_index()
    for name, sub, manifest in (("fit", fit_sub, fit_manifest), ("eval", ev_sub, ev_manifest)):
        counts = pd.Series(stratum(sub)).value_counts().to_dict()
        if counts != manifest["chosen"]:
            raise ValueError(f"{name} subset strata {counts} != manifest {manifest['chosen']}")
    ev_strata = stratum(ev_sub)
    weights = np.array([ev_manifest["population"][s] / ev_manifest["chosen"][s] if s != "masked" else 1.0
                        for s in ev_strata])
    fit_y = (fit_sub["label"] == "sdc").to_numpy()
    ev_y = (ev_sub["label"] == "sdc").to_numpy()
    fit_masked = ~fit_y
    resamples = image_resamples(ev_sub["image_id"].to_numpy(), n_resamples)

    # (name, trained on, scores used for the threshold, threshold-or-None, eval scores)
    detectors = []
    for name, group in fit_scores.groupby("name", sort=False):
        fit_s = group.set_index("run_id").loc[fit_sub["run_id"]]
        ev_s = ev_scores[ev_scores["name"] == name].set_index("run_id").loc[ev_sub["run_id"], "score"]
        variant = fit_s["variant"].iloc[0]
        if fit_s["detector"].iloc[0] in ("learned", "boosted") and variant == "full":
            # Trained on the whole fit split: keep the threshold frozen there.
            detectors.append((name, "all fit runs", None, fit_s["threshold"].iloc[0], ev_s.to_numpy()))
        else:
            trained = "fit subset" if variant == "subset" else "none"
            detectors.append((name, trained, fit_s["fit_score"].to_numpy(), None, ev_s.to_numpy()))

    if lm_prefix:
        for split in ("fit", "eval"):
            if not Path(f"{lm_prefix}-{split}/results.csv").exists():
                raise FileNotFoundError(f"{lm_prefix}-{split}/results.csv")
        fit_res = pd.read_csv(f"{lm_prefix}-fit/results.csv")
        ev_res = pd.read_csv(f"{lm_prefix}-eval/results.csv")
        hidden = {split: np.load(f"{lm_prefix}-{split}/hidden.npz")
                  for split in ("fit", "eval") if Path(f"{lm_prefix}-{split}/hidden.npz").exists()}
        if len(hidden) != 2:
            print("  LM hidden.npz not found for both splits: skipping the LM probe")
        for level in ("blackbox", "whitebox"):
            for cond in ("zero_shot", "reference"):
                def lm(res, sub):
                    r = res[(res["access_level"] == level) & (res["condition"] == cond)]
                    return r.set_index("run_id").loc[sub["run_id"], "score"].to_numpy()
                detectors.append((f"lm_single_pass_{level}_{cond}", "none",
                                  lm(fit_res, fit_sub), None, lm(ev_res, ev_sub)))
                if len(hidden) == 2:
                    h_fit = np.stack([hidden["fit"][f"{i}|{level}|{cond}"] for i in fit_sub["run_id"]])
                    h_ev = np.stack([hidden["eval"][f"{i}|{level}|{cond}"] for i in ev_sub["run_id"]])
                    oof = probe_oof(h_fit, fit_y, fit_sub["image_id"].to_numpy())
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
                    ev_p = model.fit(h_fit, fit_y).predict_proba(h_ev)[:, 1]
                    detectors.append((f"lm_probe_{level}_{cond}", "fit subset", oof, None, ev_p))

    rows = []
    in_range_keep = ev_strata != "sdc_out_of_range"
    other_keep = ev_strata != "sdc_in_range"
    for name, trained, fit_s, threshold, ev_s in detectors:
        if threshold is None:
            threshold = threshold_for_fpr(fit_s[fit_masked], target_fpr)
        frame = ev_sub[["image_id"]].assign(y=ev_y, s=ev_s, w=weights)
        point, low, high = bootstrap_ci(
            frame, lambda d: weighted_auroc(d["y"], d["s"], d["w"]), ci=ci, resamples=resamples)
        rows.append({
            "detector": name, "trained_on": trained,
            "auroc_in_range": auroc(ev_y[in_range_keep], ev_s[in_range_keep]),
            "auroc_other": auroc(ev_y[other_keep], ev_s[other_keep]),
            "auroc_weighted": point, "auroc_low": low, "auroc_high": high,
            "recall_weighted": weighted_recall(ev_y, ev_s, threshold, weights),
            "fpr": fpr_at_threshold(ev_y, ev_s, threshold),
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def main(args) -> None:
    config = load_config(args.config)
    paths, ev_cfg = config["paths"], config["eval"]
    n_resamples, ci, target_fpr = ev_cfg["bootstrap_resamples"], ev_cfg["ci"], ev_cfg["target_fpr"]
    if args.resamples is not None:
        n_resamples = args.resamples
    out_dir = Path(paths["results"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    ranges = json.loads(Path(paths["reference_ranges"]).read_text())
    range_check = RangeCheck().fit(None, ranges["points"])
    runs = {}
    for split in ("fit", "eval"):
        df = load_runs(Path(paths["fault_runs"]) / split)
        df = df[df["label"] != "crash"].reset_index(drop=True)
        df["in_range"] = range_check.score(df) == 0
        campaign = json.loads((Path(paths["fault_runs"]) / split / "campaign.json").read_text())
        plan_idx = df["plan_idx"].to_numpy(dtype=np.int64)
        df["fault_number"] = (plan_idx // SITES_PER_INJECTION) % campaign["injections_per_image"]
        runs[split] = df

    score_dir = Path(paths["detector_scores"]) / "classical"
    scores = {split: pd.read_parquet(score_dir / f"scores_{split}.parquet") for split in ("fit", "eval")}
    if "fit_score" not in scores["fit"]:
        raise ValueError("rerun scripts/06_score_classical.py: its scores predate out-of-fold thresholds")
    for split in scores:
        scores[split] = scores[split][scores[split]["label"] != "crash"]
    ev = scores["eval"].merge(runs["eval"][["run_id", "in_range"]], on="run_id")
    ev_full = ev[ev["variant"] == "full"]

    print(f"overall: {ev_full['run_id'].nunique()} eval runs, {n_resamples} image resamples")
    table = overall(ev_full, n_resamples, ci)
    table.to_csv(out_dir / "overall.csv", index=False)
    print(fmt(table[["detector", "auroc", "auroc_low", "auroc_high", "recall", "recall_low",
                     "recall_high", "fpr", "auroc_in_range"]]))

    for by in STRATA:
        print(f"\nby {by} ...", flush=True)
        strata = stratified_metrics(ev_full, by, n_resamples if by != "layer" else min(n_resamples, 200), ci)
        strata.to_csv(out_dir / f"strata_{by}.csv", index=False)
        if by != "layer":
            print(fmt(strata[["detector", by, "n_sdc", "n_masked", "auroc", "auroc_low",
                              "auroc_high", "recall"]]))

    print("\ndata size: trained detectors on k faults per image", flush=True)
    sizes = data_size(runs["fit"], runs["eval"], target_fpr, n_resamples, ci)
    sizes.to_csv(out_dir / "data_size.csv", index=False)
    print(fmt(sizes))

    if args.fit_subset and args.eval_subset:
        print("\nclassical vs. LM on the LM eval subset", flush=True)
        comparison = lm_comparison(
            runs["fit"], runs["eval"], scores["fit"], scores["eval"],
            read_manifest(args.fit_subset), read_manifest(args.eval_subset),
            args.lm_prefix, target_fpr, n_resamples, ci)
        comparison.to_csv(out_dir / "lm_comparison.csv", index=False)
        print(fmt(comparison))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--fit-subset", default=None, help="LM fit subset manifest (JSON)")
    parser.add_argument("--eval-subset", default=None, help="LM eval subset manifest (JSON)")
    parser.add_argument("--lm-prefix", default=None,
                        help="LM results prefix: <prefix>-fit/results.csv, <prefix>-eval/results.csv "
                             "(and hidden.npz for the probe)")
    parser.add_argument("--resamples", type=int, default=None,
                        help="bootstrap resamples (default: eval.bootstrap_resamples)")
    main(parser.parse_args())
