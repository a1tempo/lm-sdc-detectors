"""LAPTOP ONLY. Detection quality and cost from the long Pi run (pi_bench.py --passes probe
on prompts-fit.jsonl and prompts-eval.jsonl).

For each access level and prompt condition:
  single-pass  the LM's own score P(yes)/(P(yes)+P(no)), no training
  probe        the team's probe classifier (standardize, then logistic regression, C=1),
               trained on the fit subset's hidden states, scored on the eval subset
and, for context on the same runs (exploratory, not the team's classical detectors):
  range check  proposal Eq. 1 with the fit fault-free [min, max]        (white-box)
  entropy      the confidence check's score                              (black-box)
  raw LR       the same classifier as the probe on the raw observations  (2 or 32 numbers)
  raw trees    gradient-boosted trees on the same raw observations: a classifier that,
               unlike raw LR, can learn rules such as "entropy exactly 0 OR entropy high"
  entropy 2-sided  hand-made rule: entropy exactly 0 is most suspicious, otherwise
               higher entropy is more suspicious                         (black-box)

AUROC is reported three ways, because the subset holds ALL SDCs the range check misses
(31 per split) but only a sample of the others:
  in-range     the 31 hard SDCs vs the masked sample
  other        the other SDCs vs the masked sample
  weighted     the two combined in the full split's proportions (e.g. 31 : 736 for eval);
               this is the estimate of the full-split AUROC
Recall at 5% false positives: threshold = 95th percentile of the fit masked scores, then
the weighted share of eval SDCs above it. For trained detectors the fit scores are
out-of-fold (5 folds, grouped by image): a classifier's scores on its own training runs
are too clean to set a threshold from. 95% intervals: bootstrap over eval images.

Run from team/lm-sdc-detectors:
    ../../.venv/bin/python ../../pi-kit/analyze_long_run.py ../../pi-kit/results-from-pi-20261118/pi-0.6b
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sdc.observations.serialize import observation_columns
from sdc.records.io import load_runs

KIT = Path(__file__).resolve().parent
prefix = sys.argv[1]                      # e.g. .../pi-0.6b  ->  pi-0.6b-fit, pi-0.6b-eval
rng = np.random.default_rng(42)
POINTS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


def weights(runs, manifest):
    """SDC weight = (stratum size in the full split) / (stratum size in the subset)."""
    pop, chosen = manifest["population"], manifest["chosen"]
    w = runs["stratum"].map(lambda s: pop[s] / chosen[s]).to_numpy()
    return w


def aurocs(score, runs, w):
    y = (runs["label"] == "sdc").to_numpy()
    s = runs["stratum"].to_numpy()
    out = {}
    score = np.minimum(score, 1e300)      # roc_auc_score does not accept inf
    for name, keep in [("in-range", s != "sdc_out_of_range"), ("other", s != "sdc_in_range")]:
        out[name] = roc_auc_score(y[keep], score[keep])
    # Masked runs all get weight 1 relative to each other; SDC runs get stratum weights.
    out["weighted"] = roc_auc_score(y, score, sample_weight=np.where(y, w, 1.0))
    return out


def recall_at_5(fit_score, fit_runs, score, runs, w):
    thr = np.quantile(fit_score[(fit_runs["label"] == "masked").to_numpy()], 0.95)
    y = (runs["label"] == "sdc").to_numpy()
    flagged = score > thr
    return (np.sum(w[y] * flagged[y]) / np.sum(w[y]), flagged[~y].mean())


def range_score(runs, points):
    worst = np.zeros(len(runs))
    for p in POINTS:
        a, b = points[p]["min"], points[p]["max"]
        over = np.maximum(runs[f"{p}_max"].to_numpy() - b, a - runs[f"{p}_min"].to_numpy())
        worst = np.maximum(worst, np.maximum(0, over) / (b - a))
    return worst


def load(split):
    """One row per run: label, stratum, image, raw observations; plus per-condition scores
    and hidden states from the bench."""
    manifest = json.loads((KIT / f"prompts-{split}.manifest.json").read_text())
    runs = load_runs(f"runs/faults/{split}").set_index("run_id").loc[manifest["run_ids"]]
    # Same grouping as make_subset_prompts.py: an SDC is "in range" if the range check's
    # score is 0, i.e. no monitored point leaves the fit fault-free [min, max].
    hidden_from_range_check = range_score(runs, manifest["reference_ranges"]["points"]) == 0
    runs["stratum"] = np.where(runs["label"] == "masked", "masked",
                               np.where(hidden_from_range_check, "sdc_in_range", "sdc_out_of_range"))
    assert runs["stratum"].value_counts().to_dict() == manifest["chosen"]
    res = pd.read_csv(f"{prefix}-{split}/results.csv")
    hid = np.load(f"{prefix}-{split}/hidden.npz")
    return runs, res, hid, manifest


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def trees():
    return HistGradientBoostingClassifier(random_state=42)


def trained(make, X_fit, X_ev):
    """(out-of-fold scores on fit, scores on eval from a model trained on all of fit)."""
    oof = cross_val_predict(make(), X_fit, fit_y, groups=fit_runs["image_id"].to_numpy(),
                            cv=GroupKFold(5), method="predict_proba")[:, 1]
    return oof, make().fit(X_fit, fit_y).predict_proba(X_ev)[:, 1]


fit_runs, fit_res, fit_hid, fit_man = load("fit")
ev_runs, ev_res, ev_hid, ev_man = load("eval")
fit_w, ev_w = weights(fit_runs, fit_man), weights(ev_runs, ev_man)
fit_y = (fit_runs["label"] == "sdc").to_numpy()
print(f"fit subset {fit_man['chosen']}, {fit_runs['image_id'].nunique()} images")
print(f"eval subset {ev_man['chosen']}, {ev_runs['image_id'].nunique()} images; "
      f"full eval split {ev_man['population']}")

# Every detector as (name, fit scores, eval scores), aligned with fit_runs / ev_runs.
detectors = []
for level in ("blackbox", "whitebox"):
    for cond in ("zero_shot", "reference"):
        def lm_scores(res, runs):
            r = res[(res["access_level"] == level) & (res["condition"] == cond)]
            return r.set_index("run_id").loc[runs.index, "score"].to_numpy()
        detectors.append((f"single-pass {level} {cond}", lm_scores(fit_res, fit_runs),
                          lm_scores(ev_res, ev_runs)))
        H_fit = np.stack([fit_hid[f"{i}|{level}|{cond}"] for i in fit_runs.index])
        H_ev = np.stack([ev_hid[f"{i}|{level}|{cond}"] for i in ev_runs.index])
        detectors.append((f"probe       {level} {cond}", *trained(classifier, H_fit, H_ev)))
    cols = observation_columns(level)
    X_fit, X_ev = fit_runs[cols].to_numpy(float), ev_runs[cols].to_numpy(float)
    detectors.append((f"raw LR      {level}", *trained(classifier, X_fit, X_ev)))
    detectors.append((f"raw trees   {level}", *trained(trees, X_fit, X_ev)))
points = fit_man["reference_ranges"]["points"]
detectors.append(("range check whitebox", range_score(fit_runs, points), range_score(ev_runs, points)))
detectors.append(("entropy     blackbox", fit_runs["entropy"].to_numpy(), ev_runs["entropy"].to_numpy()))
two_sided = lambda runs: np.where(runs["entropy"] == 0, np.inf, runs["entropy"]).astype(float)
detectors.append(("entropy 2-sided blackbox", two_sided(fit_runs), two_sided(ev_runs)))
print("\nentropy exactly 0 (eval subset): " + "; ".join(
    f"{s} {(g['entropy'] == 0).sum()}/{len(g)}" for s, g in ev_runs.groupby("stratum")))

# Bootstrap over eval images: draw images with replacement, keep all their runs.
images = ev_runs["image_id"].to_numpy()
by_image = {i: np.flatnonzero(images == i) for i in np.unique(images)}
draws = [np.concatenate([by_image[i] for i in rng.choice(list(by_image), len(by_image))])
         for _ in range(1000)]
ev_y = (ev_runs["label"] == "sdc").to_numpy()

print(f"\n{'eval AUROC':38s} {'in-range':>8s} {'other':>7s} {'weighted':>8s}  {'95% interval':14s} "
      f"{'recall@5%':>9s} {'eval FPR':>8s}")
for name, f, e in detectors:
    e = np.minimum(e, 1e300)
    a = aurocs(e, ev_runs, ev_w)
    boot = [roc_auc_score(ev_y[d], e[d], sample_weight=np.where(ev_y[d], ev_w[d], 1.0)) for d in draws]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rec, fpr = recall_at_5(f, fit_runs, e, ev_runs, ev_w)
    print(f"{name:38s} {a['in-range']:8.3f} {a['other']:7.3f} {a['weighted']:8.3f}  "
          f"[{lo:.3f}, {hi:.3f}] {rec:9.3f} {fpr:8.3f}")

print("\nSingle-pass score by group (eval), median [min, max]:")
for level in ("blackbox", "whitebox"):
    for cond in ("zero_shot", "reference"):
        r = ev_res[(ev_res["access_level"] == level) & (ev_res["condition"] == cond)].set_index("run_id")
        r = r.loc[ev_runs.index].assign(stratum=ev_runs["stratum"])
        print(f"  {level} {cond}: " + "; ".join(
            f"{s} {g['score'].median():.3f} [{g['score'].min():.3f}, {g['score'].max():.3f}]"
            for s, g in r.groupby("stratum")))

print("\nCost on the Pi (probe pass = single-pass score + hidden state), both splits:")
allr = pd.concat([fit_res, ev_res])
for (level, cond), g in allr.groupby(["access_level", "condition"], sort=False):
    print(f"  {level:8s} {cond:9s} {g['prompt_tokens'].median():4.0f} tokens  median {g['seconds'].median():5.2f} s"
          f"  5th-95th pct {g['seconds'].quantile(.05):5.2f}-{g['seconds'].quantile(.95):5.2f} s")
per_run = allr.groupby("run_id")["seconds"].sum()
print(f"  all four prompts for one run: median {per_run.median():.1f} s; total {allr['seconds'].sum()/3600:.2f} h "
      f"for {len(per_run)} runs; temperature {allr['temp_c'].min():.1f}-{allr['temp_c'].max():.1f} C "
      f"(median {allr['temp_c'].median():.1f})")
