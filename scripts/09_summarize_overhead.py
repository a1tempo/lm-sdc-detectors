"""Combine the Pi overhead run with the LM detectors' Pi timings and detection quality.

Action item: Part I.3c / Part III (cost vs. quality, E2)

Reads, all measured on the same fanless (throttled) Raspberry Pi 5:
    runs/overhead/pi/overhead.json            scripts/09_overhead.py (classical detectors)
    pi/lm/results/pi-20261118/*/results.csv   LM probe pass, 400 runs per prompt per split
    pi/lm/results/pi-20261117/pi-0.6b/        LM single-pass and generative (a few runs each)
    pi/lm/results/pi-20261117/pi-1.7b/        the same for Qwen3-1.7B
and detection quality from <paths.results>/evaluation/lm_comparison.csv.

Cost of a detector = the time it adds per run on top of the classifier (73 ms here). A
white-box detector also pays for the observation (min/max/mean/std at 6 points, computed
in the model); a black-box one only reads the logits. Writes
<paths.results>/overhead/overhead.csv and the figure <paths.results>/figures/overhead.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from sdc.config import REPO_ROOT, load_config
from sdc.eval import figures

LM_RESULTS = REPO_ROOT / "pi" / "lm" / "results"


def pi_phases(path: Path) -> tuple[dict, dict]:
    run = json.loads(path.read_text())
    timed = {p["phase"]: p["timing"]["p50_s"] * 1000 for p in run["phases"] if "timing" in p}
    return run, timed


def lm_medians() -> pd.DataFrame:
    """Median seconds per (model, pass, access level, condition) from Micah's Pi runs."""
    rows = []
    long_run = pd.concat([pd.read_csv(LM_RESULTS / "pi-20261118" / f"pi-0.6b-{s}" / "results.csv")
                          for s in ("fit", "eval")])
    for (level, cond), g in long_run.groupby(["access_level", "condition"]):
        rows.append({"model": "0.6b", "pass": "probe", "access_level": level, "condition": cond,
                     "median_s": g["seconds"].median(), "n": len(g)})
    for model in ("0.6b", "1.7b"):
        short = pd.read_csv(LM_RESULTS / "pi-20261117" / f"pi-{model}" / "results.csv")
        for (pass_, level, cond), g in short.groupby(["pass", "access_level", "condition"]):
            if model == "0.6b" and pass_ == "probe":
                continue                     # the long run measured this on 800 runs
            rows.append({"model": model, "pass": pass_, "access_level": level, "condition": cond,
                         "median_s": g["seconds"].median(), "n": len(g)})
    return pd.DataFrame(rows)


def lm_peak_mb(model: str) -> float:
    summary = json.loads((LM_RESULTS / "pi-20261117" / f"pi-{model}" / "summary.json").read_text())
    return summary["rss_peak_mb"]


def build_table(run: dict, timed: dict, lm: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    observe = timed["classifier_monitored"] - timed["classifier"]     # white-box observation
    logits = timed["entropy (black-box features)"]                    # softmax features
    auc = quality.set_index("detector")
    classical_mb = run["peak_rss_mb"]
    rows = [
        ("Range check", "classical", "whitebox", observe + timed["range check"], "range_whitebox"),
        ("Entropy check", "classical", "blackbox", logits, "confidence_blackbox"),
        ("Logistic regression", "classical", "blackbox", timed["learned_blackbox"], "learned_blackbox_subset"),
        ("Logistic regression", "classical", "whitebox", observe + timed["learned_whitebox"], "learned_whitebox_subset"),
        ("Boosted trees", "classical", "blackbox", timed["boosted_blackbox"], "boosted_blackbox_subset"),
        ("Boosted trees", "classical", "whitebox", observe + timed["boosted_whitebox"], "boosted_whitebox_subset"),
    ]
    table = [{"detector": name, "family": family, "access_level": level, "model": "",
              "condition": "", "added_ms": ms, "peak_memory_mb": classical_mb, "n_timed": 5000,
              "auroc": auc.loc[key, "auroc_weighted"] if key in auc.index else None,
              "auroc_in_range": auc.loc[key, "auroc_in_range"] if key in auc.index else None}
             for name, family, level, ms, key in rows]
    names = {"probe": "LM probe", "single": "LM single-pass", "gen": "LM generative"}
    keys = {"probe": "lm_probe", "single": "lm_single_pass"}
    for r in lm.to_dict("records"):
        # Detection quality exists for the 0.6B single-pass and probe (the long Pi run).
        key = (f"{keys[r['pass']]}_{r['access_level']}_{r['condition']}"
               if r["pass"] in keys and r["model"] == "0.6b" else None)
        extra = observe if r["access_level"] == "whitebox" else logits
        table.append({
            "detector": names[r["pass"]], "family": "lm", "access_level": r["access_level"],
            "model": f"Qwen3-{r['model'].upper()}", "condition": r["condition"],
            "added_ms": r["median_s"] * 1000 + extra, "peak_memory_mb": lm_peak_mb(r["model"]),
            "n_timed": r["n"],
            "auroc": auc.loc[key, "auroc_weighted"] if key in auc.index else None,
            "auroc_in_range": auc.loc[key, "auroc_in_range"] if key in auc.index else None,
        })
    out = pd.DataFrame(table)
    out["x_classifier"] = out["added_ms"] / timed["classifier"]
    return out


def main(config_path: str | None, pi_json: str) -> None:
    results = Path(load_config(config_path)["paths"]["results"])
    run, timed = pi_phases(Path(pi_json))
    quality = pd.read_csv(results / "evaluation" / "lm_comparison.csv")
    table = build_table(run, timed, lm_medians(), quality)
    out = results / "overhead"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "overhead.csv", index=False)
    figures.plot_overhead(table, timed["classifier"], results / "figures" / "overhead")

    pd.set_option("display.width", 200)
    print(f"{run['device']}, fanless: {run['temp_before_c']}-{run['temp_after_c']} C, "
          f"{run['throttled_before']} -> {run['throttled_after']}")
    print(f"classifier alone {timed['classifier']:.1f} ms; white-box observation "
          f"+{timed['classifier_monitored'] - timed['classifier']:.1f} ms\n")
    print(table[["detector", "access_level", "model", "condition", "added_ms", "x_classifier",
                 "peak_memory_mb", "n_timed", "auroc", "auroc_in_range"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out / 'overhead.csv'} and {results / 'figures' / 'overhead.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--pi-json", default=str(REPO_ROOT / "runs" / "overhead" / "pi" / "overhead.json"))
    args = parser.parse_args()
    main(args.config, args.pi_json)
