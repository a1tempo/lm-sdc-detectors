"""LAPTOP ONLY. Write LM prompts for a subset of one split, for the long Pi run.

Like make_prompts.py, but it picks the runs itself instead of using the 40 pilot runs:
  1. every SDC run the range check misses ("in range": at all six monitored points the
     run's min and max stay inside the fault-free fit range). These are the cases where
     an LM could add something over the classical range check, and there are few of them;
  2. a random sample of the other SDC runs, to reach --n-sdc in total;
  3. a random sample of --n-masked masked runs.
Crashes are left out (they are not part of the detection task).

Because step 1 over-represents the hard SDCs, each line carries a "stratum" field
(sdc_in_range / sdc_out_of_range / masked) and the manifest stores how many runs each
stratum has in the full split, so results can be reweighted to the true mix.

The reference ranges are the fit split's fault-free ranges for both splits (the same ones
the pilot used), so nothing from the eval split leaks into a prompt.

Run from team/lm-sdc-detectors:
    ../../.venv/bin/python ../../pi-kit/make_subset_prompts.py --split fit \
        --gguf ../../models/Qwen3-0.6B-Q4_K_M.gguf
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from sdc.detectors.lm.generative import GenerativeDetector
from sdc.detectors.lm.llama_client import QwenClient
from sdc.detectors.lm.single_pass import SinglePassDetector
from sdc.records.io import load_runs

LEVELS = ["blackbox", "whitebox"]
CONDITIONS = ["zero_shot", "reference"]


def in_range(runs: pd.DataFrame, points: dict) -> pd.Series:
    """True where every monitored point's min and max lie inside the fit range."""
    ok = pd.Series(True, index=runs.index)
    for point, r in points.items():
        ok &= (runs[f"{point}_min"] >= r["min"]) & (runs[f"{point}_max"] <= r["max"])
    return ok


def main(args):
    ranges = json.loads(Path("runs/scores/pilot/manifest.json").read_text())["reference_ranges"]
    runs = load_runs(f"runs/faults/{args.split}")
    runs = runs[runs["label"].isin(["sdc", "masked"])].copy()
    inside = in_range(runs, ranges["points"])
    runs["stratum"] = "masked"
    runs.loc[(runs["label"] == "sdc") & inside, "stratum"] = "sdc_in_range"
    runs.loc[(runs["label"] == "sdc") & ~inside, "stratum"] = "sdc_out_of_range"
    population = runs["stratum"].value_counts().to_dict()

    hard = runs[runs["stratum"] == "sdc_in_range"]
    easy = runs[runs["stratum"] == "sdc_out_of_range"].sample(
        n=args.n_sdc - len(hard), random_state=args.seed)
    masked = runs[runs["stratum"] == "masked"].sample(n=args.n_masked, random_state=args.seed)

    # Interleave SDC and masked, each in a seeded random order, so that a run that
    # stops early still has a balanced, random part of the subset.
    sdc = pd.concat([hard, easy]).sample(frac=1, random_state=args.seed).to_dict("records")
    masked = masked.sample(frac=1, random_state=args.seed).to_dict("records")
    ordered = [r for pair in zip(sdc, masked) for r in pair]

    client = QwenClient(args.gguf, n_ctx=2048, n_threads=4, flash_attn=False)  # chat template only
    lines = []
    for i, row in enumerate(ordered):
        for level in LEVELS:
            for cond in CONDITIONS:
                single = SinglePassDetector(client, cond, level).fit(None, ranges)
                think = GenerativeDetector(client, cond, level).fit(None, ranges)
                lines.append({"run_index": i, "run_id": row["run_id"], "label": row["label"],
                              "stratum": row["stratum"],
                              "access_level": level, "condition": cond,
                              "prompt_single": single.prompt(row),
                              "prompt_think": think.prompt(row)})
    out = Path(args.out or Path(__file__).parent / f"prompts-{args.split}.jsonl")
    out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    chosen = pd.Series([r["stratum"] for r in ordered]).value_counts().to_dict()
    manifest = {"split": args.split, "seed": args.seed, "population": population,
                "chosen": chosen, "run_ids": [r["run_id"] for r in ordered],
                "reference_ranges": ranges}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{args.split}: full split {population}; chosen {chosen}; "
          f"wrote {len(lines)} prompt pairs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["fit", "eval"])
    p.add_argument("--gguf", required=True)
    p.add_argument("--n-sdc", type=int, default=200)
    p.add_argument("--n-masked", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    main(p.parse_args())
