"""Score a subset of fault runs with the LM detectors, across access levels and prompt conditions.

Action item: Part II.2b Implement and score the LM detectors

The subset is all-or-a-sample of SDC runs plus a sample of masked runs from one split
(proposal §6.4: LM detectors may run on a stratified subset). Crashes are never scored.
Each (detector, access level, condition) goes to its own Parquet file in --out, written
only when complete. For "probe" this step only extracts hidden states (plus P(yes), P(no)
from the same pass) into probe_hidden__<level>__<condition>.npz; training the probe on
fit runs and scoring eval runs happens later, from those files. Stop at any time and rerun the same command: finished files are
skipped. The subset and runtime settings are saved in manifest.json; rerunning with
different ones is an error, so files from different settings can't mix.

Pilot example:
    python scripts/07_score_lm.py --gguf models/Qwen3-0.6B-Q4_K_M.gguf \\
        --split fit --n-sdc 20 --n-masked 20 --out runs/scores/pilot
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import llama_cpp
import numpy as np
import pandas as pd
import pyarrow as pa

from sdc.config import load_config
from sdc.detectors.lm.generative import GenerativeDetector
from sdc.detectors.lm.llama_client import QwenClient
from sdc.detectors.lm.probe import ProbeDetector
from sdc.detectors.lm.single_pass import SinglePassDetector
from sdc.injection.campaign import current_git_commit
from sdc.observations.clean import load_clean_runs
from sdc.observations.serialize import output_ranges
from sdc.records.io import load_runs, write_table_atomic

DETECTORS = {"single_pass": SinglePassDetector, "generative": GenerativeDetector,
             "probe": ProbeDetector}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_runs(runs: pd.DataFrame, n_sdc: int | None, n_masked: int, seed: int) -> pd.DataFrame:
    """A seeded sample of SDC and masked runs, sorted by run_id. n_sdc=None keeps every SDC."""
    sdc = runs[runs["label"] == "sdc"]
    masked = runs[runs["label"] == "masked"]
    if n_sdc is not None:
        sdc = sdc.sample(n=n_sdc, random_state=seed)
    masked = masked.sample(n=n_masked, random_state=seed)
    return pd.concat([sdc, masked]).sort_values("run_id").reset_index(drop=True)


def check_manifest(path: Path, manifest: dict) -> None:
    """Write the manifest, or check that an existing one matches (except bookkeeping)."""
    if path.exists():
        old = json.loads(path.read_text())
        keys = [k for k in manifest if k not in ("git_commit",)]
        diff = [k for k in keys if old.get(k) != manifest[k]]
        if diff:
            raise SystemExit(f"{path} differs in {diff}: use a new --out, or delete the old "
                             "directory by hand if you mean to replace it")
        return
    path.write_text(json.dumps(manifest, indent=2))


def save_hidden(path: Path, run_ids, hidden: np.ndarray, details: pd.DataFrame) -> None:
    """Write the .npz so it either exists completely or not at all."""
    temp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(temp, run_ids=np.asarray(run_ids, dtype=str), hidden=hidden.astype(np.float32),
             p_yes=details["p_yes"].to_numpy(), p_no=details["p_no"].to_numpy(),
             elapsed_s=details["elapsed_s"].to_numpy())
    temp.replace(path)


def summarize(details: pd.DataFrame) -> str:
    text = f"n={len(details)} "
    if "parsed" in details:
        text += f"parsed={details['parsed'].mean():.0%} "
    text += f"elapsed median={details['elapsed_s'].median():.2f}s max={details['elapsed_s'].max():.2f}s"
    if "n_gen_tokens" in details:
        text += f" gen tokens median={details['n_gen_tokens'].median():.0f} max={details['n_gen_tokens'].max()}"
    return text


def main(args) -> None:
    config = load_config(args.config)
    paths, lm = config["paths"], config["lm"]
    gguf = Path(args.gguf or paths["qwen_gguf"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = choose_runs(load_runs(Path(paths["fault_runs"]) / args.split),
                       args.n_sdc, args.n_masked, args.seed)
    ranges = {
        "points": json.loads(Path(paths["reference_ranges"]).read_text())["points"],
        "outputs": output_ranges(load_clean_runs(str(Path(paths["clean_runs"]) / "fit.parquet"))),
    }
    manifest = {
        "split": args.split, "seed": args.seed, "n_sdc": args.n_sdc, "n_masked": args.n_masked,
        "run_ids": runs["run_id"].tolist(),
        "gguf": gguf.name, "gguf_sha256": file_sha256(gguf),
        "llama_cpp_python": llama_cpp.__version__,
        "n_ctx": lm["n_ctx"], "n_threads": args.threads, "flash_attn": False,
        "max_tokens": args.max_tokens, "decoding": "greedy",
        "reference_ranges": ranges,
        "git_commit": current_git_commit(),
    }
    check_manifest(out / "manifest.json", manifest)
    counts = runs["label"].value_counts().to_dict()
    print(f"{args.split}: scoring {len(runs)} runs ({counts}) into {out}")

    client = QwenClient(str(gguf), n_ctx=lm["n_ctx"], n_threads=args.threads, flash_attn=False)
    for det_name in args.detectors:
        for level in args.access_levels or lm["access_levels"]:
            for condition in args.conditions or lm["prompt_conditions"]:
                path = out / f"{det_name}__{level}__{condition}.parquet"
                if det_name == "probe":
                    path = out / f"probe_hidden__{level}__{condition}.npz"
                if path.exists():
                    print(f"skip {path.name}: already done")
                    continue
                kwargs = {"max_tokens": args.max_tokens} if det_name == "generative" else {}
                det = DETECTORS[det_name](client, condition, level, **kwargs)
                if det_name == "probe":
                    det.set_reference(ranges)
                    hidden, details = det.extract(runs)
                    save_hidden(path, runs["run_id"], hidden, details)
                    print(f"{path.name}: hidden {hidden.shape}, {summarize(details)}", flush=True)
                    continue
                details = det.fit(None, ranges).score_details(runs)
                write_table_atomic(pa.Table.from_pandas(details, preserve_index=False), path)
                print(f"{path.name}: {summarize(details)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--gguf", default=None, help="model file; default paths.qwen_gguf")
    parser.add_argument("--split", choices=["fit", "eval"], default="fit")
    parser.add_argument("--n-sdc", type=int, default=None, help="default: every SDC run")
    parser.add_argument("--n-masked", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048, help="generative token budget")
    parser.add_argument("--detectors", nargs="+", choices=list(DETECTORS),
                        default=list(DETECTORS))
    parser.add_argument("--access-levels", nargs="+", choices=["blackbox", "whitebox"],
                        default=None, help="default: lm.access_levels in the config")
    parser.add_argument("--conditions", nargs="+", choices=["zero_shot", "reference"],
                        default=None, help="default: lm.prompt_conditions in the config")
    parser.add_argument("--out", required=True)
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        sys.exit("interrupted: run the same command again to resume")
