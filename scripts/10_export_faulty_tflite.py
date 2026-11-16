"""Export one faulty FP32 LiteRT (.tflite) model per selected weight fault, for the Pi.

Action item: proposal §6.1 (weight faults injected in PyTorch, converted to LiteRT, run on
the Pi CPU)

Picks weight faults from the PyTorch campaign (default: every crash, plus a random sample
of SDC and masked runs from the eval split), flips each one's bit with weight_fault,
converts the faulty model with convert_fp32, and writes:

    <out-dir>/models/<run_id>.tflite   one faulty model per fault (about 43 MB each)
    <out-dir>/manifest.csv             one row per exported model: the fault, the PyTorch
                                       result, the file's sha256, and the Mac LiteRT check
    <out-dir>/export.json              the selection settings, to stop a mixed resume

Each model is checked on the Mac before moving on: LiteRT runs the fault's own image and
its label (masked / sdc / crash) is saved next to the PyTorch one. Skip with --no-check.

Stop at any time with Ctrl-C and run the same command again to continue. Needs the
LiteRT environment (pip install -e ".[tflite]"), for example:

    ~/.venvs/lm-sdc-tflite/bin/python scripts/10_export_faulty_tflite.py \\
        --out-dir "/Volumes/NO NAME/lm-sdc"
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sdc.config import load_config
from sdc.data.eurosat import load_splits, load_test_split, verify_splits
from sdc.injection.campaign import current_git_commit, weights_fingerprint
from sdc.injection.faults import fault_from_record, weight_fault
from sdc.injection.labeling import label_run
from sdc.model import resnet18
from sdc.model.tflite import convert_fp32, load_interpreter, run_tflite
from sdc.records.io import load_runs

BYTES_PER_MODEL = 45 * 1024**2     # an FP32 ResNet-18 .tflite is about 43 MB
SPACE_MARGIN = 500 * 1024**2

MANIFEST_COLUMNS = [
    "run_id", "file", "size_bytes", "sha256",
    "split", "plan_idx", "fault_number", "image_id", "image_path", "true_class",
    "layer", "element", "bit_position", "bit_group", "original_value", "flipped_value",
    "clean_pred", "pytorch_label", "pytorch_faulty_pred",
    "mac_litert_label", "mac_litert_pred", "mac_max_logit_diff", "export_time_s",
]


def select_faults(runs: pd.DataFrame, n_crash: int | None, n_sdc: int | None,
                  n_masked: int | None, seed: int) -> pd.DataFrame:
    """Pick weight-fault runs by label. None means all runs with that label.

    Runs with the same fault (layer, element, bit) would give identical models, so only the
    first is kept.
    """
    weight = runs[runs["fault_site"] == "weight"].copy()
    weight["fault_key"] = [
        (layer, tuple(element), bit)
        for layer, element, bit in zip(weight["layer"], weight["element"], weight["bit_position"])
    ]
    weight = weight.sort_values("plan_idx").drop_duplicates("fault_key")
    rng = np.random.default_rng([seed, 10])
    picked = []
    for label, count in (("crash", n_crash), ("sdc", n_sdc), ("masked", n_masked)):
        group = weight[weight["label"] == label]
        if count is not None and count < len(group):
            group = group.iloc[np.sort(rng.choice(len(group), size=count, replace=False))]
        picked.append(group)
    return pd.concat(picked).sort_values("plan_idx").drop(columns="fault_key")


def check_export_settings(out_dir: Path, settings: dict) -> None:
    path = out_dir / "export.json"
    if path.exists():
        saved = json.loads(path.read_text())
        changed = sorted(key for key in settings if saved.get(key) != settings[key])
        if changed:
            sys.exit(f"{out_dir} holds a different export (changed: {', '.join(changed)}); "
                     "use the same settings or a new --out-dir")
        return
    path.write_text(json.dumps(settings, indent=2))


def done_run_ids(manifest_path: Path, models_dir: Path) -> set[str]:
    """Runs already in the manifest whose model file is still there."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)
                if (models_dir / row["file"]).exists()}


def sha256_of(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_space(out_dir: Path, models_left: int) -> None:
    needed = models_left * BYTES_PER_MODEL + SPACE_MARGIN
    free = shutil.disk_usage(out_dir).free
    if free < needed:
        sys.exit(f"{out_dir} has {free / 1024**3:.1f} GB free but {models_left} models need "
                 f"about {needed / 1024**3:.1f} GB: free up space or use --count")


def main(args) -> None:
    config = load_config(args.config)
    paths, data = config["paths"], config["data"]
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    checkpoint_hash = resnet18.checkpoint_sha256(paths["checkpoint"])
    fault_dir = Path(paths["fault_runs"]) / args.split
    runs = load_runs(fault_dir)
    if set(runs["checkpoint_sha256"]) != {checkpoint_hash}:
        sys.exit(f"{fault_dir} was not made with {paths['checkpoint']}")
    selected = select_faults(runs, args.crash, args.sdc, args.masked, config["seed"])
    counts = selected["label"].value_counts().to_dict()
    campaign = json.loads((fault_dir / "campaign.json").read_text())

    check_export_settings(out_dir, {
        "split": args.split,
        "seed": config["seed"],
        "requested": {"crash": args.crash, "sdc": args.sdc, "masked": args.masked},
        "selected": {label: int(counts.get(label, 0)) for label in ("crash", "sdc", "masked")},
        "run_ids_sha256": hashlib.sha256(",".join(selected["run_id"]).encode()).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
        "campaign_plan_hash": campaign["plan_hash"],
        "model_format": "tflite_fp32",
    })

    done = done_run_ids(manifest_path, models_dir)
    todo = selected[~selected["run_id"].isin(done)]
    if args.count is not None:
        todo = todo.iloc[:args.count]
    print(f"selected {len(selected)} weight faults {counts}; {len(done)} already exported, "
          f"exporting {len(todo)} now to {models_dir}")
    if todo.empty:
        return
    check_space(out_dir, len(todo))

    splits = load_splits(paths["splits"])
    use_indices = splits["split_source"] == "test_indices"
    test_ds = load_test_split(
        paths["eurosat_dir"], data["split_fractions"], data["split_seed"],
        indices_path=paths["test_indices"] if use_indices else None,
    )
    verify_splits(splits, test_ds, checkpoint_hash)
    model = resnet18.load_checkpoint(paths["checkpoint"], "cpu")
    fingerprint = weights_fingerprint(model)
    git_commit = current_git_commit()

    new_file = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=MANIFEST_COLUMNS + ["git_commit"])
        if new_file:
            writer.writeheader()
        for n, row in enumerate(todo.itertuples(index=False), start=1):
            start = time.perf_counter()
            row = row._asdict()
            fault = fault_from_record(row)
            name = f"{row['run_id']}.tflite"
            part = models_dir / f"{row['run_id']}.part.tflite"

            with weight_fault(model, fault) as values:
                if not np.isclose(values.flipped, row["flipped_value"], equal_nan=True, rtol=0, atol=0):
                    raise RuntimeError(f"{row['run_id']}: flipped value {values.flipped} does not "
                                       f"match the campaign's {row['flipped_value']}")
                convert_fp32(model, str(part))
            if weights_fingerprint(model) != fingerprint:
                raise RuntimeError(f"{row['run_id']}: weights were not restored")
            os.replace(part, models_dir / name)

            check = {"mac_litert_label": None, "mac_litert_pred": None, "mac_max_logit_diff": None}
            if not args.no_check:
                image = test_ds[int(row["image_id"])][0].unsqueeze(0).numpy()
                logits = run_tflite(load_interpreter(models_dir / name), image)
                label = label_run(int(row["clean_pred"]), torch.from_numpy(logits), crashed=False)
                pytorch_logits = np.asarray(row["faulty_logits"], dtype=np.float64) \
                    if row["faulty_logits"] is not None else None
                check = {
                    "mac_litert_label": label.value,
                    "mac_litert_pred": None if label.value == "crash" else int(logits.argmax()),
                    "mac_max_logit_diff": float(np.max(np.abs(logits.flatten() - pytorch_logits)))
                        if pytorch_logits is not None and label.value != "crash" else None,
                }

            writer.writerow({
                "run_id": row["run_id"],
                "file": name,
                "size_bytes": (models_dir / name).stat().st_size,
                "sha256": sha256_of(models_dir / name),
                "split": row["split"],
                "plan_idx": row["plan_idx"],
                "fault_number": (row["plan_idx"] // 6) % campaign["injections_per_image"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "true_class": row["true_class"],
                "layer": row["layer"],
                "element": json.dumps([int(i) for i in row["element"]]),
                "bit_position": row["bit_position"],
                "bit_group": row["bit_group"],
                "original_value": row["original_value"],
                "flipped_value": row["flipped_value"],
                "clean_pred": row["clean_pred"],
                "pytorch_label": row["label"],
                "pytorch_faulty_pred": None if pd.isna(row["faulty_pred"]) else int(row["faulty_pred"]),
                **check,
                "export_time_s": round(time.perf_counter() - start, 2),
                "git_commit": git_commit,
            })
            manifest.flush()
            match = "" if args.no_check else \
                f" | mac litert {check['mac_litert_label']}" + \
                ("" if check["mac_litert_label"] == row["label"] else "  << DIFFERENT")
            print(f"[{n}/{len(todo)}] {name}: {row['layer']} bit {row['bit_position']} | "
                  f"pytorch {row['label']}{match}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="for example \"/Volumes/NO NAME/lm-sdc\"")
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--split", choices=["fit", "eval"], default="eval")
    parser.add_argument("--crash", type=int, default=None, help="crash runs to export (default: all)")
    parser.add_argument("--sdc", type=int, default=150, help="SDC runs to export")
    parser.add_argument("--masked", type=int, default=120, help="masked runs to export")
    parser.add_argument("--count", type=int, default=None, help="export at most this many now")
    parser.add_argument("--no-check", action="store_true", help="skip the Mac LiteRT check")
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        sys.exit("interrupted: run the same command again to resume")
