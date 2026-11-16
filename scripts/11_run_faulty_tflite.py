"""Run each exported faulty LiteRT model on its own image on the Pi CPU and label it.

Action item: proposal §6.1 (weight faults run as faulty FP32 LiteRT models on the Pi)

Reads the folder written by scripts/10_export_faulty_tflite.py (manifest.csv and
models/). First runs the clean model (paths.tflite_fp32) on every image in the manifest,
so each faulty run is labeled against the Pi's own fault-free prediction. Then, for each
faulty model: read the file, check its sha256, load it, run its image, and label the
result masked / sdc / crash. One row per model is appended to <out>/results.csv, with the
PyTorch label next to the Pi label.

Stop at any time with Ctrl-C and run the same command again to continue. Needs only
numpy, Pillow, PyYAML and ai-edge-litert (no PyTorch). On the Pi, for example:

    scripts/notify.sh python scripts/11_run_faulty_tflite.py \\
        --models-dir /media/$USER/fLiteRT/lm-sdc
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sdc.config import REPO_ROOT, load_config
from sdc.model.tflite import (
    blackbox_stats_np,
    label_logits,
    load_interpreter,
    preprocess_image,
    run_tflite,
)

RESULT_COLUMNS = [
    "run_id", "file", "sha256_ok", "image_id", "image_path", "layer", "element",
    "bit_position", "bit_group", "fault_number", "flipped_value",
    "pytorch_clean_pred", "pytorch_label", "pytorch_faulty_pred",
    "pi_clean_pred", "pi_label", "pi_crash_reason", "pi_pred", "same_as_pytorch",
    "top1_prob", "entropy", "logits",
    "load_time_s", "inference_time_s", "repeats", "threads",
    "device", "litert_version", "git_commit", "timestamp", "model_format",
]


def device_name() -> str:
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def litert_version() -> str:
    try:
        from importlib.metadata import version
        return version("ai-edge-litert")
    except Exception:
        return "unknown"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def timed_run(interpreter, x: np.ndarray, repeats: int):
    """Run `repeats` times; return the first run's logits and the median time."""
    times, logits = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        out = run_tflite(interpreter, x)
        times.append(time.perf_counter() - start)
        if logits is None:
            logits = out.flatten()
    return logits, statistics.median(times)


def clean_predictions(clean_model: Path, images: dict[int, Path], threads: int) -> dict[int, int]:
    interpreter = load_interpreter(clean_model, num_threads=threads)
    preds = {}
    for image_id, path in images.items():
        logits = run_tflite(interpreter, preprocess_image(path)).flatten()
        if not np.isfinite(logits).all():
            sys.exit(f"the clean model gave non-finite logits on {path}: check {clean_model}")
        preds[image_id] = int(np.argmax(logits))
    return preds


def summarize(results_path: Path) -> None:
    rows = read_csv(results_path)
    pairs = Counter((row["pytorch_label"], row["pi_label"]) for row in rows)
    same = sum(row["same_as_pytorch"] == "True" for row in rows)
    print(f"\n{len(rows)} models run; Pi label matches PyTorch for {same}")
    print("pytorch -> pi: " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(pairs.items())))
    bad = sum(row["sha256_ok"] != "True" for row in rows)
    if bad:
        print(f"WARNING: {bad} model files failed their sha256 check (sha256_ok=False)")
    times = [float(row["inference_time_s"]) for row in rows if row["inference_time_s"]]
    if times:
        print(f"median inference {statistics.median(times) * 1000:.0f} ms")


def main(args) -> None:
    config = load_config(args.config)
    models_root = Path(args.models_dir)
    manifest = read_csv(models_root / "manifest.csv")
    export = json.loads((models_root / "export.json").read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    data_dir = Path(config["paths"]["eurosat_dir"])
    images = {int(row["image_id"]): data_dir / row["image_path"] for row in manifest}
    missing = [str(path) for path in images.values() if not path.exists()]
    if missing:
        sys.exit(f"{len(missing)} images are missing, for example {missing[0]}")

    done = {row["run_id"] for row in read_csv(results_path)} if results_path.exists() else set()
    todo = [row for row in manifest if row["run_id"] not in done]
    if args.count is not None:
        todo = todo[:args.count]
    print(f"{len(manifest)} models in {models_root}; {len(done)} already run, running {len(todo)} now "
          f"({args.threads} threads, {args.repeats} timed runs each)")
    if not todo:
        summarize(results_path)
        return

    clean_model = Path(args.clean_model or config["paths"]["tflite_fp32"])
    pi_clean = clean_predictions(clean_model, images, args.threads)
    differ = sum(pi_clean[int(row["image_id"])] != int(row["clean_pred"]) for row in manifest)
    print(f"clean model {clean_model.name}: {len(pi_clean)} images, "
          f"{differ} manifest rows where the Pi's clean prediction differs from PyTorch's")

    static = {
        "repeats": args.repeats, "threads": args.threads, "device": device_name(),
        "litert_version": litert_version(), "git_commit": git_commit(),
        "model_format": export.get("model_format", "tflite_fp32"),
    }
    new_file = not results_path.exists()
    with open(results_path, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        for n, row in enumerate(todo, start=1):
            image_id = int(row["image_id"])
            content = (models_root / "models" / row["file"]).read_bytes()
            sha256_ok = hashlib.sha256(content).hexdigest() == row["sha256"]

            start = time.perf_counter()
            interpreter = load_interpreter(model_content=content, num_threads=args.threads)
            load_time = time.perf_counter() - start
            x = preprocess_image(images[image_id])
            logits, crash_reason = None, None
            try:
                logits, inference_time = timed_run(interpreter, x, args.repeats)
            except Exception as error:
                print(f"{row['run_id']} raised {error!r}", file=sys.stderr)
                inference_time, crash_reason = None, "exception"
            label = label_logits(pi_clean[image_id], logits)
            if label == "crash" and crash_reason is None:
                crash_reason = "nonfinite"
            stats = blackbox_stats_np(logits) if logits is not None else {"top1_prob": None, "entropy": None}
            del interpreter, content

            writer.writerow({
                "run_id": row["run_id"], "file": row["file"], "sha256_ok": sha256_ok,
                "image_id": image_id, "image_path": row["image_path"], "layer": row["layer"],
                "element": row["element"], "bit_position": row["bit_position"],
                "bit_group": row["bit_group"], "fault_number": row["fault_number"],
                "flipped_value": row["flipped_value"],
                "pytorch_clean_pred": row["clean_pred"], "pytorch_label": row["pytorch_label"],
                "pytorch_faulty_pred": row["pytorch_faulty_pred"],
                "pi_clean_pred": pi_clean[image_id], "pi_label": label,
                "pi_crash_reason": crash_reason,
                "pi_pred": None if label == "crash" else int(np.argmax(logits)),
                "same_as_pytorch": label == row["pytorch_label"],
                **stats,
                "logits": json.dumps([float(v) for v in logits]) if logits is not None else None,
                "load_time_s": round(load_time, 4),
                "inference_time_s": None if inference_time is None else round(inference_time, 5),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **static,
            })
            out.flush()
            flag = "" if label == row["pytorch_label"] else "  << DIFFERENT"
            flag += "" if sha256_ok else "  << BAD FILE"
            print(f"[{n}/{len(todo)}] {row['run_id']}: pytorch {row['pytorch_label']} | "
                  f"pi {label}{flag}", flush=True)

    summarize(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", required=True,
                        help="folder with manifest.csv and models/, e.g. /media/$USER/fLiteRT/lm-sdc")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "pi_tflite"),
                        help="where results.csv goes (default: runs/pi_tflite)")
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--clean-model", default=None, help="default: paths.tflite_fp32")
    parser.add_argument("--threads", type=int, default=4, help="LiteRT CPU threads (Pi 5 has 4 cores)")
    parser.add_argument("--repeats", type=int, default=1, help="timed runs per model; the median is saved")
    parser.add_argument("--count", type=int, default=None, help="run at most this many models now")
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        sys.exit("interrupted: run the same command again to resume")
