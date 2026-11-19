"""Measure each classical detector's latency and memory on the Pi (separate session).

Action item: Part I.3c / Part II.2c

Reads the kit from scripts/09_prepare_overhead.py and measures, on the Pi CPU:
    preprocess             load + resize + normalize one image
    classifier             LiteRT FP32 classifier alone (the no-detector baseline)
    classifier_monitored   the same model also computing the white-box observation in-graph
    per-run detector cost, one run at a time as in deployment, from the model's outputs:
        entropy            softmax top-1 probability and entropy from the logits
        range              range check (proposal Eq. 1) on the 6 points' min/max
        learned_*          logistic regression, black-box / white-box
        boosted_*          gradient-boosted trees, black-box / white-box
    end_to_end_*           monitored classifier + features + detector, per image
plus memory (RSS), CPU temperature and the throttle flags before and after each phase.

The trained detectors are fitted here on the kit's fit features (same settings as
src/sdc/detectors/classical); this mirrors that code in numpy + scikit-learn because the
Pi has no PyTorch. Writes <out>/overhead.json and prints a summary. Needs numpy, Pillow,
PyYAML, ai-edge-litert and scikit-learn (the version in kit.json). On the Pi:

    nohup scripts/notify.sh .venv/bin/python scripts/09_overhead.py > ~/overhead.log 2>&1 &
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sdc.config import REPO_ROOT, load_config
from sdc.model.tflite import load_interpreter, preprocess_image, run_tflite_outputs
from sdc.overhead.measure import (
    current_rss_bytes,
    peak_rss_bytes,
    read_cpu_temp_c,
    read_throttled,
    time_per_call,
)

POINTS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


def device_name() -> str:
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def signed_log(x: np.ndarray) -> np.ndarray:
    """Same transform as sdc.observations.stats._signed_log."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0,
                      posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)
    return np.sign(x) * np.log1p(np.abs(x))


def blackbox_from_logits(logits: np.ndarray) -> np.ndarray:
    """[top1_prob, entropy] from one image's logits (as observations.stats.blackbox_stats)."""
    z = logits.astype(np.float64).ravel()
    z = z - z.max()
    log_p = z - np.log(np.exp(z).sum())
    p = np.exp(log_p)
    return np.array([p.max(), -(p * log_p).sum()])


def range_score(min_max: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Proposal Eq. 1 from the 6 points' (min, max), as detectors.classical.range_check."""
    mins, maxes = min_max[0::2], min_max[1::2]
    return float(np.max(np.maximum(0.0, np.maximum(maxes - high, low - mins)) / (high - low)))


def make_learned():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def make_boosted():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0)


class Log:
    def __init__(self):
        self.phases = []

    def phase(self, name: str, **result):
        entry = {"phase": name, "temp_c": read_cpu_temp_c(), "rss_mb": _mb(current_rss_bytes()),
                 "time": datetime.now(timezone.utc).isoformat(), **result}
        self.phases.append(entry)
        timing = result.get("timing")
        if timing:
            detail = f"p50 {timing['p50_s'] * 1000:8.3f} ms  p95 {timing['p95_s'] * 1000:8.3f} ms"
        elif "seconds" in result:
            detail = f"{result['seconds']:.2f} s"
        elif "auroc" in result:
            detail = f"{result['auroc']:.3f}"
        else:
            detail = ""
        temp = f"{entry['temp_c']:.1f} C" if entry["temp_c"] is not None else ""
        print(f"{name:32s} {detail}  {temp}", flush=True)


def _mb(value):
    return None if value is None else round(value / 1024**2, 1)


def cycle(items):
    state = {"i": 0}

    def next_item():
        item = items[state["i"] % len(items)]
        state["i"] += 1
        return item
    return next_item


def main(args) -> None:
    config = load_config(args.config)
    kit = Path(args.kit)
    meta = json.loads((kit / "kit.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log()

    import sklearn
    if sklearn.__version__ != meta["scikit_learn"]:
        print(f"note: scikit-learn {sklearn.__version__} here, {meta['scikit_learn']} on the Mac")
    info = {
        "device": device_name(), "machine": platform.machine(), "python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "threads": args.threads, "repeats": args.repeats,
        "throttled_before": read_throttled(), "temp_before_c": read_cpu_temp_c(),
        "started": datetime.now(timezone.utc).isoformat(), "kit": meta,
    }
    print(f"{info['device']}, {args.threads} threads, start temp {info['temp_before_c']} C, "
          f"{info['throttled_before']}")

    data_dir = Path(config["paths"]["eurosat_dir"])
    images = [data_dir / p for p in json.loads((kit / "images.json").read_text())]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[0]}")

    # Models.
    start = time.perf_counter()
    plain = load_interpreter(kit / "classifier.tflite", num_threads=args.threads)
    log.phase("load classifier", seconds=time.perf_counter() - start)
    start = time.perf_counter()
    monitored = load_interpreter(kit / "classifier_monitored.tflite", num_threads=args.threads)
    log.phase("load monitored classifier", seconds=time.perf_counter() - start)

    next_image = cycle(images)
    log.phase("preprocess", timing=time_per_call(lambda: preprocess_image(next_image()),
                                                  warmup=5, iterations=len(images)))
    inputs = [preprocess_image(p) for p in images]
    n_calls = len(inputs) * args.repeats

    def run(interpreter):
        nxt = cycle(inputs)
        return lambda: run_tflite_outputs(interpreter, nxt())

    plain_run, monitored_run = run(plain), run(monitored)
    log.phase("classifier", timing=time_per_call(lambda: plain_run(), warmup=args.warmup, iterations=n_calls))
    log.phase("classifier_monitored",
              timing=time_per_call(lambda: monitored_run(), warmup=args.warmup, iterations=n_calls))

    # One real model output per image, to feed the per-run detector timings.
    outputs = [run_tflite_outputs(monitored, x) for x in inputs]
    logits = [next(o for o in out_ if o.shape == (1, 10)) for out_ in outputs]
    stats = [next(o for o in out_ if o.shape == (6, 5)) for out_ in outputs]

    # Train the detectors here (no pickles), on the kit's fit features.
    fit = np.load(kit / "fit_features.npz")
    ev = np.load(kit / "eval_features.npz")
    detectors = {}
    for level in ("blackbox", "whitebox"):
        for name, make in (("learned", make_learned), ("boosted", make_boosted)):
            start = time.perf_counter()
            model = make().fit(fit[level], fit["is_sdc"])
            detectors[f"{name}_{level}"] = model
            log.phase(f"train {name}_{level}", seconds=time.perf_counter() - start)
    for name, model in detectors.items():
        level = name.split("_")[1]
        p = model.predict_proba(ev[level])[:, 1]
        y = ev["is_sdc"]
        ranks = np.argsort(np.argsort(p)) + 1
        auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
        log.phase(f"check {name} (eval sample AUROC)", auroc=float(auc))

    ranges = json.loads((kit / "reference_ranges.json").read_text())["points"]
    low = np.array([ranges[p]["min"] for p in POINTS])
    high = np.array([ranges[p]["max"] for p in POINTS])
    next_logits, next_stats = cycle(logits), cycle(stats)
    calls = args.detector_calls

    log.phase("entropy (black-box features)",
              timing=time_per_call(lambda: blackbox_from_logits(next_logits()), warmup=50, iterations=calls))
    log.phase("range check",
              timing=time_per_call(lambda: range_score(next_stats()[:, :2].ravel(), low, high),
                                   warmup=50, iterations=calls))

    def features(level, logit, stat):
        bb = blackbox_from_logits(logit)
        if level == "blackbox":
            return signed_log(bb)[None, :]
        return signed_log(np.concatenate([bb, stat.ravel()]))[None, :]

    for name, model in detectors.items():
        level = name.split("_")[1]
        log.phase(name, timing=time_per_call(
            lambda m=model, lv=level: m.predict_proba(features(lv, next_logits(), next_stats())),
            warmup=50, iterations=calls))

    for name in ("boosted_whitebox", "learned_whitebox"):
        model = detectors[name]
        nxt = cycle(inputs)

        def end_to_end(m=model):
            out_ = run_tflite_outputs(monitored, nxt())
            lg = next(o for o in out_ if o.shape == (1, 10))
            st = next(o for o in out_ if o.shape == (6, 5))
            return m.predict_proba(features("whitebox", lg, st))
        log.phase(f"end_to_end {name}", timing=time_per_call(end_to_end, warmup=args.warmup,
                                                             iterations=len(inputs)))

    info.update({"throttled_after": read_throttled(), "temp_after_c": read_cpu_temp_c(),
                 "peak_rss_mb": _mb(peak_rss_bytes()), "finished": datetime.now(timezone.utc).isoformat(),
                 "phases": log.phases})
    (out / "overhead.json").write_text(json.dumps(info, indent=2))

    timed = {e["phase"]: e["timing"]["p50_s"] * 1000 for e in log.phases if "timing" in e}
    base = timed["classifier"]
    print(f"\nclassifier alone: {base:.1f} ms per image (median)")
    print(f"white-box observation in-graph: +{timed['classifier_monitored'] - base:.1f} ms")
    for name in ("entropy (black-box features)", "range check", "learned_blackbox", "learned_whitebox",
                 "boosted_blackbox", "boosted_whitebox"):
        print(f"  {name:30s} +{timed[name]:.3f} ms per run")
    temps = [e["temp_c"] for e in log.phases if e["temp_c"] is not None]
    if temps:
        print(f"temperature {min(temps):.1f}-{max(temps):.1f} C; {info['throttled_before']} -> "
              f"{info['throttled_after']}; peak memory {info['peak_rss_mb']} MB")
    print(f"wrote {out / 'overhead.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--kit", default=str(REPO_ROOT / "runs" / "overhead_kit"))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "overhead" / "pi"))
    parser.add_argument("--threads", type=int, default=4, help="LiteRT CPU threads (Pi 5 has 4 cores)")
    parser.add_argument("--repeats", type=int, default=10, help="passes over the images per classifier timing")
    parser.add_argument("--warmup", type=int, default=20, help="untimed inferences before each timing")
    parser.add_argument("--detector-calls", type=int, default=5000, help="timed single-run calls per detector")
    main(parser.parse_args())
