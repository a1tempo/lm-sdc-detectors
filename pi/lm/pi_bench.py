"""Time the three LM detectors on this machine and check their scores. No PyTorch needed.

Reads prompts.jsonl (from make_prompts.py) and runs, for the first --runs runs:
  single   one decode, thinking off: P(yes), P(no)              (single-pass detector)
  probe    one decode with the hidden state switched on         (probe detector's pass)
  gen      thinking on: reason, then P(yes), P(no)              (generative detector;
           only --gen-runs runs, one access level / condition, because it is slow)
for every access level and prompt condition. It uses the team's QwenClient unchanged,
so these are the same calls scripts/07_score_lm.py makes.

Recorded per query: wall-clock seconds, prompt length in tokens, scores, CPU temperature.
Recorded once: model load time, resident memory after load and at peak, throttling flags,
device, versions, model sha256, git commit. Writes <out>/results.csv, <out>/hidden.npz,
<out>/summary.json.

Long runs: results.csv and hidden.npz are rewritten every --save-every queries, and a rerun
with the same --out skips the queries already saved (so a crash or power cut loses a few
minutes, not the run). --passes probe skips the separate single-pass decode: the probe pass
returns the same P(yes), P(no) (equal to 6 decimal places) plus the hidden state.

With --compare <dir of another run of this script>, it prints how far this machine's scores
are from that one's on the same prompts (e.g. Pi vs laptop).

These are preliminary timings for choosing 0.6B vs 1.7B, not the paper's overhead
measurement (that one is scripts/09_overhead.py, in its own session, with power).

    python pi_bench.py --repo ~/lm-sdc-detectors --gguf models/Qwen3-0.6B-Q4_K_M.gguf \
        --out results/pi-0.6b --compare results/laptop-0.6b
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
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent


def read_proc_status() -> dict:
    """Current (VmRSS) and peak (VmHWM) resident memory of this process, in MB. The
    model file is memory-mapped, so its pages count here once they are read."""
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[key] = int(value.split()[0]) / 1024
    return out


def cpu_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def device_name() -> str:
    """"Raspberry Pi 5 Model B Rev 1.0" on a Pi; the CPU model name elsewhere."""
    tree = Path("/proc/device-tree/model")
    if tree.exists():
        return tree.read_text().strip("\x00\n ")
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.machine()


def os_name() -> str:
    for line in Path("/etc/os-release").read_text().splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return platform.platform()


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(p_yes, p_no):
    """Proposal Eq. 2, same rule as sdc.detectors.lm.common.yes_no_score."""
    if p_yes is None or p_no is None or p_yes + p_no == 0:
        return 0.5
    return p_yes / (p_yes + p_no)


NUMERIC = ("seconds", "p_yes", "p_no", "score", "temp_c", "prompt_tokens", "n_gen_tokens")


def load_saved(out: Path) -> tuple[list[dict], dict]:
    """Rows and hidden states saved by an earlier, interrupted run with the same --out."""
    if not (out / "results.csv").exists():
        return [], {}
    with open(out / "results.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:   # the csv gives text; turn the number columns back into numbers
        for k in NUMERIC:
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
            elif k in r:
                r[k] = None
    hidden = dict(np.load(out / "hidden.npz")) if (out / "hidden.npz").exists() else {}
    return rows, hidden


def save(out: Path, rows: list[dict], hidden: dict) -> None:
    """Write results.csv and hidden.npz. Each goes to a temporary file first and is then
    renamed, so a crash in the middle of saving cannot leave a half-written file."""
    fields = list(dict.fromkeys(k for r in rows for k in r))   # every column, first-seen order
    with open(out / "results.csv.tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.csv.tmp").replace(out / "results.csv")
    with open(out / "hidden.npz.tmp", "wb") as f:
        np.savez(f, **hidden)
    (out / "hidden.npz.tmp").replace(out / "hidden.npz")


def compare(rows: list[dict], hidden: dict, other_dir: Path) -> None:
    """Print score differences against another run of this script on the same prompts."""
    with open(other_dir / "results.csv") as f:
        other = {(r["run_id"], r["access_level"], r["condition"], r["pass"]): r
                 for r in csv.DictReader(f)}
    other_hidden = np.load(other_dir / "hidden.npz")
    print(f"\nCompared with {other_dir}:")
    for name in ("single", "probe", "gen"):
        pairs = [(r, other[k]) for r in rows
                 if r["pass"] == name and (k := (r["run_id"], r["access_level"],
                                                 r["condition"], r["pass"])) in other]
        if not pairs:
            continue
        diffs = [abs(r["score"] - float(o["score"])) for r, o in pairs]
        line = f"  {name:6s} n={len(pairs)}  largest score difference {max(diffs):.2g}"
        if name == "gen":
            same = sum(int(r["n_gen_tokens"]) == int(o["n_gen_tokens"]) for r, o in pairs)
            line += f"; same reasoning length on {same}/{len(pairs)}"
        print(line)
    cos = []
    for key, h in hidden.items():
        if key in other_hidden.files:
            g = other_hidden[key]
            cos.append(float(h @ g / (np.linalg.norm(h) * np.linalg.norm(g))))
    if cos:
        print(f"  hidden states: n={len(cos)}, lowest cosine similarity {min(cos):.6f}")


def main(args) -> None:
    sys.path.insert(0, str(Path(args.repo).resolve() / "src"))
    import llama_cpp
    from sdc.detectors.lm.llama_client import QwenClient

    prompts = [json.loads(line) for line in open(args.prompts)]
    prompts = [p for p in prompts if p["run_index"] < args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gguf = Path(args.gguf)

    env = {
        "command": " ".join(sys.argv),
        "device": device_name(),
        "os": os_name(), "machine": platform.machine(), "kernel": platform.release(),
        "python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__,
        "gguf": gguf.name, "gguf_sha256": sha256(gguf),
        "git_commit": run_cmd(["git", "-C", args.repo, "rev-parse", "HEAD"]),
        "threads": args.threads, "n_ctx": 2048, "flash_attn": False,
        # CPU features llama.cpp was compiled with (e.g. NEON, DOTPROD on the Pi).
        "llama_system_info": llama_cpp.llama_print_system_info().decode().strip(),
        "throttled_before": run_cmd(["vcgencmd", "get_throttled"]),
        "temp_before_c": cpu_temp_c(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    start = time.perf_counter()
    client = QwenClient(str(gguf), n_ctx=2048, n_threads=args.threads, flash_attn=False)
    env["load_s"] = time.perf_counter() - start
    env["rss_after_load_mb"] = read_proc_status()["VmRSS"]
    print(f"{env['device']} | {gguf.name} | {args.threads} threads | "
          f"loaded in {env['load_s']:.1f} s, {env['rss_after_load_mb']:.0f} MB resident")

    rows, hidden = load_saved(out)
    # A probe row only counts as done if its hidden state was saved too.
    done = {(r["run_id"], r["access_level"], r["condition"], r["pass"]) for r in rows
            if r["pass"] != "probe"
            or f"{r['run_id']}|{r['access_level']}|{r['condition']}" in hidden}
    rows = [r for r in rows if (r["run_id"], r["access_level"], r["condition"], r["pass"]) in done]
    env["resumed_rows"] = len(rows)
    if rows:
        print(f"resuming: {len(rows)} queries already saved in {out}")
    passes = args.passes.split(",")

    def record(p, name, seconds, p_yes, p_no, **extra):
        row = {"run_id": p["run_id"], "label": p["label"], "access_level": p["access_level"],
               "condition": p["condition"], "pass": name, "seconds": seconds,
               "p_yes": p_yes, "p_no": p_no, "score": score(p_yes, p_no),
               "temp_c": cpu_temp_c(), **extra}
        rows.append(row)
        if len(rows) % args.save_every == 0:
            save(out, rows, hidden)
        print(f"  {name:6s} {p['run_id']} {p['label']:6s} {p['access_level']:8s} "
              f"{p['condition']:9s} score={row['score']:.3f}  {seconds:6.2f} s", flush=True)

    # Single-pass and probe passes on every selected prompt.
    for name in passes:
        print(f"\n{name} pass, {len(prompts)} prompts")
        for p in prompts:
            if (p["run_id"], p["access_level"], p["condition"], name) in done:
                continue
            n_tok = len(client._tokenize(p["prompt_single"]))
            t = time.perf_counter()
            if name == "single":
                p_yes, p_no = client.yes_no_probs(p["prompt_single"])
            else:
                p_yes, p_no, h = client.yes_no_probs_and_hidden(p["prompt_single"])
                hidden[f"{p['run_id']}|{p['access_level']}|{p['condition']}"] = h
            record(p, name, time.perf_counter() - t, p_yes, p_no, prompt_tokens=n_tok)

    # Generative pass on a few runs, one access level / condition.
    gen = [p for p in prompts if p["run_index"] < args.gen_runs
           and p["access_level"] == args.gen_level and p["condition"] == args.gen_condition]
    print(f"\ngenerative pass, {len(gen)} prompts ({args.gen_level}, {args.gen_condition}), "
          f"budget {args.max_tokens} tokens")
    for p in gen:
        if (p["run_id"], p["access_level"], p["condition"], "gen") in done:
            continue
        n_tok = len(client._tokenize(p["prompt_think"]))
        t = time.perf_counter()
        text, p_yes, p_no = client.generate_then_answer(p["prompt_think"], args.max_tokens)
        record(p, "gen", time.perf_counter() - t, p_yes, p_no, prompt_tokens=n_tok,
               n_gen_tokens=client.last_n_generated, parsed=p_yes is not None)
        (out / f"reasoning_{p['run_id']}_{p['access_level']}_{p['condition']}.txt").write_text(text)

    mem = read_proc_status()
    env.update({"rss_end_mb": mem["VmRSS"], "rss_peak_mb": mem["VmHWM"],
                "temp_after_c": cpu_temp_c(),
                "throttled_after": run_cmd(["vcgencmd", "get_throttled"]),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    for name in ("single", "probe", "gen"):
        secs = [r["seconds"] for r in rows if r["pass"] == name]
        if secs:
            env[f"{name}_median_s"] = statistics.median(secs)
            env[f"{name}_max_s"] = max(secs)
    for level in ("blackbox", "whitebox"):
        for cond in ("zero_shot", "reference"):
            for name in ("single", "probe"):
                secs = [r["seconds"] for r in rows if r["pass"] == name
                        and r["access_level"] == level and r["condition"] == cond]
                if secs:
                    env[f"{name}_median_s__{level}__{cond}"] = statistics.median(secs)
    gens = [r for r in rows if r["pass"] == "gen"]
    if gens:   # includes reading the prompt, so a slight underestimate of generation speed
        env["gen_tokens_per_s"] = sum(r["n_gen_tokens"] for r in gens) / sum(r["seconds"] for r in gens)

    save(out, rows, hidden)
    (out / "summary.json").write_text(json.dumps(env, indent=2))

    print(f"\npeak resident memory {env['rss_peak_mb']:.0f} MB"
          + "".join(f" | {name} median {env[f'{name}_median_s']:.2f} s" for name in passes)
          + (f" | gen median {env['gen_median_s']:.1f} s, ~{env['gen_tokens_per_s']:.1f} tok/s"
             if gens else ""))
    print(f"temperature {env['temp_before_c']} -> {env['temp_after_c']} C | "
          f"throttled {env['throttled_before']} -> {env['throttled_after']}")
    print(f"wrote {out}/results.csv, summary.json, hidden.npz")
    if args.compare:
        compare(rows, hidden, Path(args.compare))
    assert sys.modules.get("torch") is None, "torch was imported; this script should not need it"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="lm-sdc-detectors checkout on micah/lm-detectors")
    p.add_argument("--gguf", required=True)
    p.add_argument("--prompts", default=str(KIT / "prompts.jsonl"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--runs", type=int, default=4, help="first k runs (half SDC, half masked)")
    p.add_argument("--gen-runs", type=int, default=2)
    p.add_argument("--gen-level", default="whitebox", choices=["blackbox", "whitebox"])
    p.add_argument("--gen-condition", default="zero_shot", choices=["zero_shot", "reference"])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--passes", default="single,probe", help="single, probe, or both (comma-separated)")
    p.add_argument("--save-every", type=int, default=20, help="save after this many queries")
    p.add_argument("--compare", default=None, help="another output dir of this script")
    p.add_argument("--out", required=True)
    main(p.parse_args())
