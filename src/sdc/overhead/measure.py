"""Per-detector overhead on the Raspberry Pi 5 (proposal §7.2).

Measured in a separate session from the detection runs, with temperature
logged to catch thermal throttling. Needs only the standard library, so it runs on a Pi
without PyTorch.
"""

import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


def time_per_call(fn: Callable[[], object], warmup: int = 10, iterations: int = 100) -> dict:
    """Mean / p50 / p95 / min latency in seconds (time.perf_counter), after `warmup` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return {
        "mean_s": statistics.fmean(times), "p50_s": times[len(times) // 2],
        "p95_s": times[min(len(times) - 1, int(0.95 * len(times)))], "min_s": times[0],
        "iterations": iterations,
    }


def peak_rss_bytes() -> int:
    """Peak resident memory of this process so far (ru_maxrss: KiB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def current_rss_bytes() -> int | None:
    """Current resident memory from /proc (Linux only; None elsewhere)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_memory(fn: Callable[[], object]) -> int:
    """Peak resident memory in bytes after running fn (the process peak, not fn's alone)."""
    fn()
    return peak_rss_bytes()


def read_cpu_temp_c() -> float | None:
    """CPU temperature from /sys/class/thermal/thermal_zone0/temp (None if unavailable)."""
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def read_throttled() -> str | None:
    """`vcgencmd get_throttled` on a Pi, e.g. "throttled=0xe0006" (None elsewhere)."""
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
