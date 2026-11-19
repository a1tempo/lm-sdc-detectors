import numpy as np

from sdc.detectors.base import Detector
from sdc.model.monitor import MONITORED_POINTS


def range_violation(runs, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Proposal Eq. 1: s_i = max_l max(0, m_il - b_l, a_l - n_il) / (b_l - a_l).

    A run with any non-finite value at a monitored point scores +inf.
    """
    score = np.zeros(len(runs))
    for point, (low, high) in bounds.items():
        n = runs[f"{point}_min"].to_numpy(dtype=float)
        m = runs[f"{point}_max"].to_numpy(dtype=float)
        width = high - low
        if not width > 0:
            raise ValueError(f"empty range at {point}: [{low}, {high}]")
        excess = np.maximum(0.0, np.maximum(m - high, low - n)) / width
        nonfinite = (runs[f"{point}_nonfinite"].fillna(0).to_numpy() > 0) | np.isnan(n) | np.isnan(m)
        score = np.maximum(score, np.where(nonfinite, np.inf, excess))
    return score


class RangeCheck(Detector):
    """Range restriction used as a detector (Chen et al., 2021), proposal Eq. 1.

    s_i = max_l max(0, m_il - b_l, a_l - n_il) / (b_l - a_l)
    where [a_l, b_l] is layer l's fault-free range on the fitting set and
    n_il, m_il are run i's min/max at layer l. Needs white-box access.
    """
    name = "range"
    low_key, high_key = "min", "max"

    def __init__(self, access_level: str = "whitebox"):
        if access_level != "whitebox":
            raise ValueError(f"{self.name} needs white-box observations")
        super().__init__(access_level)
        self.bounds: dict[str, tuple[float, float]] | None = None

    def fit(self, fit_runs, reference_ranges=None):
        """reference_ranges: the "points" of runs/reference_ranges.json (from clean fit runs)."""
        if reference_ranges is None:
            raise ValueError(f"{self.name} needs reference_ranges")
        missing = set(MONITORED_POINTS) - set(reference_ranges)
        if missing:
            raise ValueError(f"reference_ranges is missing {sorted(missing)}")
        self.bounds = {point: (float(reference_ranges[point][self.low_key]),
                               float(reference_ranges[point][self.high_key]))
                       for point in MONITORED_POINTS}
        return self

    def score(self, runs):
        if self.bounds is None:
            raise RuntimeError("call fit() first")
        return range_violation(runs, self.bounds)
