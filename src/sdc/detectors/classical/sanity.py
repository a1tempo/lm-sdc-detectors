import numpy as np

from sdc.detectors.base import Detector
from sdc.model.monitor import MONITORED_POINTS


class SanityCheck(Detector):
    """s_i = 1 if the run crashed or produced NaN/inf, else 0.

    Uses only what is observable at run time, never the label: a crash (exception or
    timeout) leaves no output, so top1_prob is missing, and a non-finite output makes
    top1_prob NaN. White-box also counts non-finite values at every monitored point.
    """
    name = "sanity"

    def score(self, runs):
        bad = ~np.isfinite(runs["top1_prob"].to_numpy(dtype=float))
        if self.access_level == "whitebox":
            for point in MONITORED_POINTS:
                bad |= runs[f"{point}_nonfinite"].fillna(0).to_numpy() > 0
        return bad.astype(float)
