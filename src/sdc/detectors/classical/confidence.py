import numpy as np

from sdc.detectors.base import Detector


class ConfidenceCheck(Detector):
    """s_i = entropy of the output distribution (black-box).

    High entropy means an unsure prediction. A non-finite output (entropy NaN) gets +inf.
    """
    name = "confidence"

    def __init__(self, access_level: str = "blackbox"):
        super().__init__(access_level)

    def score(self, runs):
        entropy = runs["entropy"].to_numpy(dtype=float)
        return np.where(np.isnan(entropy), np.inf, entropy)
