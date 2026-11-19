from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sdc.detectors.base import Detector
from sdc.observations.stats import feature_matrix


class LearnedDetector(Detector):
    """Logistic regression on the raw observation features (supervised baseline).

    Uses the same classifier as the LM probe detector (StandardScaler, then L2 logistic
    regression with C=1), so probe vs. learned isolates what the LM representation adds
    (ablation, RQ4). Features are observations.stats.feature_matrix (signed log).
    """
    name = "learned"
    supervised = True

    def __init__(self, access_level: str = "whitebox", C: float = 1.0):
        super().__init__(access_level)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        self.fitted = False

    def fit(self, fit_runs, reference_ranges=None):
        """Train on fitting runs labeled sdc (1) or masked (0). Crashes aren't allowed."""
        labels = set(fit_runs["label"])
        if not labels <= {"sdc", "masked"} or len(labels) < 2:
            raise ValueError(f"the learned detector trains on sdc and masked runs only, got {sorted(labels)}")
        self.model.fit(feature_matrix(fit_runs, self.access_level),
                       (fit_runs["label"] == "sdc").to_numpy())
        self.fitted = True
        return self

    def score(self, runs):
        if not self.fitted:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(feature_matrix(runs, self.access_level))[:, 1]
