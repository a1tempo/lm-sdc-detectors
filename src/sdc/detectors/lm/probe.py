import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sdc.detectors.lm.common import LMDetector, yes_no_score


class ProbeDetector(LMDetector):
    """Logistic-regression probe on the LM's final hidden state at the answer
    position (Alain & Bengio, 2016): an open analog of a "System One" decision head.

    The prompt is the single-pass (non-thinking) prompt, and one decode returns both the
    hidden state and P(yes)/P(no): the forward pass the probe shares with single-pass.
    The hidden state is the 1024-dim post-final-RMSNorm vector that feeds the output layer.

    Hidden states are kept in `self.cache` (run_id -> vector). Fill it from disk with
    load_hidden() to avoid recomputing; runs missing from the cache are computed with
    the client. Same classifier as LearnedDetector; only the input representation differs.
    """
    name = "lm_probe"
    supervised = True
    thinking = False

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox",
                 C: float = 1.0):
        super().__init__(client, condition, access_level)
        self.cache: dict[str, np.ndarray] = {}
        # Standardize each of the 1024 dimensions, then L2-regularized logistic regression.
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

    def extract(self, runs: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Hidden states for `runs` (shape [n, n_embd]) and, for runs computed now,
        details from the same pass: run_id, p_yes, p_no, single_pass_score, elapsed_s."""
        details = []
        for row in runs.to_dict("records"):
            if row["run_id"] in self.cache:
                continue
            start = time.perf_counter()
            p_yes, p_no, hidden = self.client.yes_no_probs_and_hidden(self.prompt(row))
            details.append({"run_id": row["run_id"], "p_yes": p_yes, "p_no": p_no,
                            "single_pass_score": yes_no_score(p_yes, p_no),
                            "elapsed_s": time.perf_counter() - start})
            self.cache[row["run_id"]] = hidden
        hidden = np.stack([self.cache[run_id] for run_id in runs["run_id"]])
        return hidden, pd.DataFrame(details)

    def load_hidden(self, run_ids, hidden: np.ndarray) -> None:
        self.cache.update(zip(run_ids, hidden))

    def fit(self, fit_runs: pd.DataFrame, reference_ranges: dict | None = None) -> "ProbeDetector":
        """Train on fitting runs labeled sdc (1) or masked (0). Crashes aren't allowed."""
        labels = set(fit_runs["label"])
        if not labels <= {"sdc", "masked"} or len(labels) < 2:
            raise ValueError(f"the probe trains on sdc and masked runs only, got {sorted(labels)}")
        self.set_reference(reference_ranges)
        hidden, _ = self.extract(fit_runs)
        self.model.fit(hidden, (fit_runs["label"] == "sdc").to_numpy())
        return self

    def score(self, runs: pd.DataFrame) -> np.ndarray:
        """Predicted probability of SDC."""
        hidden, _ = self.extract(runs)
        return self.model.predict_proba(hidden)[:, 1]

    def score_details(self, runs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"run_id": runs["run_id"].to_numpy(), "detector": self.name,
                             "access_level": self.access_level,
                             "prompt_condition": self.condition, "score": self.score(runs)})
