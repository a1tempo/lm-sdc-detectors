"""What the single-pass and generative detectors share: prompt building and the score."""

import time

import numpy as np
import pandas as pd

from sdc.detectors.base import Detector
from sdc.detectors.lm.prompts import build_messages
from sdc.observations.serialize import serialize_observation, serialize_reference

CONDITIONS = ("zero_shot", "reference")


def yes_no_score(p_yes: float | None, p_no: float | None) -> float:
    """Proposal Eq. 2: s = P(yes) / (P(yes) + P(no)). 0.5 when there is no answer."""
    if p_yes is None or p_no is None or p_yes + p_no == 0:
        return 0.5
    return p_yes / (p_yes + p_no)


class LMDetector(Detector):
    """A detector that asks the LM one yes/no question per run.

    Subclasses set `thinking` and implement `_query(prompt) -> dict` with at least
    "p_yes", "p_no", "parsed".
    """
    thinking: bool = False

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox"):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        super().__init__(access_level)
        self.client = client
        self.condition = condition
        self.reference_text: str | None = None

    def fit(self, fit_runs: pd.DataFrame, reference_ranges: dict | None = None) -> "LMDetector":
        """Nothing is trained: only the reference text is fixed (see set_reference)."""
        self.set_reference(reference_ranges)
        return self

    def set_reference(self, reference_ranges: dict | None) -> None:
        """For the reference condition, fix the text of normal ranges shown in every prompt.

        reference_ranges = {"points": <"points" of runs/reference_ranges.json>,
                            "outputs": <serialize.output_ranges(clean fit runs)>}
        """
        if self.condition == "reference":
            if reference_ranges is None:
                raise ValueError("the reference condition needs reference_ranges")
            self.reference_text = serialize_reference(
                reference_ranges["points"], reference_ranges["outputs"], self.access_level)

    def prompt(self, row: dict) -> str:
        """The full chat-formatted prompt for one run."""
        if self.condition == "reference" and self.reference_text is None:
            raise RuntimeError("call fit() with reference_ranges before scoring")
        system, user = build_messages(
            serialize_observation(row, self.access_level), self.access_level, self.reference_text)
        return self.client.chat_prompt(system, user, thinking=self.thinking)

    def score_details(self, runs: pd.DataFrame) -> pd.DataFrame:
        """One row per run: run_id, score, p_yes, p_no, parsed, elapsed_s, and extras."""
        out = []
        for row in runs.to_dict("records"):
            start = time.perf_counter()
            result = self._query(self.prompt(row))
            result["elapsed_s"] = time.perf_counter() - start
            result["score"] = yes_no_score(result["p_yes"], result["p_no"])
            out.append({"run_id": row["run_id"], "detector": self.name,
                        "access_level": self.access_level,
                        "prompt_condition": self.condition, **result})
        return pd.DataFrame(out)

    def score(self, runs: pd.DataFrame) -> np.ndarray:
        return self.score_details(runs)["score"].to_numpy(dtype=float)

    def _query(self, prompt: str) -> dict:
        raise NotImplementedError
