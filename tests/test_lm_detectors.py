"""Single-pass and generative detectors, with a fake client (no model needed),
plus one wiring check against the real GGUF (skipped if it's missing)."""

import os
from pathlib import Path

import pandas as pd
import pytest

from sdc.detectors.lm.common import yes_no_score
from sdc.detectors.lm.generative import GenerativeDetector
from sdc.detectors.lm.single_pass import SinglePassDetector
from sdc.model.monitor import MONITORED_POINTS

RANGES = {
    "points": {p: {"min": 0.0, "max": 8.2} for p in MONITORED_POINTS},
    "outputs": {"top1_prob": {"min": 0.51, "max": 1.0}, "entropy": {"min": 0.0, "max": 1.5}},
}


class FakeClient:
    """Records the prompts it gets and returns fixed answers."""

    def __init__(self, answer=(0.3, 0.6)):
        self.answer = answer
        self.prompts = []
        self.last_n_generated = 0

    def chat_prompt(self, system, user, thinking):
        return f"thinking={thinking}\n{system}\n{user}"

    def yes_no_probs(self, prompt):
        self.prompts.append(prompt)
        return self.answer

    def generate_then_answer(self, prompt, max_tokens):
        self.prompts.append(prompt)
        self.last_n_generated = 42
        return "<think>reasoning</think>\n\n", *self.answer


def make_runs(n=2):
    rows = []
    for i in range(n):
        row = {"run_id": f"fit-{i:08d}", "label": "sdc", "flipped_value": 5.678798848251858e37,
               "top1_prob": 0.9999, "entropy": 0.000946}
        for p in MONITORED_POINTS:
            row.update({f"{p}_min": 0.0, f"{p}_max": 3.2, f"{p}_mean": 0.27,
                        f"{p}_std": 0.27, f"{p}_nonfinite": 0})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("p_yes, p_no, score", [
    (0.3, 0.6, 1 / 3), (0.9, 0.0, 1.0), (None, None, 0.5), (0.0, 0.0, 0.5),
])
def test_yes_no_score(p_yes, p_no, score):
    assert yes_no_score(p_yes, p_no) == pytest.approx(score)


def test_single_pass_scores_and_details():
    client = FakeClient()
    det = SinglePassDetector(client).fit(make_runs())
    details = det.score_details(make_runs())
    assert list(details.run_id) == ["fit-00000000", "fit-00000001"]
    assert details.score.tolist() == pytest.approx([1 / 3, 1 / 3])
    assert details.parsed.all() and details.yes_no_mass.tolist() == pytest.approx([0.9, 0.9])
    assert set(details.detector) == {"lm_single_pass"}
    assert client.prompts[0].startswith("thinking=False")


def test_generative_uses_thinking_and_counts_tokens():
    client = FakeClient()
    details = GenerativeDetector(client).fit(make_runs()).score_details(make_runs(1))
    assert client.prompts[0].startswith("thinking=True")
    assert details.n_gen_tokens.tolist() == [42]
    assert details.reasoning.tolist() == ["<think>reasoning</think>\n\n"]


def test_generative_unparseable_gets_half():
    details = GenerativeDetector(FakeClient(answer=(None, None))).score_details(make_runs(1))
    assert details.score.tolist() == [0.5]
    assert not details.parsed.any()


@pytest.mark.parametrize("cls", [SinglePassDetector, GenerativeDetector])
def test_prompts_never_contain_answer_key(cls):
    client = FakeClient()
    cls(client, condition="reference").fit(make_runs(), RANGES).score(make_runs(1))
    assert "sdc" not in client.prompts[0] and "5.679e+37" not in client.prompts[0]


def test_reference_condition_needs_ranges():
    det = SinglePassDetector(FakeClient(), condition="reference")
    with pytest.raises(RuntimeError):
        det.score(make_runs(1))          # fit() not called
    with pytest.raises(ValueError):
        det.fit(make_runs())             # fit() without ranges


def test_reference_text_reaches_prompt_and_zero_shot_has_none():
    for level, n_lines in [("whitebox", len(MONITORED_POINTS) + 2), ("blackbox", 2)]:
        client = FakeClient()
        det = SinglePassDetector(client, "reference", level).fit(make_runs(), RANGES)
        assert len(det.reference_text.splitlines()) == n_lines
        det.score(make_runs(1))
        assert "Normal ranges" in client.prompts[0]
    client = FakeClient()
    SinglePassDetector(client).fit(make_runs()).score(make_runs(1))
    assert "Normal ranges" not in client.prompts[0]


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        SinglePassDetector(FakeClient(), condition="few_shot")


GGUF = os.environ.get("SDC_GGUF",
                      str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B-Q4_K_M.gguf"))


@pytest.mark.skipif(not Path(GGUF).exists(), reason=f"model not found: {GGUF}")
def test_single_pass_with_real_model():
    pytest.importorskip("llama_cpp")
    from sdc.detectors.lm.llama_client import QwenClient
    det = SinglePassDetector(QwenClient(GGUF, n_ctx=2048, n_threads=8), "reference")
    details = det.fit(make_runs(), RANGES).score_details(make_runs(1))
    assert 0.0 <= details.score[0] <= 1.0
    assert details.yes_no_mass[0] > 0.99


# ---------- probe ----------

import numpy as np  # noqa: E402

from sdc.detectors.lm.probe import ProbeDetector  # noqa: E402


class FakeHiddenClient(FakeClient):
    """Hidden state = a fixed vector, shifted along dimension 0 for SDC-looking prompts."""

    def yes_no_probs_and_hidden(self, prompt):
        self.prompts.append(prompt)
        h = np.zeros(8)
        h[0] = 5.0 if "e+37" in prompt else -5.0
        return 0.3, 0.6, h


def make_labeled_runs():
    runs = make_runs(6)
    runs["label"] = ["sdc", "masked"] * 3
    runs.loc[runs.label == "sdc", "layer2_max"] = 1.1e37   # visible in the white-box prompt
    return runs


def test_probe_learns_a_separable_direction():
    client = FakeHiddenClient()
    det = ProbeDetector(client).fit(make_labeled_runs())
    scores = det.score(make_labeled_runs())
    assert (scores[::2] > 0.5).all() and (scores[1::2] < 0.5).all()
    assert len(client.prompts) == 6                  # scoring reused the cached states
    assert client.prompts[0].startswith("thinking=False")


def test_probe_extract_returns_same_pass_details():
    det = ProbeDetector(FakeHiddenClient())
    hidden, details = det.extract(make_labeled_runs())
    assert hidden.shape == (6, 8)
    assert details.single_pass_score.tolist() == pytest.approx([1 / 3] * 6)


def test_probe_uses_loaded_hidden_states_without_client():
    runs = make_labeled_runs()
    det = ProbeDetector(client=None)
    det.load_hidden(runs.run_id, np.eye(6)[:, :4] + (runs.label == "sdc").to_numpy()[:, None])
    det.fit(runs)                                    # would fail if it called the client
    assert det.score(runs).shape == (6,)


def test_probe_rejects_crashes_and_one_class():
    runs = make_labeled_runs()
    with pytest.raises(ValueError):
        ProbeDetector(FakeHiddenClient()).fit(runs.assign(label="sdc"))
    runs.loc[0, "label"] = "crash"
    with pytest.raises(ValueError):
        ProbeDetector(FakeHiddenClient()).fit(runs)
