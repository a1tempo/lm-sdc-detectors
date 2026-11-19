"""QwenClient checks against values measured on 2026-11-02 (results-log.md).

Needs llama-cpp-python and the GGUF. Set SDC_GGUF to the model path, otherwise
models/Qwen3-0.6B-Q4_K_M.gguf under the repo root is used. Skipped if either is missing.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

llama_cpp = pytest.importorskip("llama_cpp")
from sdc.detectors.lm.llama_client import QwenClient  # noqa: E402

GGUF = os.environ.get("SDC_GGUF",
                      str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B-Q4_K_M.gguf"))
pytestmark = pytest.mark.skipif(not Path(GGUF).exists(), reason=f"model not found: {GGUF}")

SYSTEM = (
    "You monitor a ResNet-18 image classifier for silent data corruption caused by "
    "hardware bit flips. You are given activation statistics from one inference run. "
    "Decide whether the run is corrupted."
)
USER = (  # hand-written placeholder observation, not real ResNet data
    "stem: min=0.00 max=8.31 mean=0.62 std=0.88 nonfinite=0\n"
    "layer1: min=0.00 max=9.47 mean=0.71 std=0.95 nonfinite=0\n"
    "layer2: min=0.00 max=10.12 mean=0.55 std=0.91 nonfinite=0\n"
    "layer3: min=0.00 max=11.84 mean=0.41 std=1.22 nonfinite=0\n"
    "layer4: min=0.00 max=24.60 mean=1.03 std=2.10 nonfinite=0\n"
    "logits: min=-6.21 max=14.77 mean=-0.35 std=4.02 nonfinite=0\n"
    "output: top1_prob=0.998 entropy=0.014"
    "\n\nIs this run corrupted? Answer with exactly one word: yes or no."
)


@pytest.fixture(scope="module")
def client():
    return QwenClient(GGUF, n_ctx=4096, n_threads=8)


def test_token_ids(client):
    assert (client.yes_id, client.no_id) == (9693, 2152)


def test_nonthinking_prompt_has_empty_think_block(client):
    p = client.chat_prompt(SYSTEM, USER, thinking=False)
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(client._tokenize(p)) == 283


def test_yes_no_matches_measured(client):
    # Measured with flash attention off: logprob yes -0.9384, no -0.4966.
    p_yes, p_no = client.yes_no_probs(client.chat_prompt(SYSTEM, USER, thinking=False))
    assert math.log(p_yes) == pytest.approx(-0.9384, abs=2e-3)
    assert math.log(p_no) == pytest.approx(-0.4966, abs=2e-3)


def test_one_pass_matches_separate_calls(client):
    prompt = client.chat_prompt(SYSTEM, USER, thinking=False)
    p_yes, p_no, h = client.yes_no_probs_and_hidden(prompt)
    p_yes2, p_no2 = client.yes_no_probs(prompt)
    h2 = client.answer_hidden_state(prompt)
    assert h.shape == (1024,)
    assert p_yes == pytest.approx(p_yes2, abs=1e-5) and p_no == pytest.approx(p_no2, abs=1e-5)
    assert np.allclose(h, h2, atol=1e-4)


def test_thinking_answer(client):
    text, p_yes, p_no = client.generate_then_answer(
        client.chat_prompt(SYSTEM, USER, thinking=True), max_tokens=1500)
    assert "</think>" in text, text[-300:]
    assert p_yes is not None and p_yes + p_no > 0.99


def test_prompt_longer_than_512_tokens(client):
    # llama.cpp's default n_batch is 512; longer prompts used to abort the process.
    long_user = USER.replace("\n\nIs this", "\n" + "\n".join([USER.splitlines()[0]] * 20) + "\n\nIs this")
    prompt = client.chat_prompt(SYSTEM, long_user, thinking=False)
    assert len(client._tokenize(prompt)) > 600
    p_yes, p_no = client.yes_no_probs(prompt)
    assert p_yes + p_no > 0.9
