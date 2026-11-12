from dataclasses import fields

import pandas as pd
import pytest

from sdc.detectors.lm.prompts import QUESTION, build_messages
from sdc.model.monitor import MONITORED_POINTS
from sdc.observations.serialize import (
    fmt, observation_columns, output_ranges, serialize_observation, serialize_reference,
)
from sdc.records.schema import RunRecord

# Every RunRecord field except the observation itself: the answer key and bookkeeping.
NOT_OBSERVATIONS = {f.name for f in fields(RunRecord)} - {"top1_prob", "entropy", "layer_stats"}


def make_row(**overrides):
    row = {"top1_prob": 0.9999, "entropy": 0.000946}
    for p in MONITORED_POINTS:
        row.update({f"{p}_min": 0.0, f"{p}_max": 3.231, f"{p}_mean": 0.2669,
                    f"{p}_std": 0.2686, f"{p}_nonfinite": 0})
    # Answer-key columns that must never reach a prompt.
    row.update({"label": "sdc", "flipped_value": 5.678798848251858e37, "clean_pred": 7})
    row.update(overrides)
    return row


def test_observation_columns_hold_no_answer_key():
    assert observation_columns("blackbox") == ["top1_prob", "entropy"]
    whitebox = observation_columns("whitebox")
    assert len(whitebox) == 2 + 5 * len(MONITORED_POINTS)
    assert not set(whitebox) & NOT_OBSERVATIONS


def test_unknown_access_level_raises():
    with pytest.raises(ValueError):
        observation_columns("greybox")


@pytest.mark.parametrize("x, text", [
    (0.0, "0"), (-0.0, "0"), (0.40756, "0.4076"), (1.1033389e38, "1.103e+38"),
    (-11.243, "-11.24"), (float("nan"), "nan"),
])
def test_fmt(x, text):
    assert fmt(x) == text


def test_serializer_ignores_answer_key_columns():
    for level in ("blackbox", "whitebox"):
        text = serialize_observation(make_row(), level)
        assert "sdc" not in text and "5.679e+37" not in text and "clean" not in text


def test_whitebox_lines():
    lines = serialize_observation(make_row(), "whitebox").splitlines()
    assert lines[0] == "stem: min=0 max=3.231 mean=0.2669 std=0.2686 nonfinite=0"
    assert lines[-1] == "output: top1_prob=0.9999 entropy=0.000946"
    assert len(lines) == len(MONITORED_POINTS) + 1


def test_blackbox_is_output_line_only():
    assert serialize_observation(make_row(), "blackbox") == "output: top1_prob=0.9999 entropy=0.000946"


def test_reference_matches_access_level():
    layer_ranges = {p: {"min": 0.0, "max": 8.205926} for p in MONITORED_POINTS}
    clean = pd.DataFrame({"top1_prob": [0.5146, 0.99], "entropy": [0.01, 1.544]})
    out = output_ranges(clean)
    assert out == {"top1_prob": {"min": 0.5146, "max": 0.99}, "entropy": {"min": 0.01, "max": 1.544}}
    black = serialize_reference(layer_ranges, out, "blackbox").splitlines()
    white = serialize_reference(layer_ranges, out, "whitebox").splitlines()
    assert black == ["top1_prob: min=0.5146 max=0.99", "entropy: min=0.01 max=1.544"]
    assert white[0] == "stem: min=0 max=8.206"
    assert white[-2:] == black


def test_build_messages():
    _, user = build_messages("OBS", "whitebox")
    assert user == f"This run:\nOBS\n\n{QUESTION}"
    system, user = build_messages("OBS", "blackbox", reference_text="REF")
    assert user.startswith("Normal ranges, from fault-free runs:\nREF\n\nThis run:\nOBS")
    assert "six points" not in system
