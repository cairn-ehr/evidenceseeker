# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import pytest

from evidenceseeker.config import P5SpikeConfig


def test_defaults_have_distinct_reference_and_judges() -> None:
    cfg = P5SpikeConfig()
    assert cfg.reference_model not in cfg.judge_models


def test_reference_model_overlapping_a_judge_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not also appear in judge_models"):
        P5SpikeConfig(judge_models=["ollama:x"], reference_model="ollama:x")


def test_generator_runs_hotter_than_judges_by_default() -> None:
    cfg = P5SpikeConfig()
    assert cfg.temperature == 0.0
    assert cfg.generator_temperature > cfg.temperature
