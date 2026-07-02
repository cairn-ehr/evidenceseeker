# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import pytest

from spikes.p5_viability.env_check import require_api_keys


def test_missing_anthropic_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        require_api_keys(["anthropic:claude-sonnet-4-6", "ollama:phi4:14b"])


def test_present_anthropic_key_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    require_api_keys(["anthropic:claude-sonnet-4-6"])  # no raise


def test_local_only_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    require_api_keys(["ollama:phi4:14b", "ollama:gemma4:12b-it-qat"])  # no raise
