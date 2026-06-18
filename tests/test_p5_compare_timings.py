# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from spikes.p5_viability.compare_model import _load_timings, _persist_timings


def test_timings_round_trip_and_merge(tmp_path: Path) -> None:
    _persist_timings(tmp_path, {"ollama:A": 12.5})
    _persist_timings(tmp_path, {"ollama:B": 7.0})  # second run must not clobber A
    assert _load_timings(tmp_path) == {"ollama:A": 12.5, "ollama:B": 7.0}
    _persist_timings(tmp_path, {"ollama:A": 99.0})  # re-run refreshes only A
    assert _load_timings(tmp_path) == {"ollama:A": 99.0, "ollama:B": 7.0}


def test_load_timings_absent_is_empty(tmp_path: Path) -> None:
    assert _load_timings(tmp_path) == {}
