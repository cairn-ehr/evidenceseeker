# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 spike case schema and JSON persistence.

A case's reference label is its human ``gold_label`` if present, else the
frontier model's blind judgment — see ``reference_label``. This is the seam
that lets a future gold set slot in with no runner change.
"""

from __future__ import annotations

import json
from pathlib import Path

from evidenceseeker.contracts import PicoFrame, SupportJudgment, _Frozen


class P5Case(_Frozen):
    id: str
    pico: PicoFrame
    claim: str
    passage: str
    intended_class: SupportJudgment
    gold_label: SupportJudgment | None = None
    notes: str | None = None


def load_cases(path: Path) -> list[P5Case]:
    raw = json.loads(Path(path).read_text())
    return [P5Case.model_validate(item) for item in raw]


def save_cases(cases: list[P5Case], path: Path) -> None:
    payload = [c.model_dump(mode="json") for c in cases]
    Path(path).write_text(json.dumps(payload, indent=2))


def reference_label(case: P5Case, frontier: SupportJudgment) -> SupportJudgment:
    """Human gold label when present, else the frontier model's judgment."""
    return case.gold_label if case.gold_label is not None else frontier
