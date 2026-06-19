# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from evidenceseeker.contracts import SupportJudgment
from spikes.p5_viability.cases import (
    P5Case,
    load_cases,
    reference_label,
    save_cases,
)
from tests._helpers import make_pico


def _case(**overrides: object) -> P5Case:
    base = dict(
        id="supports-0",
        pico=make_pico(),
        claim="DrugA is non-inferior to DrugB for mortality.",
        passage="In a non-inferiority RCT, DrugA was non-inferior to DrugB.",
        intended_class=SupportJudgment.SUPPORTS,
    )
    base.update(overrides)
    return P5Case(**base)  # type: ignore[arg-type]


def test_roundtrip_preserves_fields(tmp_path: Path) -> None:
    cases = [_case(), _case(id="partial-0", intended_class=SupportJudgment.PARTIAL)]
    path = tmp_path / "cases.json"
    save_cases(cases, path)
    loaded = load_cases(path)
    assert loaded == cases


def test_reference_label_prefers_gold() -> None:
    case = _case(gold_label=SupportJudgment.DOES_NOT)
    assert reference_label(case, SupportJudgment.SUPPORTS) is SupportJudgment.DOES_NOT


def test_reference_label_falls_back_to_frontier() -> None:
    case = _case(gold_label=None)
    assert reference_label(case, SupportJudgment.CONTRADICTS) is SupportJudgment.CONTRADICTS


def test_p5case_failure_mode_defaults_none_and_roundtrips() -> None:
    from evidenceseeker.contracts import SupportJudgment as S

    from spikes.p5_viability.cases import P5Case
    from tests._helpers import make_pico

    tagged = P5Case(
        id="x", pico=make_pico(), claim="c", passage="p",
        intended_class=S.PARTIAL, failure_mode="significance_overstatement",
    )
    assert tagged.failure_mode == "significance_overstatement"
    baseline = P5Case(
        id="y", pico=make_pico(), claim="c", passage="p", intended_class=S.SUPPORTS,
    )
    assert baseline.failure_mode is None
