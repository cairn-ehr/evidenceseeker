# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import pytest

from evidenceseeker.contracts import SupportJudgment as S
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.worksheet import apply_labels, parse_worksheet, render_worksheet
from tests._helpers import make_pico


def _case(cid: str, cls: S) -> P5Case:
    return P5Case(id=cid, pico=make_pico(), claim="the claim", passage="the passage", intended_class=cls)


def _j(s: S) -> CitationJudgment:
    return CitationJudgment(support=s, reason="because")


def test_render_then_parse_recovers_frontier_proposals() -> None:
    cases = [_case("supports-0", S.SUPPORTS), _case("applicability_mismatch-0", S.PARTIAL)]
    judgments = [_j(S.SUPPORTS), _j(S.PARTIAL)]
    md = render_worksheet(cases, judgments)
    labels = parse_worksheet(md)
    assert labels == {"supports-0": S.SUPPORTS, "applicability_mismatch-0": S.PARTIAL}


def test_parse_picks_up_a_human_override() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)]).replace("gold: supports", "gold: partial")
    assert parse_worksheet(md) == {"supports-0": S.PARTIAL}


def test_parse_rejects_invalid_label() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)]).replace("gold: supports", "gold: bogus")
    with pytest.raises(ValueError):
        parse_worksheet(md)


def test_apply_labels_sets_gold_and_validates_coverage() -> None:
    cases = [_case("supports-0", S.SUPPORTS), _case("partial-0", S.PARTIAL)]
    labelled = apply_labels(cases, {"supports-0": S.DOES_NOT, "partial-0": S.PARTIAL})
    assert labelled[0].gold_label is S.DOES_NOT
    assert labelled[1].gold_label is S.PARTIAL
    with pytest.raises(ValueError, match="no gold label"):
        apply_labels(cases, {"supports-0": S.DOES_NOT})  # missing partial-0
    with pytest.raises(ValueError, match="unknown"):
        apply_labels(cases, {"supports-0": S.DOES_NOT, "partial-0": S.PARTIAL, "ghost-9": S.SUPPORTS})
