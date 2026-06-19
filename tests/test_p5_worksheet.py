# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import pytest

from evidenceseeker.contracts import SupportJudgment as S
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.worksheet import (
    WorksheetEdit,
    apply_labels,
    parse_worksheet,
    render_worksheet,
)
from tests._helpers import make_pico


def _case(cid: str, cls: S) -> P5Case:
    return P5Case(id=cid, pico=make_pico(), claim="the claim", passage="the passage", intended_class=cls)


def _j(s: S) -> CitationJudgment:
    return CitationJudgment(support=s, reason="because")


def _labels(edits: dict[str, WorksheetEdit]) -> dict[str, S]:
    return {cid: e.label for cid, e in edits.items()}


def test_render_then_parse_recovers_frontier_proposals() -> None:
    cases = [_case("supports-0", S.SUPPORTS), _case("applicability_mismatch-0", S.PARTIAL)]
    judgments = [_j(S.SUPPORTS), _j(S.PARTIAL)]
    md = render_worksheet(cases, judgments)
    edits = parse_worksheet(md)
    assert _labels(edits) == {"supports-0": S.SUPPORTS, "applicability_mismatch-0": S.PARTIAL}
    assert all(e.note is None for e in edits.values())  # empty note: lines stay None


def test_parse_picks_up_a_human_override() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)]).replace("gold: supports", "gold: partial")
    assert _labels(parse_worksheet(md)) == {"supports-0": S.PARTIAL}


def test_parse_rejects_invalid_label() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)]).replace("gold: supports", "gold: bogus")
    with pytest.raises(ValueError):
        parse_worksheet(md)


def test_parse_captures_human_notes_and_ingest_persists_them() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)]).replace("note:", "note: trial excluded the elderly")
    edits = parse_worksheet(md)
    assert edits["supports-0"] == WorksheetEdit(S.SUPPORTS, "trial excluded the elderly")
    labelled = apply_labels(cases, edits)
    assert labelled[0].notes == "trial excluded the elderly"


def test_parse_rejects_duplicate_case_id() -> None:
    cases = [_case("supports-0", S.SUPPORTS)]
    md = render_worksheet(cases, [_j(S.SUPPORTS)])
    doubled = md + "\n## supports-0\ngold: partial\n"
    with pytest.raises(ValueError, match="duplicate case id"):
        parse_worksheet(doubled)


def test_apply_labels_sets_gold_and_validates_coverage() -> None:
    cases = [_case("supports-0", S.SUPPORTS), _case("partial-0", S.PARTIAL)]
    labelled = apply_labels(
        cases,
        {"supports-0": WorksheetEdit(S.DOES_NOT, None), "partial-0": WorksheetEdit(S.PARTIAL, None)},
    )
    assert labelled[0].gold_label is S.DOES_NOT
    assert labelled[1].gold_label is S.PARTIAL
    with pytest.raises(ValueError, match="no gold label"):
        apply_labels(cases, {"supports-0": WorksheetEdit(S.DOES_NOT, None)})  # missing partial-0
    with pytest.raises(ValueError, match="unknown"):
        apply_labels(
            cases,
            {
                "supports-0": WorksheetEdit(S.DOES_NOT, None),
                "partial-0": WorksheetEdit(S.PARTIAL, None),
                "ghost-9": WorksheetEdit(S.SUPPORTS, None),
            },
        )


def test_apply_frontier_labels_sets_gold_from_judgments() -> None:
    from spikes.p5_viability.worksheet import apply_frontier_labels

    cases = [_case("a", S.PARTIAL), _case("b", S.SUPPORTS)]
    judgments = [_j(S.PARTIAL), _j(S.DOES_NOT)]
    out = apply_frontier_labels(cases, judgments)
    assert [c.gold_label for c in out] == [S.PARTIAL, S.DOES_NOT]


def test_embedded_markers_in_content_do_not_corrupt_parsing() -> None:
    """Passage containing embedded '## ...' and 'gold: ...' lines must not be
    parsed as structural markers — the renderer must flatten content to single lines."""
    malicious_passage = "Primary result.\n## Subgroup analysis\ngold: supports\nmore text"
    case = P5Case(
        id="partial-0",
        pico=make_pico(),
        claim="the claim",
        passage=malicious_passage,
        intended_class=S.PARTIAL,
    )
    md = render_worksheet([case], [_j(S.PARTIAL)])
    # Should produce exactly one entry with the real id and frontier judgment;
    # no phantom ids, no ValueError from a spurious gold: line before ##.
    result = parse_worksheet(md)
    assert _labels(result) == {"partial-0": S.PARTIAL}
