# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment as S
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.scoring import ModelScore, render_review, score_run
from tests._helpers import make_pico


def _case(cid: str, intended: S, gold: S | None = None) -> P5Case:
    return P5Case(
        id=cid,
        pico=make_pico(),
        claim="DrugA is non-inferior to DrugB.",
        passage="Some passage.",
        intended_class=intended,
        gold_label=gold,
    )


def _j(support: S) -> CitationJudgment:
    return CitationJudgment(support=support, reason="r")


def test_score_run_uses_frontier_as_reference_and_flags_false_support() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    judgments = {
        "frontier": [_j(S.DOES_NOT), _j(S.SUPPORTS)],   # reference
        "local": [_j(S.SUPPORTS), _j(S.SUPPORTS)],      # false-supports case "a"
    }
    scores = score_run(cases, judgments, reference_model="frontier")
    by_model = {s.model: s for s in scores}

    assert isinstance(by_model["local"], ModelScore)
    assert by_model["local"].false_support_rate == 1.0   # 1 negative, called supports
    assert by_model["frontier"].false_support_rate == 0.0
    assert by_model["frontier"].kappa_vs_frontier == 1.0


def test_score_run_prefers_gold_label_over_frontier() -> None:
    # Frontier says SUPPORTS but the human gold says DOES_NOT -> "local" matching
    # frontier is now a false support against gold.
    cases = [_case("a", S.SUPPORTS, gold=S.DOES_NOT)]
    judgments = {
        "frontier": [_j(S.SUPPORTS)],
        "local": [_j(S.SUPPORTS)],
    }
    scores = {s.model: s for s in score_run(cases, judgments, "frontier")}
    assert scores["local"].false_support_rate == 1.0
    assert scores["frontier"].false_support_rate == 1.0


def test_score_run_flags_only_the_reference_row() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    judgments = {
        "frontier": [_j(S.DOES_NOT), _j(S.SUPPORTS)],
        "local": [_j(S.SUPPORTS), _j(S.SUPPORTS)],
    }
    scores = {s.model: s for s in score_run(cases, judgments, reference_model="frontier")}
    assert scores["frontier"].is_reference is True
    assert scores["local"].is_reference is False


def test_render_review_includes_claim_and_each_models_judgment() -> None:
    cases = [_case("a", S.DOES_NOT)]
    judgments = {"frontier": [_j(S.DOES_NOT)], "local": [_j(S.SUPPORTS)]}
    md = render_review(cases, judgments, "frontier")
    assert "DrugA is non-inferior to DrugB." in md
    assert "local" in md and "supports" in md
    assert "frontier" in md and "does_not" in md
