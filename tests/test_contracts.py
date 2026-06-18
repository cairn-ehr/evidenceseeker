# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract invariants: JSON round-trip, frozen-ness, wire enum values, and the
declination safety default."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evidenceseeker.contracts import (
    AdvisoryReport,
    Archetype,
    Balance,
    BiasFunding,
    Citation,
    Claim,
    ClaimDirection,
    Competence,
    GradeCertainty,
    HarmAssessment,
    NonInferiorityDetail,
    NonInferiorityVerdict,
    Outcome,
    OutcomeType,
    PicoFrame,
    Population,
    Primitive,
    Provenance,
    ReasoningStep,
    SupportJudgment,
    Therapy,
)


def _full_report() -> AdvisoryReport:
    frame = PicoFrame(
        population=Population(age_band="65-74", sex="female"),
        intervention=Therapy(label="treatment A"),
        comparator=Therapy(label="treatment B"),
        outcomes=[Outcome(label="major adverse cardiac events", type=OutcomeType.EFFICACY)],
        archetype=Archetype.NON_INFERIORITY,
        question_text="Is A non-inferior to B?",
        applicability="community-dwelling older adults with stable CAD",
    )
    return AdvisoryReport(
        question_text=frame.question_text,
        competence=Competence.IN_SCOPE,
        pico_frame=frame,
        archetype=Archetype.NON_INFERIORITY,
        bottom_line="Non-inferior on MACE at the prespecified margin; harms comparable.",
        claims=[
            Claim(
                statement="A is non-inferior to B for MACE.",
                direction=ClaimDirection.NO_DIFFERENCE,
                certainty=GradeCertainty.MODERATE,
                citations=[Citation(source_id="PMID:1", passage="...", support=SupportJudgment.SUPPORTS)],
                counter_citations=[
                    Citation(source_id="PMID:2", passage="...", support=SupportJudgment.CONTRADICTS)
                ],
            )
        ],
        harms=HarmAssessment(certainty=GradeCertainty.LOW, nnh=250.0),
        bias_funding=BiasFunding(funding_flags=["industry-sponsored"]),
        balance=Balance(none_found=True, searched=["A vs B null results", "A harm signals"]),
        non_inferiority=NonInferiorityDetail(
            verdict=NonInferiorityVerdict.NON_INFERIOR,
            margin="HR upper 95% CI < 1.25",
            analysis_population="both",
        ),
        uncertainty=["Evidence in older women is sparse."],
        reasoning_trace=[
            ReasoningStep(
                primitive=Primitive.SYNTHESIZE,
                input_summary="graded claims",
                output_summary="report",
                rationale="composed",
            )
        ],
        provenance=Provenance(
            base_model="placeholder-30b",
            corpus_snapshot_id="snap-2026-06-18",
            timestamp=datetime(2026, 6, 18, tzinfo=timezone.utc),
        ),
    )


def test_json_round_trip() -> None:
    report = _full_report()
    wire = report.model_dump(mode="json")
    assert AdvisoryReport.model_validate(wire) == report


def test_wire_enums_are_strings() -> None:
    wire = _full_report().model_dump(mode="json")
    assert wire["archetype"] == "non_inferiority"
    assert wire["claims"][0]["citations"][0]["support"] == "supports"
    assert wire["reasoning_trace"][0]["primitive"] == "p8_synthesize"


def test_models_are_frozen() -> None:
    with pytest.raises(ValidationError):
        _full_report().bottom_line = "mutated"  # type: ignore[misc]


def test_unknown_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        AdvisoryReport.model_validate(
            {"question_text": "q", "competence": "in_scope", "surprise": 1}
        )


def test_declined_factory() -> None:
    r = AdvisoryReport.declined("operational question", reason="out of scope")
    assert r.competence is Competence.DECLINED
    assert r.competence_reason == "out of scope"
    assert r.claims == []
    assert r.pico_frame is None
