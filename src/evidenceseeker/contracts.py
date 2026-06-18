# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed contracts for the clinical evidence advisory service.

This is the surface the broker (Kastellan) codes against: a de-identified
``PicoFrame`` goes in, a fully-graded, cited, auditable ``AdvisoryReport`` comes
out. Every field maps to a step in the reasoning-primitive catalog
(``docs/design/2026-06-18-reasoning-primitive-catalog.md``); the ``Primitive``
enum names the eight primitives P1..P8.

Models are frozen and forbid unknown fields so the wire contract can't drift
silently. ``model_dump(mode="json")`` is the canonical serialization (StrEnum
members serialize as their string value, datetimes as ISO-8601).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- enumerations ---------------------------------------------------------


class Archetype(StrEnum):
    EXISTENCE = "existence"
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non_inferiority"
    HARM = "harm"
    BEST_PRACTICE = "best_practice"


class OutcomeType(StrEnum):
    EFFICACY = "efficacy"
    HARM = "harm"
    SURROGATE = "surrogate"


class Competence(StrEnum):
    IN_SCOPE = "in_scope"
    LOW_CONFIDENCE = "low_confidence"
    DECLINED = "declined"


class GradeCertainty(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class GradeDomain(StrEnum):
    RISK_OF_BIAS = "risk_of_bias"
    INCONSISTENCY = "inconsistency"
    INDIRECTNESS = "indirectness"
    IMPRECISION = "imprecision"
    PUBLICATION_BIAS = "publication_bias"


class ClaimDirection(StrEnum):
    FAVORS_INTERVENTION = "favors_intervention"
    FAVORS_COMPARATOR = "favors_comparator"
    NO_DIFFERENCE = "no_difference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SupportJudgment(StrEnum):
    SUPPORTS = "supports"
    PARTIAL = "partial"
    DOES_NOT = "does_not"
    CONTRADICTS = "contradicts"


class RobTool(StrEnum):
    ROB2 = "rob2"
    ROBINS_I = "robins_i"


class RobJudgment(StrEnum):
    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"


class StrengthOfRecommendation(StrEnum):
    STRONG_FOR = "strong_for"
    CONDITIONAL_FOR = "conditional_for"
    CONDITIONAL_AGAINST = "conditional_against"
    STRONG_AGAINST = "strong_against"
    NONE = "none"


class NonInferiorityVerdict(StrEnum):
    NON_INFERIOR = "non_inferior"
    NOT_NON_INFERIOR = "not_non_inferior"
    INCONCLUSIVE = "inconclusive"


class Primitive(StrEnum):
    FRAME = "p1_frame"
    RETRIEVE_SCREEN = "p2_retrieve_screen"
    APPRAISE_ROB = "p3_appraise_rob"
    APPRAISE_GRADE = "p4_appraise_grade"
    VERIFY_CITATION = "p5_verify_citation"
    APPRAISE_HARMS = "p6_appraise_harms"
    COUNTERFACTUAL = "p7_counterfactual"
    SYNTHESIZE = "p8_synthesize"


# --- the PICO frame (P1 output, threaded through every primitive) ----------


class CodedTerm(_Frozen):
    system: str
    code: str
    display: str


class Population(_Frozen):
    age_band: str | None = None
    sex: str | None = None
    settings: list[str] = Field(default_factory=list)
    comorbidities: list[CodedTerm] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class Therapy(_Frozen):
    """An intervention or comparator. ``label`` may be ``"placebo"`` /
    ``"usual care"`` / ``"none"`` for existence and harm questions."""

    label: str
    code: CodedTerm | None = None
    dose: str | None = None
    route: str | None = None
    regimen: str | None = None


class Outcome(_Frozen):
    label: str
    type: OutcomeType
    code: CodedTerm | None = None


class PicoFrame(_Frozen):
    population: Population
    intervention: Therapy
    comparator: Therapy
    outcomes: list[Outcome]
    archetype: Archetype
    question_text: str
    # The population/setting the answer must generalize TO. Kept distinct from
    # ``population`` because it is what the GRADE indirectness judgment (P4) is
    # made against — most "wrong citation" errors are applicability errors.
    applicability: str | None = None


# --- report components -----------------------------------------------------


class Citation(_Frozen):
    source_id: str
    passage: str
    support: SupportJudgment
    pico_match_notes: str | None = None


class Claim(_Frozen):
    statement: str
    direction: ClaimDirection
    certainty: GradeCertainty
    citations: list[Citation] = Field(default_factory=list)
    counter_citations: list[Citation] = Field(default_factory=list)


class GradeDomainJudgment(_Frozen):
    domain: GradeDomain
    judgment: str
    rationale: str


class GradeAssessment(_Frozen):
    outcome: str
    certainty: GradeCertainty
    domains: list[GradeDomainJudgment] = Field(default_factory=list)
    upgrades: list[str] = Field(default_factory=list)


class RobDomain(_Frozen):
    name: str
    judgment: RobJudgment
    quote: str | None = None


class RobAssessment(_Frozen):
    source_id: str
    tool: RobTool
    domains: list[RobDomain] = Field(default_factory=list)
    overall: RobJudgment


class HarmAssessment(_Frozen):
    certainty: GradeCertainty
    absolute_risk: str | None = None
    nnh: float | None = None
    sources: list[str] = Field(default_factory=list)
    note: str | None = None


class BiasFunding(_Frozen):
    rob: list[RobAssessment] = Field(default_factory=list)
    funding_flags: list[str] = Field(default_factory=list)
    registration_flags: list[str] = Field(default_factory=list)


class Balance(_Frozen):
    """P7 output. ``none_found=True`` is only acceptable with a non-empty
    ``searched`` trace — an empty balance with no search trace fails the gate."""

    contradicting: list[Citation] = Field(default_factory=list)
    null_or_negative: list[Citation] = Field(default_factory=list)
    searched: list[str] = Field(default_factory=list)
    none_found: bool = False


class NonInferiorityDetail(_Frozen):
    verdict: NonInferiorityVerdict
    margin: str | None = None
    margin_justified: bool | None = None
    assay_sensitivity: str | None = None
    analysis_population: str | None = None  # e.g. "ITT", "per_protocol", "both"
    efficacy_vs_safety_note: str | None = None


class ReasoningStep(_Frozen):
    primitive: Primitive
    input_summary: str
    output_summary: str
    rationale: str


class Provenance(_Frozen):
    base_model: str
    corpus_snapshot_id: str
    timestamp: datetime
    adapter_versions: dict[str, str] = Field(default_factory=dict)


# --- the report ------------------------------------------------------------


class AdvisoryReport(_Frozen):
    question_text: str
    competence: Competence
    competence_reason: str | None = None

    pico_frame: PicoFrame | None = None
    archetype: Archetype | None = None

    bottom_line: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    evidence_quality: list[GradeAssessment] = Field(default_factory=list)
    harms: HarmAssessment | None = None
    bias_funding: BiasFunding | None = None
    balance: Balance | None = None
    certainty_overall: GradeCertainty | None = None
    strength_of_recommendation: StrengthOfRecommendation | None = None
    non_inferiority: NonInferiorityDetail | None = None
    uncertainty: list[str] = Field(default_factory=list)

    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    provenance: Provenance | None = None

    @classmethod
    def declined(cls, question_text: str, reason: str) -> "AdvisoryReport":
        """Honest declination — the safety default when no calibrated recipe
        covers the question (P1). Carries no claims, only the reason."""
        return cls(
            question_text=question_text,
            competence=Competence.DECLINED,
            competence_reason=reason,
        )
