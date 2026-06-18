# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared builders for tests."""

from __future__ import annotations

from evidenceseeker.contracts import (
    Archetype,
    Outcome,
    OutcomeType,
    PicoFrame,
    Population,
    Therapy,
)


def make_pico(
    *,
    intervention: str = "DrugA",
    comparator: str = "DrugB",
    dose: str | None = None,
) -> PicoFrame:
    return PicoFrame(
        population=Population(age_band="65-74", sex="any", settings=["outpatient"]),
        intervention=Therapy(label=intervention, dose=dose),
        comparator=Therapy(label=comparator),
        outcomes=[Outcome(label="all-cause mortality", type=OutcomeType.EFFICACY)],
        archetype=Archetype.NON_INFERIORITY,
        question_text="Is DrugA non-inferior to DrugB for mortality?",
        applicability="community outpatients aged 65-74",
    )
