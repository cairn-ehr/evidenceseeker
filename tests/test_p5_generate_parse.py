# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment
from spikes.p5_viability.generate_cases import build_generation_prompt, parse_generated


def test_build_generation_prompt_names_target_and_count() -> None:
    prompt = build_generation_prompt(SupportJudgment.PARTIAL, 3)
    assert "partial" in prompt
    assert "3" in prompt


def test_parse_generated_attaches_id_and_intended_class() -> None:
    data = {
        "cases": [
            {
                "pico": {
                    "population": {"age_band": "65-74", "sex": "any", "settings": ["outpatient"]},
                    "intervention": {"label": "DrugA", "dose": "10mg"},
                    "comparator": {"label": "DrugB"},
                    "outcomes": [{"label": "mortality", "type": "efficacy"}],
                    "archetype": "non_inferiority",
                    "question_text": "Is DrugA non-inferior to DrugB?",
                    "applicability": "outpatients 65-74",
                },
                "claim": "DrugA is non-inferior to DrugB.",
                "passage": "A pediatric trial found DrugA non-inferior to DrugB.",
                "notes": "population mismatch (pediatric vs elderly)",
            }
        ]
    }
    cases = parse_generated(data, SupportJudgment.DOES_NOT)
    assert len(cases) == 1
    assert cases[0].id == "does_not-0"
    assert cases[0].intended_class is SupportJudgment.DOES_NOT
    assert cases[0].pico.intervention.label == "DrugA"
    assert cases[0].notes is not None


def test_parse_generated_coerces_freetext_comorbidities() -> None:
    # The frontier model emits comorbidities as bare strings, not CodedTerms.
    data = {
        "cases": [
            {
                "pico": {
                    "population": {
                        "age_band": "65-74",
                        "sex": "any",
                        "settings": ["outpatient"],
                        "comorbidities": ["type 2 diabetes", "hypertension"],
                    },
                    "intervention": {"label": "DrugA", "dose": "10mg"},
                    "comparator": {"label": "DrugB"},
                    "outcomes": [{"label": "mortality", "type": "efficacy"}],
                    "archetype": "non_inferiority",
                    "question_text": "Is DrugA non-inferior to DrugB?",
                    "applicability": "outpatients 65-74",
                },
                "claim": "c",
                "passage": "p",
            }
        ]
    }
    cases = parse_generated(data, SupportJudgment.PARTIAL)
    displays = [c.display for c in cases[0].pico.population.comorbidities]
    assert displays == ["type 2 diabetes", "hypertension"]
