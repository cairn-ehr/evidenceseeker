# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frontier-authored P5 cases for the first (screen) spike.

The generator records the class it was asked to construct as ``intended_class``
but that label is NEVER shown to a judge. Hand-proofread the emitted JSON before
running the comparison. Disposable: Stage 2 replaces this with real literature.

Scope: every generated case uses the ``non_inferiority`` archetype, so this
screen assesses P5 viability for non-inferiority framing only — false-support
behavior may differ for harm/superiority frames.

    uv run python spikes/p5_viability/generate_cases.py \
        --out spikes/p5_viability/cases/pool.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bmlib.agents import BaseAgent  # noqa: E402
from bmlib.llm import LLMClient, LLMMessage  # noqa: E402

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.contracts import PicoFrame, SupportJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, save_cases  # noqa: E402

_GUIDANCE = {
    SupportJudgment.SUPPORTS: "the passage directly substantiates the claim for the exact PICO",
    SupportJudgment.PARTIAL: "the passage is relevant but mismatched on population, dose, comparator, or outcome",
    SupportJudgment.DOES_NOT: "the passage is off-target and does not substantiate the claim",
    SupportJudgment.CONTRADICTS: "the passage provides evidence AGAINST the claim for this PICO",
}

# Failure modes the gold set over-samples. Both are correctly judged "partial".
FAILURE_MODES: dict[str, str] = {
    "significance_overstatement": (
        "the PICO matches and the effect direction matches, BUT a claimed outcome is NOT "
        "statistically significant (95% CI crosses 1.0 / p>0.05) while the CLAIM asserts it "
        "is 'significant' — so the claim overstates significance"
    ),
    "applicability_mismatch": (
        "the trial explicitly EXCLUDED or never enrolled the population the CLAIM is about "
        "(e.g. the claim says renal impairment but the trial excluded CrCl<30), so the "
        "evidence does not generalize to the claimed population"
    ),
}

_JSON_SCHEMA = (
    "Return ONLY JSON. comorbidities is a list of plain condition-name strings (may be "
    'empty), e.g. ["type 2 diabetes", "hypertension"]:\n'
    '{"cases": [{"pico": {"population": {"age_band": str, "sex": str, "settings": [str], '
    '"comorbidities": [str]}, "intervention": {"label": str, "dose": str|null}, '
    '"comparator": {"label": str}, "outcomes": [{"label": str, "type": '
    '"efficacy|harm|surrogate"}], "archetype": "non_inferiority", "question_text": str, '
    '"applicability": str}, "claim": str, "passage": str, "notes": str}]}'
)


def build_generation_prompt(target: SupportJudgment, n: int) -> str:
    return (
        f"Author {n} realistic clinical evidence cases where the relationship between the "
        f"CLAIM and the PASSAGE is '{target.value}': {_GUIDANCE[target]}.\n"
        "Make the near-misses subtle and adversarial (e.g. right drug but wrong population "
        "or dose). Each case needs a full PICO frame.\n\n" + _JSON_SCHEMA
    )


def build_failure_mode_prompt(mode: str, n: int) -> str:
    return (
        f"Author {n} realistic clinical evidence cases that are each a '{mode}' near-miss: "
        f"{FAILURE_MODES[mode]}. The correct judgment for every case is 'partial'. Make them "
        "subtle and adversarial; each needs a full PICO frame.\n\n" + _JSON_SCHEMA
    )


def _coerce_pico(pico: dict[str, Any]) -> dict[str, Any]:
    """Make a generated PICO dict validate-able. The model emits comorbidities
    as bare condition names; ``CodedTerm`` needs ``{system, code, display}``, so
    wrap each string (the spike only renders ``display``)."""
    population = pico.get("population")
    if isinstance(population, dict):
        comorbidities = population.get("comorbidities")
        if isinstance(comorbidities, list):
            population["comorbidities"] = [
                {"system": "freetext", "code": "", "display": c} if isinstance(c, str) else c
                for c in comorbidities
            ]
    return pico


def parse_generated(
    data: dict[str, Any],
    intended_class: SupportJudgment,
    *,
    failure_mode: str | None = None,
) -> list[P5Case]:
    prefix = failure_mode or intended_class.value
    out: list[P5Case] = []
    for i, raw in enumerate(data["cases"]):
        out.append(
            P5Case(
                id=f"{prefix}-{i}",
                pico=PicoFrame.model_validate(_coerce_pico(raw["pico"])),
                claim=str(raw["claim"]),
                passage=str(raw["passage"]),
                intended_class=intended_class,
                failure_mode=failure_mode,
                notes=raw.get("notes"),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = P5SpikeConfig()
    llm = LLMClient()
    # case-authoring wants variety, hence cfg.generator_temperature; judges run
    # greedy at cfg.temperature (0.0).
    agent = BaseAgent(llm=llm, model=cfg.generator_model, temperature=cfg.generator_temperature)

    all_cases: list[P5Case] = []
    for target in SupportJudgment:
        prompt = build_generation_prompt(target, cfg.cases_per_class)
        data = agent.chat_json([LLMMessage(role="user", content=prompt)])
        all_cases.extend(parse_generated(data, target))
        print(f"generated {cfg.cases_per_class} baseline cases for {target.value}", file=sys.stderr)
    for mode in FAILURE_MODES:
        prompt = build_failure_mode_prompt(mode, cfg.cases_per_mode)
        data = agent.chat_json([LLMMessage(role="user", content=prompt)])
        all_cases.extend(parse_generated(data, SupportJudgment.PARTIAL, failure_mode=mode))
        print(f"generated {cfg.cases_per_mode} {mode} cases", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_cases(all_cases, args.out)
    print(f"wrote {len(all_cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
