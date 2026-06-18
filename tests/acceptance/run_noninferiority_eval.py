# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-inferiority acceptance harness (walking skeleton).

Mirrors localmail's ``run_recall_eval.py`` pattern: load a clinician-authored
gold set, run an injected advisor over each question, apply the whole-report
rubric from the design doc (§9), and gate on the pass rate.

The rubric here is *structural* — it checks a report is shaped like a
trustworthy non-inferiority report (citations verified, balance present, harms
separated from efficacy, margin + population stated, bias/funding flagged,
uncertainty present, and out-of-scope questions correctly declined). Semantic
correctness (does the bottom line match the reference answer?) is a separate
human / LLM-judge pass and is deliberately NOT done here.

Run end-to-end today with the trivial stub advisor::

    PYTHONPATH=src python tests/acceptance/run_noninferiority_eval.py \
        --gold tests/acceptance/gold/noninferiority.example.json --dry-run

The stub declines everything, so in-scope items fail the gate and the
out-of-scope decoy passes — proving the machinery before a real advisor exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidenceseeker.config import EvalConfig  # noqa: E402
from evidenceseeker.contracts import (  # noqa: E402
    AdvisoryReport,
    Competence,
    SupportJudgment,
)


class Advisor(Protocol):
    def __call__(self, question_text: str) -> AdvisoryReport: ...


class DeclineEverythingAdvisor:
    """Trivial stand-in until a real advisor exists; declines every question."""

    def __call__(self, question_text: str) -> AdvisoryReport:
        return AdvisoryReport.declined(
            question_text, reason="stub advisor: no reasoning primitives wired yet"
        )


class Outcome(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NA = "n/a"


@dataclass(frozen=True)
class GoldItem:
    id: str
    question_text: str
    in_scope: bool

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "GoldItem":
        return cls(
            id=raw["id"],
            question_text=raw["question_text"],
            in_scope=bool(raw["expected"]["in_scope"]),
        )


RubricCheck = Callable[[GoldItem, AdvisoryReport, EvalConfig], Outcome]


def check_competence(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    expected_decline = not gold.in_scope
    declined = r.competence is Competence.DECLINED
    return Outcome.PASS if declined == expected_decline else Outcome.FAIL


def _na_if_not_in_scope(gold: GoldItem, r: AdvisoryReport) -> Outcome | None:
    if not gold.in_scope or r.competence is Competence.DECLINED:
        return Outcome.NA
    return None


def check_citations_verified(gold: GoldItem, r: AdvisoryReport, cfg: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    if not r.claims:
        return Outcome.FAIL
    for claim in r.claims:
        supporting = [c for c in claim.citations if c.support is SupportJudgment.SUPPORTS]
        if len(supporting) < cfg.min_supporting_citations_per_claim:
            return Outcome.FAIL
        if any(c.support is not SupportJudgment.SUPPORTS for c in claim.citations):
            return Outcome.FAIL
        if any(c.support is not SupportJudgment.CONTRADICTS for c in claim.counter_citations):
            return Outcome.FAIL
    return Outcome.PASS


def check_counterfactual_present(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    b = r.balance
    if b is None:
        return Outcome.FAIL
    if b.contradicting or b.null_or_negative:
        return Outcome.PASS
    return Outcome.PASS if (b.none_found and b.searched) else Outcome.FAIL


def check_harms_separated(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    return Outcome.PASS if r.harms is not None else Outcome.FAIL


def check_ni_margin_stated(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    ni = r.non_inferiority
    if ni is None or not ni.margin:
        return Outcome.FAIL
    population_stated = bool(r.pico_frame and r.pico_frame.applicability)
    return Outcome.PASS if population_stated else Outcome.FAIL


def check_bias_funding_flagged(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    return Outcome.PASS if r.bias_funding is not None else Outcome.FAIL


def check_uncertainty(gold: GoldItem, r: AdvisoryReport, _: EvalConfig) -> Outcome:
    if (na := _na_if_not_in_scope(gold, r)) is not None:
        return na
    return Outcome.PASS if r.uncertainty else Outcome.FAIL


RUBRIC: dict[str, RubricCheck] = {
    "competence_correct": check_competence,
    "citations_verified": check_citations_verified,
    "counterfactual_present": check_counterfactual_present,
    "harms_separated": check_harms_separated,
    "ni_margin_and_population": check_ni_margin_stated,
    "bias_funding_flagged": check_bias_funding_flagged,
    "uncertainty_calibrated": check_uncertainty,
}


@dataclass(frozen=True)
class ItemResult:
    gold: GoldItem
    outcomes: dict[str, Outcome]

    @property
    def passed(self) -> bool:
        return all(o is not Outcome.FAIL for o in self.outcomes.values())


def evaluate(gold: list[GoldItem], advisor: Advisor, cfg: EvalConfig) -> list[ItemResult]:
    results: list[ItemResult] = []
    for item in gold:
        report = advisor(item.question_text)
        outcomes = {name: check(item, report, cfg) for name, check in RUBRIC.items()}
        results.append(ItemResult(gold=item, outcomes=outcomes))
    return results


def _print_text(results: list[ItemResult], cfg: EvalConfig, pass_rate: float) -> None:
    for res in results:
        flag = "PASS" if res.passed else "FAIL"
        print(f"\n[{flag}] {res.gold.id}  (in_scope={res.gold.in_scope})")
        print(f"       {res.gold.question_text}")
        for name, outcome in res.outcomes.items():
            print(f"         - {name:<28} {outcome.value}")
    n = len(results)
    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 60}")
    print(f"reports passing: {n_pass}/{n}  (pass_rate={pass_rate:.2f}, gate={cfg.min_pass_rate:.2f})")


def _to_json(results: list[ItemResult], pass_rate: float) -> dict[str, Any]:
    return {
        "pass_rate": pass_rate,
        "results": [
            {
                "id": r.gold.id,
                "in_scope": r.gold.in_scope,
                "passed": r.passed,
                "outcomes": {k: v.value for k, v in r.outcomes.items()},
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, required=True, help="gold-set JSON file")
    ap.add_argument("--dry-run", action="store_true", help="use the decline-everything stub advisor")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    gold = [GoldItem.from_json(x) for x in json.loads(args.gold.read_text())]

    if not args.dry_run:
        print("no real advisor wired yet; re-run with --dry-run", file=sys.stderr)
        return 2

    results = evaluate(gold, DeclineEverythingAdvisor(), cfg)
    pass_rate = (sum(1 for r in results if r.passed) / len(results)) if results else 0.0

    if args.format == "json":
        print(json.dumps(_to_json(results, pass_rate), indent=2))
    else:
        _print_text(results, cfg, pass_rate)

    return 0 if pass_rate >= cfg.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
