# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score a P5 viability run and render a human-proofreadable review artifact.

Pure: no LLM, no I/O. Live judging lives in ``run_viability.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidenceseeker.contracts import SupportJudgment
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability import metrics
from spikes.p5_viability.cases import P5Case, reference_label


@dataclass(frozen=True)
class ModelScore:
    model: str
    n: int
    accuracy: float
    false_support_rate: float
    missed_support_rate: float
    kappa_vs_frontier: float


def _reference_labels(
    cases: list[P5Case], frontier: list[CitationJudgment]
) -> list[SupportJudgment]:
    return [
        reference_label(case, fj.support)
        for case, fj in zip(cases, frontier, strict=True)
    ]


def score_run(
    cases: list[P5Case],
    judgments: dict[str, list[CitationJudgment]],
    reference_model: str,
) -> list[ModelScore]:
    frontier = judgments[reference_model]
    reference = _reference_labels(cases, frontier)
    frontier_supports = [j.support for j in frontier]

    scores: list[ModelScore] = []
    for model, judged in judgments.items():
        supports = [j.support for j in judged]
        scores.append(
            ModelScore(
                model=model,
                n=len(cases),
                accuracy=metrics.accuracy(supports, reference),
                false_support_rate=metrics.false_support_rate(supports, reference),
                missed_support_rate=metrics.missed_support_rate(supports, reference),
                kappa_vs_frontier=metrics.cohen_kappa(supports, frontier_supports),
            )
        )
    return scores


def render_review(
    cases: list[P5Case],
    judgments: dict[str, list[CitationJudgment]],
    reference_model: str,
) -> str:
    lines: list[str] = ["# P5 viability review\n"]
    for idx, case in enumerate(cases):
        ref = reference_label(case, judgments[reference_model][idx].support)
        lines.append(f"## {case.id}  (intended={case.intended_class.value}, reference={ref.value})")
        lines.append(f"- **claim:** {case.claim}")
        lines.append(f"- **passage:** {case.passage}")
        lines.append(f"- **applicability:** {case.pico.applicability}")
        lines.append("")
        lines.append("| model | judgment | reason |")
        lines.append("|---|---|---|")
        for model, judged in judgments.items():
            j = judged[idx]
            lines.append(f"| {model} | {j.support.value} | {j.reason} |")
        lines.append("")
    return "\n".join(lines)
