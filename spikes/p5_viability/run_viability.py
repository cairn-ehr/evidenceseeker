# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 viability runner: judge every case with each local model + the frontier
reference, score, and emit a comparison table plus a proofreadable review.

    # prove the machinery without any model:
    PYTHONPATH=src uv run python spikes/p5_viability/run_viability.py \
        --cases spikes/p5_viability/cases/generated.json --dry-run

    # real screen:
    PYTHONPATH=src uv run python spikes/p5_viability/run_viability.py \
        --cases spikes/p5_viability/cases/generated.json --out spikes/p5_viability/out
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.primitives.types import CitationJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, load_cases  # noqa: E402
from spikes.p5_viability.scoring import ModelScore, render_review, score_run  # noqa: E402


def dry_run_judgments(
    cases: list[P5Case], models: list[str]
) -> dict[str, list[CitationJudgment]]:
    """Each model echoes the intended class — proves the pipeline, no LLM."""
    return {
        model: [
            CitationJudgment(support=c.intended_class, reason="dry-run echo")
            for c in cases
        ]
        for model in models
    }


def live_judgments(
    cases: list[P5Case], models: list[str], cfg: P5SpikeConfig
) -> dict[str, list[CitationJudgment]]:
    from bmlib.llm import LLMClient

    from evidenceseeker.primitives.p5_verify_citation import make_p5_agent

    llm = LLMClient()
    out: dict[str, list[CitationJudgment]] = {}
    for model in models:
        agent = make_p5_agent(llm, model, temperature=cfg.temperature)
        out[model] = []
        for c in cases:
            out[model].append(agent.verify(pico=c.pico, claim=c.claim, passage=c.passage))
        print(f"judged {len(cases)} cases with {model}", file=sys.stderr)
    return out


def verdict_lines(scores: list[ModelScore], cfg: P5SpikeConfig) -> list[str]:
    lines: list[str] = []
    for s in scores:
        if s.model not in cfg.judge_models:
            continue
        flag = "GO" if s.false_support_rate <= cfg.max_false_support_rate else "NO-GO"
        lines.append(
            f"[{flag}] {s.model}: false_support={s.false_support_rate:.2f} "
            f"(threshold {cfg.max_false_support_rate:.2f}, informational)"
        )
    return lines


def _print_text(scores: list[ModelScore], cfg: P5SpikeConfig) -> None:
    print(f"\n{'model':<34} {'n':>3} {'acc':>5} {'false_sup':>10} {'missed':>7} {'kappa':>6}")
    print("-" * 70)
    for s in scores:
        print(
            f"{s.model:<34} {s.n:>3} {s.accuracy:>5.2f} "
            f"{s.false_support_rate:>10.2f} {s.missed_support_rate:>7.2f} "
            f"{s.kappa_vs_frontier:>6.2f}"
        )
    print()
    for line in verdict_lines(scores, cfg):
        print(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--out", type=Path, default=None, help="dir for the review artifact")
    ap.add_argument("--dry-run", action="store_true", help="echo intended class, no LLM")
    args = ap.parse_args(argv)

    cfg = P5SpikeConfig()
    cases = load_cases(args.cases)
    models = [*cfg.judge_models, cfg.reference_model]

    judgments = (
        dry_run_judgments(cases, models)
        if args.dry_run
        else live_judgments(cases, models, cfg)
    )
    scores = score_run(cases, judgments, cfg.reference_model)

    if args.format == "json":
        print(json.dumps({"scores": [dataclasses.asdict(s) for s in scores]}, indent=2))
    else:
        _print_text(scores, cfg)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        review = render_review(cases, judgments, cfg.reference_model)
        (args.out / "review.md").write_text(review)
        print(f"\nreview -> {args.out / 'review.md'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
