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
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.contracts import SupportJudgment  # noqa: E402
from evidenceseeker.primitives.types import CitationJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, load_cases  # noqa: E402
from spikes.p5_viability.scoring import (  # noqa: E402
    JUDGE_ERROR_PREFIX,
    ModelScore,
    attach_timings,
    render_review,
    score_run,
)

if TYPE_CHECKING:
    from evidenceseeker.primitives.p5_verify_citation import P5VerifyCitation

_T = TypeVar("_T")


def progress(seq: Sequence[_T], *, desc: str) -> Iterator[_T]:
    """Yield items with progress feedback. Uses tqdm if installed, else prints a
    lightweight ``desc: i/total`` counter to stderr — so there's always feedback
    without forcing a tqdm dependency."""
    total = len(seq)
    try:
        from tqdm import tqdm

        yield from tqdm(seq, desc=desc, total=total, unit="case", file=sys.stderr, leave=False)
        return
    except ImportError:
        pass
    for i, item in enumerate(seq, 1):
        print(f"\r  {desc}: {i}/{total}", end="", file=sys.stderr, flush=True)
        yield item
    print("", file=sys.stderr)  # terminate the \r line


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


def _dump_judgments(out_dir: Path, judgments: dict[str, list[CitationJudgment]]) -> None:
    """Persist raw judgments so a crash mid-sweep doesn't discard completed work."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {m: [j.model_dump(mode="json") for j in js] for m, js in judgments.items()}
    (out_dir / "judgments.json").write_text(json.dumps(payload, indent=2))


def safe_verify(agent: "P5VerifyCitation", case: P5Case) -> CitationJudgment:
    """Judge one case, never raising. A model that errors or emits malformed
    JSON (the likeliest failure for the small models under test) is recorded as
    a conservative ``does_not`` whose reason carries the error, so one bad case
    cannot abort a paid sweep and the failure is visible in the review."""
    try:
        return agent.verify(pico=case.pico, claim=case.claim, passage=case.passage)
    except Exception as exc:  # noqa: BLE001 — spike harness must survive any model failure
        print(f"  ! {case.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return CitationJudgment(
            support=SupportJudgment.DOES_NOT,
            reason=f"{JUDGE_ERROR_PREFIX} {type(exc).__name__}: {exc}",
        )


def live_judgments(
    cases: list[P5Case],
    models: list[str],
    cfg: P5SpikeConfig,
    out_dir: Path | None = None,
) -> tuple[dict[str, list[CitationJudgment]], dict[str, float]]:
    """Judge every case with each model, returning judgments and per-model
    end-to-end wall-clock seconds."""
    from bmlib.llm import LLMClient

    from evidenceseeker.primitives.p5_verify_citation import make_p5_agent

    llm = LLMClient()
    out: dict[str, list[CitationJudgment]] = {}
    timings: dict[str, float] = {}
    for model in models:
        agent = make_p5_agent(llm, model, temperature=cfg.temperature)
        start = time.perf_counter()
        out[model] = [safe_verify(agent, c) for c in progress(cases, desc=model)]
        timings[model] = time.perf_counter() - start
        print(
            f"judged {len(cases)} cases with {model} in {timings[model]:.1f}s",
            file=sys.stderr,
        )
        if out_dir is not None:
            _dump_judgments(out_dir, out)  # persist progressively, per model
    return out, timings


def verdict_lines(scores: list[ModelScore], cfg: P5SpikeConfig) -> list[str]:
    lines: list[str] = []
    for s in scores:
        if s.model not in cfg.judge_models:
            continue
        # A model that mostly errored can't earn a GO — its low false_support is
        # an artifact of synthesized does_not judgments, not real performance.
        if s.error_count and s.error_count * 2 >= s.n:
            flag = "ERRORED"
        elif s.false_support_rate <= cfg.max_false_support_rate:
            flag = "GO"
        else:
            flag = "NO-GO"
        err_note = f", {s.error_count}/{s.n} errored" if s.error_count else ""
        lines.append(
            f"[{flag}] {s.model}: false_support={s.false_support_rate:.2f} "
            f"(threshold {cfg.max_false_support_rate:.2f}, informational{err_note})"
        )
    return lines


def _fmt_secs(seconds: float | None) -> str:
    return f"{seconds:.1f}" if seconds is not None else "-"


def _print_text(scores: list[ModelScore], cfg: P5SpikeConfig) -> None:
    header = (
        f"\n{'model':<34} {'n':>3} {'err':>3} {'secs':>7} {'acc':>5} "
        f"{'false_sup':>10} {'missed':>7} {'kappa':>6}"
    )
    print(header)
    print("-" * 92)
    for s in scores:
        # The reference is scored against itself (absent human gold labels), so
        # its row is trivially perfect — mark it so it isn't read as a result.
        marker = "  <- reference (scored vs self)" if s.is_reference else ""
        print(
            f"{s.model:<34} {s.n:>3} {s.error_count:>3} {_fmt_secs(s.seconds):>7} "
            f"{s.accuracy:>5.2f} {s.false_support_rate:>10.2f} "
            f"{s.missed_support_rate:>7.2f} {s.kappa_vs_frontier:>6.2f}{marker}"
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

    if args.dry_run:
        judgments = dry_run_judgments(cases, models)
        timings: dict[str, float] = {}
    else:
        judgments, timings = live_judgments(cases, models, cfg, out_dir=args.out)
    scores = attach_timings(score_run(cases, judgments, cfg.reference_model), timings)

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
