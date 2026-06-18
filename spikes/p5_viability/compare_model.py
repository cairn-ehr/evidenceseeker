# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score one or more *additional* models on the existing P5 cases against the
Sonnet reference judgments we already saved — no frontier re-run, no full gamut.

    # test extra candidates against the saved reference:
    PYTHONPATH=src uv run python spikes/p5_viability/compare_model.py \
        ollama:gpt-oss:20b ollama:phi4:14b --out spikes/p5_viability/out

    # prove the wiring without any model:
    PYTHONPATH=src uv run python spikes/p5_viability/compare_model.py \
        ollama:gpt-oss:20b --dry-run

Reads cases from ``cases/generated.json`` and the reference judgments from
``out/judgments.json`` (written by ``run_viability.py``). A case's human
``gold_label`` still overrides the reference, exactly as in the full runner, so
relabelling a case in the JSON immediately changes what the candidates are
scored against.
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
from spikes.p5_viability.run_viability import (  # noqa: E402
    _print_text,
    dry_run_judgments,
    safe_verify,
)
from spikes.p5_viability.scoring import render_review, score_run  # noqa: E402

_DEFAULT_CASES = _ROOT / "spikes/p5_viability/cases/generated.json"
_DEFAULT_REF = _ROOT / "spikes/p5_viability/out/judgments.json"


def load_reference_judgments(
    path: Path, reference_model: str, n_cases: int
) -> list[CitationJudgment]:
    """Pull the saved reference model's judgments out of a judgments.json and
    make sure they line up positionally with the cases we're about to score."""
    if not path.exists():
        raise SystemExit(
            f"reference judgments {path} not found — run run_viability.py with "
            f"--out first to produce it."
        )
    raw = json.loads(path.read_text())
    if reference_model not in raw:
        raise SystemExit(
            f"{path} has no entry for reference model {reference_model!r}; "
            f"found {list(raw)}."
        )
    judged = [CitationJudgment.model_validate(j) for j in raw[reference_model]]
    if len(judged) != n_cases:
        raise SystemExit(
            f"reference has {len(judged)} judgments but the cases file has "
            f"{n_cases} — they must come from the same case set."
        )
    return judged


def _persist(out_dir: Path, judgments: dict[str, list[CitationJudgment]]) -> None:
    """Write candidate judgments to a SEPARATE file so we never clobber the
    reference run's judgments.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {m: [j.model_dump(mode="json") for j in js] for m, js in judgments.items()}
    (out_dir / "compare_judgments.json").write_text(json.dumps(payload, indent=2))


def judge_with(
    models: list[str],
    cases: list[P5Case],
    cfg: P5SpikeConfig,
    out_dir: Path | None = None,
) -> dict[str, list[CitationJudgment]]:
    from bmlib.llm import LLMClient

    from evidenceseeker.primitives.p5_verify_citation import make_p5_agent

    llm = LLMClient()
    out: dict[str, list[CitationJudgment]] = {}
    for model in models:
        agent = make_p5_agent(llm, model, temperature=cfg.temperature)
        out[model] = [safe_verify(agent, c) for c in cases]
        print(f"judged {len(cases)} cases with {model}", file=sys.stderr)
        if out_dir is not None:
            _persist(out_dir, out)  # progressive, so a crash keeps completed models
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="provider:model strings to test")
    ap.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    ap.add_argument("--reference-judgments", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--out", type=Path, default=None, help="dir for the review artifact")
    ap.add_argument("--dry-run", action="store_true", help="echo intended class, no LLM")
    args = ap.parse_args(argv)

    base = P5SpikeConfig()
    if base.reference_model in args.models:
        raise SystemExit(
            f"{base.reference_model!r} is the reference; don't pass it as a candidate"
        )

    cases = load_cases(args.cases)
    reference = load_reference_judgments(
        args.reference_judgments, base.reference_model, len(cases)
    )

    candidates = (
        dry_run_judgments(cases, args.models)
        if args.dry_run
        else judge_with(args.models, cases, base, out_dir=args.out)
    )
    judgments = {base.reference_model: reference, **candidates}
    scores = score_run(cases, judgments, base.reference_model)

    # A config whose judge_models are the candidates, so the verdict flags them
    # (the table marks the reference row regardless).
    verdict_cfg = P5SpikeConfig(
        judge_models=list(args.models), reference_model=base.reference_model
    )

    if args.format == "json":
        print(json.dumps({"scores": [dataclasses.asdict(s) for s in scores]}, indent=2))
    else:
        _print_text(scores, verdict_cfg)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        review = render_review(cases, judgments, base.reference_model)
        (args.out / "compare_review.md").write_text(review)
        print(f"\nreview -> {args.out / 'compare_review.md'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
