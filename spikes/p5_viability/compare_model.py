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

With ``--out`` the results ACCUMULATE: each run merges its candidates (keyed by
model) into ``out/compare_judgments.json`` and the printed table + review cover
every candidate scored so far, so you can add models one at a time. Re-running a
model refreshes only its row. Per-model e2e seconds are persisted alongside in
``out/compare_timings.json``, so the leaderboard's ``secs`` survives the run and
shows for every accumulated model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
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
    progress,
    safe_verify,
)
from spikes.p5_viability.scoring import (  # noqa: E402
    attach_timings,
    is_all_errored,
    render_review,
    score_run,
)

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


_COMPARE_FILE = "compare_judgments.json"
_TIMINGS_FILE = "compare_timings.json"


def _persist_timings(out_dir: Path, timings: dict[str, float]) -> None:
    """Merge per-model e2e seconds into a sidecar so the leaderboard's ``secs``
    survives the run and shows for every accumulated model, not just this one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _TIMINGS_FILE
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(timings)
    path.write_text(json.dumps(existing, indent=2))


def _load_timings(out_dir: Path) -> dict[str, float]:
    path = out_dir / _TIMINGS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _persist(out_dir: Path, judgments: dict[str, list[CitationJudgment]]) -> None:
    """Merge candidate judgments into a SEPARATE file (never the reference run's
    judgments.json), keyed by model so repeated runs accumulate a leaderboard
    rather than overwrite it. Re-running a model refreshes just its entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _COMPARE_FILE
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(
        {m: [j.model_dump(mode="json") for j in js] for m, js in judgments.items()}
    )
    path.write_text(json.dumps(existing, indent=2))


def _load_accumulated(
    out_dir: Path, n_cases: int
) -> dict[str, list[CitationJudgment]]:
    """All candidate judgments persisted so far, dropping any whose length no
    longer matches the current case set (stale from an earlier case file)."""
    path = out_dir / _COMPARE_FILE
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, list[CitationJudgment]] = {}
    for model, js in raw.items():
        if len(js) != n_cases:
            print(
                f"skipping stale {model!r}: {len(js)} judgments != {n_cases} cases",
                file=sys.stderr,
            )
            continue
        out[model] = [CitationJudgment.model_validate(j) for j in js]
    return out


def judge_with(
    models: list[str],
    cases: list[P5Case],
    cfg: P5SpikeConfig,
    out_dir: Path | None = None,
) -> tuple[dict[str, list[CitationJudgment]], dict[str, float]]:
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
        if is_all_errored(out[model]):
            print(
                f"WARNING: {model!r} errored on all {len(cases)} cases — NOT saved "
                f"to {_COMPARE_FILE} (check the model name).",
                file=sys.stderr,
            )
        else:
            print(
                f"judged {len(cases)} cases with {model} in {timings[model]:.1f}s",
                file=sys.stderr,
            )
        if out_dir is not None:
            # Persist progressively (a crash keeps completed models), but keep
            # all-errored runs out of the accumulated leaderboard.
            good = {m: js for m, js in out.items() if not is_all_errored(js)}
            if good:
                _persist(out_dir, good)
                _persist_timings(out_dir, {m: timings[m] for m in good})
    return out, timings


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

    if args.dry_run:
        candidates = dry_run_judgments(cases, args.models)
        timings: dict[str, float] = {}
    else:
        candidates, timings = judge_with(args.models, cases, base, out_dir=args.out)
    # With --out, judge_with merged this run into the accumulated leaderboard;
    # score and review the whole accumulated set so earlier candidates persist,
    # and use the persisted timings so every model shows its last-measured secs.
    if args.out is not None and not args.dry_run:
        candidates = {**_load_accumulated(args.out, len(cases)), **candidates}
        timings = {**_load_timings(args.out), **timings}
    judgments = {base.reference_model: reference, **candidates}
    scores = attach_timings(score_run(cases, judgments, base.reference_model), timings)

    # A config whose judge_models are every scored candidate, so the verdict
    # flags them all (the table marks the reference row regardless).
    verdict_cfg = P5SpikeConfig(
        judge_models=list(candidates), reference_model=base.reference_model
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
