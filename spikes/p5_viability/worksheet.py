# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gold-label the P5 eval partition via a markdown worksheet round-trip (the
frontier proposes a label per case; you audit it), and frontier-label the train
partition. Pure render/parse/apply helpers + thin LLM-bound CLI subcommands.

    PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py emit \
        --cases spikes/p5_viability/cases/eval.json --out spikes/p5_viability/cases/eval_worksheet.md
    # ...edit eval_worksheet.md...
    PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py ingest \
        --worksheet spikes/p5_viability/cases/eval_worksheet.md --cases spikes/p5_viability/cases/eval.json
    PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py label-train \
        --cases spikes/p5_viability/cases/train.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.contracts import SupportJudgment  # noqa: E402
from evidenceseeker.primitives.types import CitationJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, load_cases, save_cases  # noqa: E402
from spikes.p5_viability.run_viability import progress, safe_verify  # noqa: E402

_ID_RE = re.compile(r"^## (.+)$")
_GOLD_RE = re.compile(r"^gold:\s*(\S+)\s*$")


def render_worksheet(cases: list[P5Case], judgments: list[CitationJudgment]) -> str:
    lines = [
        "# P5 eval gold worksheet",
        "# Edit the `gold:` line for any case you disagree with; leave the rest.",
        "",
    ]
    for case, j in zip(cases, judgments, strict=True):
        lines += [
            f"## {case.id}",
            f"- intended_class: {case.intended_class.value}",
            f"- failure_mode: {case.failure_mode or '-'}",
            f"- claim: {case.claim}",
            f"- passage: {case.passage}",
            f"- applicability: {case.pico.applicability}",
            f"- frontier: {j.support.value} — {j.reason}",
            f"gold: {j.support.value}",
            "note:",
            "",
        ]
    return "\n".join(lines)


def parse_worksheet(md: str) -> dict[str, SupportJudgment]:
    labels: dict[str, SupportJudgment] = {}
    current: str | None = None
    for line in md.splitlines():
        header = _ID_RE.match(line)
        if header:
            current = header.group(1).strip()
            continue
        gold = _GOLD_RE.match(line)
        if gold:
            if current is None:
                raise ValueError(f"`gold:` line before any `## <id>`: {line!r}")
            labels[current] = SupportJudgment(gold.group(1))  # raises ValueError on a bad label
            current = None
    return labels


def apply_labels(cases: list[P5Case], labels: dict[str, SupportJudgment]) -> list[P5Case]:
    by_id = {c.id: c for c in cases}
    unknown = set(labels) - set(by_id)
    if unknown:
        raise ValueError(f"worksheet has unknown ids: {sorted(unknown)}")
    missing = set(by_id) - set(labels)
    if missing:
        raise ValueError(f"no gold label for: {sorted(missing)}")
    return [c.model_copy(update={"gold_label": labels[c.id]}) for c in cases]


def _frontier_judge(cases: list[P5Case], cfg: P5SpikeConfig) -> list[CitationJudgment]:
    from bmlib.llm import LLMClient

    from evidenceseeker.primitives.p5_verify_citation import make_p5_agent

    agent = make_p5_agent(LLMClient(), cfg.reference_model, temperature=cfg.temperature)
    return [safe_verify(agent, c) for c in progress(cases, desc=cfg.reference_model)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--cases", type=Path, required=True)
    emit.add_argument("--out", type=Path, required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--worksheet", type=Path, required=True)
    ingest.add_argument("--cases", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = P5SpikeConfig()
    if args.cmd == "emit":
        cases = load_cases(args.cases)
        md = render_worksheet(cases, _frontier_judge(cases, cfg))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"worksheet -> {args.out} ({len(cases)} cases)", file=sys.stderr)
    elif args.cmd == "ingest":
        cases = load_cases(args.cases)
        labels = parse_worksheet(args.worksheet.read_text())
        save_cases(apply_labels(cases, labels), args.cases)
        print(f"gold labels written -> {args.cases}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
