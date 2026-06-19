# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split a P5 case pool into a human-verified eval partition and a frontier-
labeled train partition. Pure + deterministic: same pool -> same split.

    PYTHONPATH=src uv run python spikes/p5_viability/split_dataset.py \
        --pool spikes/p5_viability/cases/pool.json \
        --eval spikes/p5_viability/cases/eval.json \
        --train spikes/p5_viability/cases/train.json
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from spikes.p5_viability.cases import P5Case, load_cases, save_cases  # noqa: E402


def _content_key(case: P5Case) -> str:
    return re.sub(r"\s+", " ", f"{case.claim}\n{case.passage}".lower()).strip()


def dedupe(cases: list[P5Case]) -> list[P5Case]:
    """Drop cases with an identical normalized (claim, passage); keep the first.
    Exact-normalized only — no semantic near-duplicate detection."""
    seen: set[str] = set()
    out: list[P5Case] = []
    for c in cases:
        key = _content_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _check_disjoint(eval_: list[P5Case], train: list[P5Case]) -> None:
    """Raise if any id or normalized content leaks across the two partitions.
    Raises (not ``assert``) so the leakage guard survives ``python -O``."""
    id_overlap = {c.id for c in eval_} & {c.id for c in train}
    if id_overlap:
        raise ValueError(f"id leakage across partitions: {sorted(id_overlap)}")
    key_overlap = {_content_key(c) for c in eval_} & {_content_key(c) for c in train}
    if key_overlap:
        raise ValueError("content leakage across partitions")


def split(cases: list[P5Case], eval_frac: float) -> tuple[list[P5Case], list[P5Case]]:
    """Stratified, deterministic split into (eval, train). Within each
    (intended_class, failure_mode) stratum, the first ``int(n*eval_frac + 0.5)``
    cases (sorted by id, round half up) go to eval, the rest to train."""
    strata: dict[tuple[str, str], list[P5Case]] = defaultdict(list)
    for c in cases:
        strata[(c.intended_class.value, c.failure_mode or "")].append(c)
    eval_: list[P5Case] = []
    train: list[P5Case] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda c: c.id)
        n_eval = int(len(group) * eval_frac + 0.5)  # round half up (deterministic)
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])
    _check_disjoint(eval_, train)
    return eval_, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--eval", dest="eval_path", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval-frac", type=float, default=None, help="overrides config")
    args = ap.parse_args(argv)

    frac = args.eval_frac if args.eval_frac is not None else P5SpikeConfig().eval_frac
    pool = dedupe(load_cases(args.pool))
    eval_, train = split(pool, frac)
    save_cases(eval_, args.eval_path)
    save_cases(train, args.train)
    print(f"pool {len(pool)} -> eval {len(eval_)}, train {len(train)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
