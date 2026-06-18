# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure scoring functions for the P5 viability spike.

All functions take equal-length lists of ``SupportJudgment`` and use
``zip(..., strict=True)`` so a length mismatch is a loud error, not a silent
truncation.
"""

from __future__ import annotations

from collections import Counter

from evidenceseeker.contracts import SupportJudgment

_SUPPORTS = SupportJudgment.SUPPORTS


def false_support_rate(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    """Of cases whose reference is NOT 'supports', the fraction the model
    called 'supports'. The trust headline."""
    negatives = [m for m, r in zip(model, reference, strict=True) if r is not _SUPPORTS]
    if not negatives:
        return 0.0
    return sum(1 for m in negatives if m is _SUPPORTS) / len(negatives)


def missed_support_rate(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    """Of cases whose reference IS 'supports', the fraction the model missed."""
    positives = [m for m, r in zip(model, reference, strict=True) if r is _SUPPORTS]
    if not positives:
        return 0.0
    return sum(1 for m in positives if m is not _SUPPORTS) / len(positives)


def accuracy(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    pairs = list(zip(model, reference, strict=True))
    if not pairs:
        return 0.0
    return sum(1 for m, r in pairs if m is r) / len(pairs)


def cohen_kappa(a: list[SupportJudgment], b: list[SupportJudgment]) -> float:
    pairs = list(zip(a, b, strict=True))
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in pairs if x is y) / n
    count_a = Counter(a)
    count_b = Counter(b)
    labels = set(a) | set(b)
    expected = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> dict[tuple[SupportJudgment, SupportJudgment], int]:
    return dict(Counter(zip(model, reference, strict=True)))
