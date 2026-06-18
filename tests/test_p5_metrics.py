# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment as S
from spikes.p5_viability.metrics import (
    accuracy,
    cohen_kappa,
    confusion,
    false_support_rate,
    missed_support_rate,
)


def test_false_support_rate_counts_only_non_support_references() -> None:
    # references: 2 non-support (DOES_NOT, PARTIAL), 1 support.
    reference = [S.DOES_NOT, S.PARTIAL, S.SUPPORTS]
    # model calls the first non-support case "supports" -> 1/2.
    model = [S.SUPPORTS, S.PARTIAL, S.SUPPORTS]
    assert false_support_rate(model, reference) == 0.5


def test_false_support_rate_zero_when_no_negatives() -> None:
    assert false_support_rate([S.SUPPORTS], [S.SUPPORTS]) == 0.0


def test_missed_support_rate() -> None:
    reference = [S.SUPPORTS, S.SUPPORTS]
    model = [S.SUPPORTS, S.PARTIAL]
    assert missed_support_rate(model, reference) == 0.5


def test_accuracy() -> None:
    assert accuracy([S.SUPPORTS, S.PARTIAL], [S.SUPPORTS, S.DOES_NOT]) == 0.5


def test_cohen_kappa_perfect_agreement() -> None:
    a = [S.SUPPORTS, S.PARTIAL, S.DOES_NOT]
    assert cohen_kappa(a, a) == 1.0


def test_confusion_counts_pairs() -> None:
    c = confusion([S.SUPPORTS, S.SUPPORTS], [S.SUPPORTS, S.PARTIAL])
    assert c[(S.SUPPORTS, S.SUPPORTS)] == 1
    assert c[(S.SUPPORTS, S.PARTIAL)] == 1
