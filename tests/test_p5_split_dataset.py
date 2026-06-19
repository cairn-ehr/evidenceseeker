# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment as S
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.split_dataset import dedupe, split
from tests._helpers import make_pico


def _case(cid: str, cls: S, *, claim: str = "c", passage: str = "p", mode: str | None = None) -> P5Case:
    return P5Case(
        id=cid, pico=make_pico(), claim=claim, passage=passage,
        intended_class=cls, failure_mode=mode,
    )


def test_dedupe_drops_identical_claim_passage() -> None:
    cases = [_case("a", S.SUPPORTS), _case("b", S.SUPPORTS), _case("c", S.SUPPORTS, claim="other")]
    out = dedupe(cases)
    assert [c.id for c in out] == ["a", "c"]  # "b" is a dup of "a"


def test_split_is_deterministic() -> None:
    cases = [_case(f"s{i}", S.SUPPORTS, claim=f"c{i}") for i in range(10)]
    first = split(cases, 0.3)
    second = split(cases, 0.3)
    assert [c.id for c in first[0]] == [c.id for c in second[0]]
    assert [c.id for c in first[1]] == [c.id for c in second[1]]


def test_split_is_disjoint_and_covers_train() -> None:
    cases = (
        [_case(f"sup{i}", S.SUPPORTS, claim=f"a{i}") for i in range(4)]
        + [_case(f"sig{i}", S.PARTIAL, claim=f"b{i}", mode="significance_overstatement") for i in range(4)]
        + [_case(f"app{i}", S.PARTIAL, claim=f"d{i}", mode="applicability_mismatch") for i in range(4)]
    )
    eval_, train = split(cases, 0.3)
    eval_ids = {c.id for c in eval_}
    train_ids = {c.id for c in train}
    assert not (eval_ids & train_ids)                       # disjoint
    assert eval_ids | train_ids == {c.id for c in cases}    # nothing dropped
    train_strata = {(c.intended_class, c.failure_mode) for c in train}
    assert (S.SUPPORTS, None) in train_strata               # no stratum starved from train
    assert (S.PARTIAL, "significance_overstatement") in train_strata
    assert (S.PARTIAL, "applicability_mismatch") in train_strata
