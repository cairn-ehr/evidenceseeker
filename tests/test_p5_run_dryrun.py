# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.config import P5SpikeConfig
from evidenceseeker.contracts import SupportJudgment as S
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.run_viability import (
    dry_run_judgments,
    safe_verify,
    verdict_lines,
)
from spikes.p5_viability.scoring import score_run
from tests._helpers import make_pico


def _case(cid: str, intended: S) -> P5Case:
    return P5Case(
        id=cid, pico=make_pico(), claim="c", passage="p", intended_class=intended
    )


def test_dry_run_echoes_intended_class() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    judged = dry_run_judgments(cases, ["frontier", "local"])
    assert [j.support for j in judged["local"]] == [S.DOES_NOT, S.SUPPORTS]


def test_verdict_lines_flag_each_local_model() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    judged = dry_run_judgments(cases, ["frontier", "local"])
    scores = score_run(cases, judged, reference_model="frontier")
    cfg = P5SpikeConfig(judge_models=["local"], reference_model="frontier")
    lines = verdict_lines(scores, cfg)
    # dry-run echoes intended == reference, so false-support is 0 -> GO.
    assert any("local" in line and "GO" in line for line in lines)


class _RaisingAgent:
    """Stands in for a P5 agent that fails (unreachable model, bad JSON, ...)."""

    def verify(self, *, pico: object, claim: str, passage: str) -> object:
        raise RuntimeError("model unreachable")


def test_safe_verify_records_error_as_conservative_does_not() -> None:
    case = _case("a", S.SUPPORTS)
    judgment = safe_verify(_RaisingAgent(), case)  # type: ignore[arg-type]
    # A failure must never count as a false support, and the error must be visible.
    assert judgment.support is S.DOES_NOT
    assert "JUDGE-ERROR" in judgment.reason
    assert "model unreachable" in judgment.reason


def test_mostly_errored_model_cannot_earn_a_go() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    errored = safe_verify(_RaisingAgent(), cases[0])  # type: ignore[arg-type]
    judged = {
        "frontier": dry_run_judgments(cases, ["frontier"])["frontier"],
        "local": [errored, errored],  # both cases failed
    }
    scores = score_run(cases, judged, reference_model="frontier")
    cfg = P5SpikeConfig(judge_models=["local"], reference_model="frontier")
    line = next(line for line in verdict_lines(scores, cfg) if "local" in line)
    assert "ERRORED" in line and "2/2 errored" in line
