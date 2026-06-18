# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.config import P5SpikeConfig
from evidenceseeker.contracts import SupportJudgment as S
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.run_viability import dry_run_judgments, verdict_lines
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
