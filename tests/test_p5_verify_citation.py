# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
from typing import Any

from bmlib.llm import LLMResponse

from evidenceseeker.contracts import SupportJudgment
from evidenceseeker.primitives.p5_verify_citation import make_p5_agent
from tests._helpers import make_pico


class FakeLLM:
    """Records the last messages and returns a canned JSON response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_messages: list[Any] | None = None

    def chat(self, *, messages: list[Any], model: str, **kwargs: Any) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(content=self.content, model=model)


def test_verify_parses_judgment_and_conditions_on_pico() -> None:
    canned = json.dumps(
        {
            "support": "partial",
            "reason": "Right drug, wrong population.",
            "pico_match_notes": "intervention matches; population mismatch",
        }
    )
    llm = FakeLLM(canned)
    agent = make_p5_agent(llm, "ollama:test", temperature=0.0)

    result = agent.verify(
        pico=make_pico(intervention="DrugA", dose="10mg"),
        claim="DrugA is non-inferior to DrugB.",
        passage="A trial in children found DrugA non-inferior to DrugB.",
    )

    assert result.support is SupportJudgment.PARTIAL
    assert result.reason == "Right drug, wrong population."
    # The rendered prompt must actually carry the PICO frame.
    assert llm.last_messages is not None
    rendered = llm.last_messages[1].content
    assert "DrugA" in rendered and "10mg" in rendered
    assert "community outpatients aged 65-74" in rendered
