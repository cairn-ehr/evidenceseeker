# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 — Verify Citation Support (PICO-conditioned).

{claim, passage, PicoFrame} -> supports | partial | does_not | contradicts.
Model-agnostic: the ``provider:model`` string is passed at construction, so a
comparison across several models is just several instances.
"""

from __future__ import annotations

from pathlib import Path

from bmlib.agents import BaseAgent
from bmlib.llm import LLMClient
from bmlib.templates import TemplateEngine

from evidenceseeker.contracts import PicoFrame, SupportJudgment
from evidenceseeker.primitives.types import CitationJudgment

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_TEMPLATE = "p5_verify_citation.jinja"
_SYSTEM = (
    "You are a clinical evidence appraiser. You judge citation support strictly, "
    "conditioned on the PICO frame, and you return only the requested JSON."
)


class P5VerifyCitation(BaseAgent):  # type: ignore[misc]
    def verify(self, *, pico: PicoFrame, claim: str, passage: str) -> CitationJudgment:
        prompt = self.render_template(_TEMPLATE, pico=pico, claim=claim, passage=passage)
        data = self.chat_json([self.system_msg(_SYSTEM), self.user_msg(prompt)])
        notes = data.get("pico_match_notes")
        return CitationJudgment(
            support=SupportJudgment(data["support"]),
            reason=str(data["reason"]),
            pico_match_notes=str(notes) if notes is not None else None,
        )


def make_p5_agent(
    llm: LLMClient, model: str, *, temperature: float = 0.0
) -> P5VerifyCitation:
    engine = TemplateEngine(default_dir=_PROMPT_DIR)
    return P5VerifyCitation(
        llm=llm, model=model, template_engine=engine, temperature=temperature
    )
