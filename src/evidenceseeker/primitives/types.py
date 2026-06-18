# SPDX-License-Identifier: AGPL-3.0-or-later
"""bmlib-free primitive IO types.

Kept separate from ``p5_verify_citation`` (which imports bmlib) so the pure
spike modules can import ``CitationJudgment`` without pulling in an LLM stack.
"""

from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment, _Frozen


class CitationJudgment(_Frozen):
    """P5 output: how a passage relates to a claim, conditioned on PICO."""

    support: SupportJudgment
    reason: str
    pico_match_notes: str | None = None
