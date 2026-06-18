# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tunables for the acceptance harness. No magic numbers in harness code —
every threshold lives here so a gate change is a config change, not an edit."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Fraction of gold reports that must clear every applicable rubric item.
    # Structural completeness is binary and cheap, so the default gate is total:
    # every report must be shaped like a trustworthy report. Semantic
    # correctness (does the bottom line match the reference answer?) is a
    # separate human / LLM-judge pass, deliberately out of this harness.
    min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # A claim must cite at least this many supporting passages to count as
    # grounded (P5). Counter-claims are exempt.
    min_supporting_citations_per_claim: int = Field(default=1, ge=0)


class P5SpikeConfig(BaseModel):
    """Tunables for the P5 small-model viability spike."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_models: list[str] = Field(
        default_factory=lambda: [
            "ollama:medgemma1.5:4b-it-q8_0",
            "ollama:medgemma:27b-it-q8_0",
            "ollama:qwen3.6:35b-a3b-q8_0",
        ]
    )
    reference_model: str = "anthropic:claude-sonnet-4-6"
    generator_model: str = "anthropic:claude-sonnet-4-6"
    cases_per_class: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Informational verdict threshold — NOT a hard gate.
    max_false_support_rate: float = Field(default=0.10, ge=0.0, le=1.0)
