# P5 Small-Model Viability Spike — Design

**Date:** 2026-06-18
**Status:** approved design, pre-implementation
**Topic:** Spike-test whether a stock small/local model can execute the P5
citation-support primitive well enough to justify a fine-tuning investment.

## Purpose

The `cairn-advisor` design (`0002-cairn-advisor-reasoning-primitives.md`) bets
that clinical evidence appraisal is **procedural** and therefore executable at
20–30B / local scale, with an explicit **PICO frame** compensating for what a
small model cannot supply alone. Before any fine-tuning budget is spent, this
spike cheaply de-risks that bet on the single highest-signal primitive:

> **P5 — Verify Citation Support (PICO-conditioned):**
> `{claim, candidate_passage, PicoFrame}` → `supports | partial | does_not | contradicts`.

The trust headline for the whole system is P5's **false-support rate**: how
often a model calls a near-miss passage (right drug, wrong population/dose)
`supports`. This spike measures that rate across several local model sizes
against a frontier reference, and finds the size — if any — where viability
kicks in.

### Hypotheses under test
1. Procedural reasoning at modest scale: a stock local model executes P5
   acceptably without fine-tuning.
2. PICO scaffolding: an explicit frame threaded into the judgment compensates
   for small-model limitations.

### Out of scope
- All other primitives (P1–P4, P6–P8).
- Fine-tuning, adapters, retrieval, the cold/hot path.
- Semantic correctness of any downstream report. This spike measures one
  classification task only.

## Staged strategy (why the design has a seam)

Building a clinician-verified gold set is expensive. Prior experience says
frontier models are already very good at P5. So the spike runs in two stages
sharing one code path:

- **Stage 1 — frontier-as-reference screen (this spike).** The frontier model's
  blind judgment is the provisional yardstick. If the best local model is
  hopeless against the frontier here, stop — the gold set is never built.
- **Stage 2 — gold confirm (only if Stage 1 looks promising).** Replace
  generated cases with real-literature passages; fill human-verified
  `gold_label`s. The runner is unchanged.

The seam that makes both stages one code path: **a case's reference label is
`gold_label` if present, else the frontier judge's blind output.**

## Approach (chosen: A)

Build P5 as a **real, reusable primitive** in `src/evidenceseeker/`; the case
generator and comparison runner are **disposable spike artifacts** in a new
top-level `spikes/` tree. If viability fails, the P5 primitive still survives as
README next-step #2 ("Implement P5… behind its own harness"). Rejected: a single
throwaway script (P5 logic not reusable), and a notebook (poor fit for the
repo's CLI/harness convention).

## Layout

```
src/evidenceseeker/
  primitives/
    __init__.py
    p5_verify_citation.py      # P5VerifyCitation(BaseAgent) + CitationJudgment — REUSABLE, model-agnostic
  prompts/
    p5_verify_citation.jinja   # PICO-conditioned support-judgment prompt
spikes/p5_viability/           # NOT under tests/ — calls live LLMs, non-deterministic, costs money
  cases.py                     # P5Case schema + load/save helpers
  metrics.py                   # confusion matrix, false-support rate, Cohen's κ (pure functions)
  generate_cases.py            # frontier authors {pico, claim, passage, intended_class}
  run_viability.py             # run all judges over cases → score → comparison + review artifact
  cases/                       # generated case sets (versioned JSON, hand-editable)
  out/                         # comparison tables + per-case review docs (gitignored)
tests/
  test_p5_verify_citation.py   # unit test with a FAKE LLMClient (deterministic, no network)
  test_p5_metrics.py           # metrics over synthetic judgments
  test_p5_case_schema.py       # P5Case round-trip
```

`spikes/` is deliberately outside `pytest`'s `testpaths=["tests"]` and outside
mypy's `files=["src","tests"]`: it is live, non-deterministic, and nimble. The
reusable primitive lives in `src/` and IS covered by strict mypy + unit tests.

## Components

### `P5VerifyCitation` (reusable — the real primitive)
A `bmlib.agents.BaseAgent` subclass. Model-agnostic: the `provider:model`
string is passed at construction, so "run across N models" is just N instances.

```python
class CitationJudgment(_Frozen):
    support: SupportJudgment
    reason: str
    pico_match_notes: str | None = None

class P5VerifyCitation(BaseAgent):
    def verify(self, *, pico: PicoFrame, claim: str, passage: str) -> CitationJudgment:
        prompt = self.render_template("p5_verify_citation.jinja",
                                      pico=pico, claim=claim, passage=passage)
        data = self.chat_json([self.system_msg(_SYSTEM), self.user_msg(prompt)])
        return CitationJudgment(
            support=SupportJudgment(data["support"]),
            reason=data["reason"],
            pico_match_notes=data.get("pico_match_notes"),
        )
```

Prompt requirements (the PICO-conditioning that is the hypothesis): the model
must judge support **conditioned on the PICO frame** — a passage about a
different population, endpoint, or dose is `partial` or `does_not`, never
`supports`. Output is strict JSON: `{support, reason, pico_match_notes}`.

### `P5Case` (spike-local schema)
```python
class P5Case(_Frozen):
    id: str
    pico: PicoFrame
    claim: str
    passage: str
    intended_class: SupportJudgment        # what the generator was asked to construct
    gold_label: SupportJudgment | None = None   # filled in Stage 2 after human verification
    notes: str | None = None               # generator rationale / near-miss type
```
Reuses `PicoFrame` and `SupportJudgment` from `contracts.py`. Kept in the spike
tree (not promoted to `contracts.py`, which is the Kastellan wire surface) until
Stage 2 proves it stable.

### `generate_cases.py` (disposable)
For each `SupportJudgment` class, asks `generator_model` (frontier) to author
`cases_per_class` realistic clinical `{pico, claim, passage}` triples, with
explicit emphasis on **near-misses** for the non-`supports` classes (right
drug/wrong population, right outcome/wrong dose). Writes `cases/<name>.json`.
The human proofreads/edits before judging. The `intended_class` is recorded but
**never shown to any judge**.

### `metrics.py` (pure functions, unit-tested)
- `confusion(model_judgments, reference_labels) -> matrix`
- `false_support_rate(...)` — of cases whose reference is **not** `supports`,
  the fraction the model labeled `supports`. **The headline.**
- `missed_support_rate(...)` — secondary (supports → not).
- `cohen_kappa(model_judgments, frontier_judgments)` — agreement vs frontier.

### `run_viability.py` (disposable)
1. Load a case set (`cases/*.json` → `list[P5Case]`).
2. For each model in `config.judge_models` + `config.reference_model`,
   instantiate `P5VerifyCitation` and judge every case blind. Track tokens via
   bmlib `TokenTracker`.
3. Reference label per case = `gold_label` if present, else the frontier judge's
   output; `intended_class` reported as a secondary check.
4. Compute per-model metrics (above).
5. Emit (a) a comparison table to stdout (`--format text|json`) and (b) a
   per-case **review artifact** (markdown in `out/`) listing
   pico/claim/passage/intended + every model's judgment **and reason**, for
   human proofreading.
6. Print a verdict against `max_false_support_rate` — **informational, not a
   hard CLI failure** (this is exploratory).

CLI mirrors `run_noninferiority_eval.py`: argparse, `--cases`, `--format`,
`--out`.

## Config

New frozen `P5SpikeConfig` in `src/evidenceseeker/config.py` (separate from
`EvalConfig`):

```python
class P5SpikeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    judge_models: list[str] = [           # local models under test — EDIT to installed Ollama tags
        "ollama:medgemma4B_it_q8",
        "ollama:medgemma-27b",
        "ollama:qwen2.5:7b",
    ]
    reference_model: str = "anthropic:claude-sonnet-4-20250514"
    generator_model: str = "anthropic:claude-sonnet-4-20250514"
    cases_per_class: int = 5
    temperature: float = 0.0              # judgment stability
    max_false_support_rate: float = 0.10  # informational verdict, not a hard gate
```

## Dependencies

Add to `pyproject.toml`: `bmlib[anthropic,ollama]`. `bmlib` is a local sibling
repo (`/Users/hherb/src/bmlib`), so it is wired as a path/editable source via
`[tool.uv.sources]` rather than a PyPI version.

## Testing

| Test | What | How |
|------|------|-----|
| `test_p5_verify_citation.py` | agent builds prompt + parses JSON into `CitationJudgment` | inject a **fake** `LLMClient` returning a canned `LLMResponse`; no network |
| `test_p5_metrics.py` | false-support / missed-support / κ correct | synthetic judgment+reference lists |
| `test_p5_case_schema.py` | `P5Case` JSON round-trip, optional `gold_label` | pydantic `model_dump`/reload |

The live `run_viability.py` and `generate_cases.py` are **not** pytest tests.

## Success criteria for the spike (not a CI gate)

The spike answers a go/no-go question, reported as a table + verdict:
- **GO signal:** at least one local model's false-support rate is within a small
  margin of the frontier reference's (target `≤ max_false_support_rate`), with
  reasonable overall agreement (κ). → proceed to Stage 2 gold construction.
- **NO-GO signal:** every local model has a materially higher false-support rate
  than the frontier across sizes. → do not build the gold set; reconsider base
  model or whether P5 needs fine-tuning before it is viable at all.

Either way the verdict is human-confirmed by proofreading the review artifact.

## Ramp path (Stage 2, future)

Replace generator cases with real-literature passages (via `bmlib.fulltext`) and
fill `gold_label` with human-verified labels. `P5VerifyCitation`, `metrics.py`,
and `run_viability.py` are unchanged — only the case source and the reference
label change.
