# P5 Gold-Set Scaffolding — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Builds on:** `docs/superpowers/specs/2026-06-18-p5-viability-spike-design.md` (Stage 2 of the staged plan)

## Goal

Turn the P5 viability spike's frontier-as-reference screen (Stage 1, done) into a
**human-verified eval set plus a frontier-labeled training set** for fine-tuning the
chosen base model (phi-4 14B). Two purposes, split from the start with leakage guards:

- **eval** — a small, human-verified held-out benchmark to rank base vs fine-tuned and
  measure false-support honestly.
- **train** — a larger, frontier-labeled partition to fine-tune on.

The set is sourced **synthetically now** (frontier-authored adversarial cases, expanded
to target the two known blind spots), with a clean seam to swap in **real-literature
passages later** without changing the workflow.

## Context

The `P5Case.gold_label` seam already exists: a case's reference label is its `gold_label`
if present, else the frontier judge's blind output (`scoring.reference_label`). The runner
(`run_viability.py`/`compare_model.py`) is unchanged by this work — it already prefers
`gold_label`.

Stage-1 screening surfaced two systematic, *learnable* failure modes shared by every
strong model (phi-4, qwen, gemma4, …):

1. **significance overstatement** — the claim says an outcome is "significantly" improved
   but the passage's CI crosses 1.0 / p>0.05 (e.g. `supports-2`: mortality HR 0.74, p=0.057).
2. **applicability mismatch** — the trial excluded/never enrolled the claimed population
   (e.g. `partial-2`: claim says renal impairment, trial excluded CrCl<30).

Both should be judged `partial`. The gold set must over-sample these.

## Architecture (composable stages — approach A)

New/extended one-purpose modules in `spikes/p5_viability/` (matches the existing flat
convention; live generation/labeling stays out of pytest):

| stage | module | LLM? | output |
|---|---|---|---|
| generate pool | `generate_cases.py` (extended) | yes (frontier) | `cases/pool.json` |
| split | `split_dataset.py` (new) | no (pure) | `cases/eval.json`, `cases/train.json` |
| label eval | `worksheet.py emit` / `ingest` (new) | emit: yes | `cases/eval_worksheet.md` → `eval.json` gold |
| label train | `worksheet.py label-train` (new) | yes (frontier) | `train.json` gold (= frontier) |

Everything downstream of `generate` operates on `list[P5Case]`, so the **real-literature
seam** is a future loader that emits `P5Case`s into `pool.json`; split/worksheet/label are
unchanged.

## Data model

One new field on `P5Case` (in `cases.py`), frozen + `extra="forbid"`, default keeps it
backward-compatible:

```python
failure_mode: str | None = None  # "significance_overstatement" | "applicability_mismatch" | None
```

`None` = baseline 4-class case. Drives split stratification and per-mode error analysis.
Partitions are **separate files** (no `partition` field) so they cannot be accidentally
mixed.

## Failure-mode generators (`generate_cases.py`)

A small registry alongside the existing 4-class generation. Each mode has a targeted
prompt and a fixed `intended_class`:

- `significance_overstatement` → `intended_class = partial`. PICO matches and the effect
  direction matches, but a claimed outcome's CI crosses 1.0 / p>0.05, so the
  "significantly X" clause is unsupported.
- `applicability_mismatch` → `intended_class = partial`. The trial explicitly
  excluded/never enrolled the claimed population, so the evidence doesn't generalize.

Each emitted case is tagged with its `failure_mode`. `parse_generated` sets both
`intended_class` and `failure_mode`. Count controlled by `cases_per_mode` (config).

`intended_class` records the class the generator was *asked* to build; the human/frontier
label may still differ — that's what the gold pass resolves.

## Split (`split_dataset.py`) — pure, deterministic, leakage-guarded

1. **Dedup** the pool by normalized `(claim, passage)` hash (lowercased, whitespace-
   collapsed); keep first. Exact-normalized only — *no* semantic dedup (known limitation).
2. **Stratify** by `(intended_class, failure_mode)` so the mix is proportional across
   partitions (a stratum smaller than `~1/eval_frac` may land in a single partition —
   acceptable; the larger train partition must never be starved of a class/mode).
3. **Assign** deterministically: stable sort by `id`, bucket into eval/train by `eval_frac`
   (no RNG → same input yields same split).
4. Emit disjoint `eval.json` / `train.json`. **Assert** no shared `id` and no shared
   content-hash across partitions.

## Labeling (`worksheet.py`)

Single-model frontier judging reuses `make_p5_agent` + `safe_verify`.

- **`emit`** — frontier-judges the eval cases, writes `eval_worksheet.md`: per case a
  `## <id>` section with `intended_class`, `failure_mode`, claim, passage, PICO
  applicability, the frontier's **proposed** label + reason, and editable lines
  `gold: <proposed>` (pre-filled) and `note:`.
- **`ingest`** — parses the edited worksheet by `## <id>` headers + `gold:` lines,
  validates each value is a real `SupportJudgment`, writes `gold_label` into `eval.json`.
  Loud errors on: unknown label, a case missing its `gold:` line, an unknown/duplicate id,
  or a count mismatch vs `eval.json`.
- **`label-train`** — frontier-judges the train partition and writes `gold_label = frontier`
  into `train.json`.

The human edits only the `gold:` lines they disagree with; the frontier proposal is the
default, leveraging the prior that frontier models are strong at P5.

## Sizing (knobs, not hardcoded)

In `P5SpikeConfig` (+ CLI overrides): `cases_per_class` (existing), `cases_per_mode` (new),
`eval_frac` (new, default `0.3`).

**First pass to prove the loop cheaply:** 5/class (20 baseline) + 2 modes × 8 (16) ≈ 36 pool
→ ~12 eval / ~24 train. Tiny for a real fine-tune; crank `cases_per_*` into the hundreds
once the pipeline is validated. Frontier generation + labeling cost scales with pool size.

## Testing

Pure stages are unit-tested (no LLM in pytest; live generation/labeling stays out, like
`generate_cases` today):

- `split_dataset`: dedup; stratified mix is proportional and no class/mode is wholly
  absent from train; determinism (same input → same split); disjointness / no-leakage
  assertion.
- `worksheet`: emit→ingest round-trip (labels match); an override changes the label;
  invalid label errors; missing/extra id errors.
- failure-mode `parse_generated`: sets `intended_class` + `failure_mode`.
- `mypy --strict` (transitively via the test import chain) + `ruff` clean.

## CLI flow

```bash
# 1. generate the pool (baseline 4-class + failure modes); PROOFREAD pool.json
PYTHONPATH=src uv run python spikes/p5_viability/generate_cases.py --out spikes/p5_viability/cases/pool.json

# 2. split into eval/train (pure, deterministic)
PYTHONPATH=src uv run python spikes/p5_viability/split_dataset.py \
    --pool spikes/p5_viability/cases/pool.json \
    --eval spikes/p5_viability/cases/eval.json \
    --train spikes/p5_viability/cases/train.json

# 3a. emit the eval worksheet (frontier proposals), edit it, ingest gold labels
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py emit \
    --cases spikes/p5_viability/cases/eval.json --out spikes/p5_viability/cases/eval_worksheet.md
#    ...edit eval_worksheet.md...
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py ingest \
    --worksheet spikes/p5_viability/cases/eval_worksheet.md --cases spikes/p5_viability/cases/eval.json

# 3b. frontier-label the train partition
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py label-train --cases spikes/p5_viability/cases/train.json
```

## Out of scope

- The fine-tuning training run itself (consumes `train.json`; separate effort).
- Real-literature retrieval (the seam is designed; the loader is future work).
- Semantic dedup / near-duplicate detection beyond exact-normalized.
- Changes to `run_viability.py` / `compare_model.py` (already honor `gold_label`).
