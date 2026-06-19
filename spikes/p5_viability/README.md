# P5 small-model viability spike

Compares local models against a frontier reference on P5 (PICO-conditioned
citation support). See `docs/superpowers/specs/2026-06-18-p5-viability-spike-design.md`.

## Run

```bash
# 1. Generate adversarial cases with the frontier model, then PROOFREAD the JSON.
PYTHONPATH=src uv run python spikes/p5_viability/generate_cases.py \
    --out spikes/p5_viability/cases/generated.json

# 2. Prove the pipeline with no model (echoes intended class):
PYTHONPATH=src uv run python spikes/p5_viability/run_viability.py \
    --cases spikes/p5_viability/cases/generated.json --dry-run

# 3. Real screen (needs Ollama running + ANTHROPIC_API_KEY):
PYTHONPATH=src uv run python spikes/p5_viability/run_viability.py \
    --cases spikes/p5_viability/cases/generated.json \
    --out spikes/p5_viability/out

# 4. Test extra candidate models against the SAVED Sonnet reference (no
#    frontier re-run); writes compare_judgments.json + compare_review.md:
PYTHONPATH=src uv run python spikes/p5_viability/compare_model.py \
    ollama:gpt-oss:20b ollama:phi4:14b --out spikes/p5_viability/out
```

`compare_model.py` reuses `out/judgments.json` as the reference, so it only
pays for the models you name. With `--out` it **accumulates**: each run merges
its candidates into `out/compare_judgments.json` and prints the full leaderboard
so far, so you can add models one at a time (re-running a model refreshes its
row). A case's human `gold_label` overrides the Sonnet reference for both
runners.

The headline is **false_support** per local model vs the frontier reference.
The GO/NO-GO verdict is informational. The reference model's own row is scored
against itself (absent human gold labels), so it is trivially perfect and
flagged `<- reference` — don't read it as a result. Proofread `out/review.md`
to confirm; raw judgments are persisted to `out/judgments.json` after each
model, so a crash mid-sweep keeps completed work. Models are configured in
`src/evidenceseeker/config.py::P5SpikeConfig`.

Both runners report per-model **e2e wall-clock** in the `secs` column and show
live progress while judging (a `tqdm` bar if `tqdm` is installed, otherwise a
plain `model: i/total` counter on stderr). `compare_model.py` persists these to
`out/compare_timings.json` and reloads them, so the `secs` column is populated
for every accumulated model on later runs — not just the one judged this run.
`secs` is still `-` for the saved reference and for dry-runs.

## Gold set (Stage 2)

Build a human-verified **eval** set + a frontier-labeled **train** set for fine-tuning.
See `docs/superpowers/specs/2026-06-19-p5-goldset-design.md`.

```bash
# 1. generate the pool (baseline 4-class + significance/applicability failure modes); PROOFREAD it
PYTHONPATH=src uv run python spikes/p5_viability/generate_cases.py --out spikes/p5_viability/cases/pool.json

# 2. deterministic, leakage-guarded split
PYTHONPATH=src uv run python spikes/p5_viability/split_dataset.py \
    --pool spikes/p5_viability/cases/pool.json \
    --eval spikes/p5_viability/cases/eval.json --train spikes/p5_viability/cases/train.json

# 3. eval: emit worksheet (frontier proposes), edit the `gold:` lines, ingest
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py emit \
    --cases spikes/p5_viability/cases/eval.json --out spikes/p5_viability/cases/eval_worksheet.md
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py ingest \
    --worksheet spikes/p5_viability/cases/eval_worksheet.md --cases spikes/p5_viability/cases/eval.json

# 4. train: frontier-label at scale
PYTHONPATH=src uv run python spikes/p5_viability/worksheet.py label-train --cases spikes/p5_viability/cases/train.json
```

Sizes are config knobs (`cases_per_class`, `cases_per_mode`, `eval_frac` in `P5SpikeConfig`).
Real-literature passages can replace step 1 later — split/worksheet/label are source-agnostic.
