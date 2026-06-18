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
```

The headline is **false_support** per local model vs the frontier reference.
The GO/NO-GO verdict is informational. Proofread `out/review.md` to confirm.
Models are configured in `src/evidenceseeker/config.py::P5SpikeConfig`.
