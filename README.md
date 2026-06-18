# evidenceseeker

Clinical **evidence advisory service** for the [Cairn EHR](https://github.com/cairn-ehr/cairn-ehr)
ecosystem. Given a de-identified clinical question, it returns a **graded,
cited, balanced, auditable** report — every claim carries a verified citation, a
GRADE certainty, the contradicting evidence, and a visible reasoning trace.

> **Status:** skeleton. Typed contracts + the non-inferiority acceptance harness
> are in place; the reasoning primitives (P1–P8) and the service transport are
> not yet built.

## Where it sits in the ecosystem

```
evidenceseeker  →  Kastellan  ↔  Cairn EHR
(evidence)         (trust/egress       (record)
                    de-identification)
```

It is a **pluggable** service — nothing depends on it. Its only client is the
Kastellan broker, which owns de-identification and the local-vs-cloud routing
decision. evidenceseeker itself never sees PHI and holds no per-clinician ACL.
See [`0002-cairn-advisor-reasoning-primitives.md`](https://github.com/cairn-ehr/cairn-ehr/blob/main/docs/ecosystem/0002-cairn-advisor-reasoning-primitives.md)
(grafted into `cairn-ehr/docs/ecosystem`) for the full design — it is
simultaneously the system design, the fine-tuning spec, and the acceptance
criteria.

## Layout

```
src/evidenceseeker/
  contracts.py   # PicoFrame, AdvisoryReport + enums — the wire surface Kastellan codes against
  config.py      # EvalConfig — acceptance-gate tunables (no magic numbers in harness code)
tests/
  test_contracts.py
  acceptance/
    run_noninferiority_eval.py     # walking-skeleton harness (mirrors localmail's run_recall_eval.py)
    gold/noninferiority.example.json
```

## Quickstart

```bash
uv sync --extra dev
uv run pytest                       # contract invariants
uv run mypy                         # strict types

# Run the acceptance harness end-to-end with the stub advisor.
# The stub declines everything, so in-scope items fail the gate and the
# out-of-scope decoy passes — proving the machinery before a real advisor exists.
PYTHONPATH=src uv run python tests/acceptance/run_noninferiority_eval.py \
    --gold tests/acceptance/gold/noninferiority.example.json --dry-run
```

## Next steps

1. Author the real non-inferiority gold set (clinician-written) behind the
   example format.
2. Implement P1 (frame + route, with honest declination) and P5 (PICO-conditioned
   citation support) — the two highest-value primitives — each behind its own harness.
3. Wire a real `Advisor` into the harness and drive the gate to green.

## License

AGPL-3.0-or-later.
