# P5 Small-Model Viability Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable P5 citation-support primitive plus a disposable harness that compares several local models against a frontier reference on P5, so we can decide whether small-model evidence appraisal is worth fine-tuning.

**Architecture:** P5 is a `bmlib.agents.BaseAgent` subclass in `src/` (model-agnostic — the `provider:model` string is passed in, so "run across N models" is N instances). Case generation, pure metrics/scoring, and the live comparison runner are disposable artifacts in a new top-level `spikes/` tree. A case's reference label is its human `gold_label` if present, else the frontier model's blind judgment — the seam that lets a future gold set slot in with no runner change.

**Tech Stack:** Python 3.12, pydantic v2, bmlib (LLM abstraction + agent base + Jinja templates), Ollama (local judges), Anthropic Claude Sonnet 4.6 (frontier reference/generator), pytest, mypy strict.

## Global Constraints

- Python `>=3.12`; pydantic `>=2.7`. Copy verbatim into any new code.
- Every source file starts with `# SPDX-License-Identifier: AGPL-3.0-or-later` and `from __future__ import annotations`.
- Reuse existing contract types from `src/evidenceseeker/contracts.py` (`PicoFrame`, `SupportJudgment`, `_Frozen`, etc.) — do NOT redefine them.
- `contracts.py` is the Kastellan wire surface; do NOT add spike-only types to it.
- Frozen pydantic models everywhere (`extra="forbid"`), mirroring `contracts.py`.
- Local judge models: `ollama:medgemma1.5:4b-it-q8_0`, `ollama:medgemma:27b-it-q8_0`, `ollama:qwen3.6:35b-a3b-q8_0`. Frontier reference + generator: `anthropic:claude-sonnet-4-6`.
- `spikes/` is live, non-deterministic, paid: it stays out of `pytest` collection (`testpaths=["tests"]`). Pure spike modules (`cases.py`, `metrics.py`, `scoring.py`) are imported by tests and so must be fully type-annotated and mypy-strict-clean. Live modules (`generate_cases.py`, `run_viability.py`) are never imported by tests.
- The `max_false_support_rate` verdict is informational; the runner does NOT hard-fail on it.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` (modify) | bmlib path dependency + extras, pytest/mypy config |
| `.gitignore` (modify) | ignore `spikes/p5_viability/out/` |
| `src/evidenceseeker/primitives/__init__.py` (create) | package marker |
| `src/evidenceseeker/primitives/types.py` (create) | `CitationJudgment` — P5 output, bmlib-free so pure modules can import it |
| `src/evidenceseeker/primitives/p5_verify_citation.py` (create) | `P5VerifyCitation` agent + `make_p5_agent` factory |
| `src/evidenceseeker/prompts/p5_verify_citation.jinja` (create) | PICO-conditioned support-judgment prompt |
| `src/evidenceseeker/config.py` (modify) | add `P5SpikeConfig` |
| `spikes/__init__.py`, `spikes/p5_viability/__init__.py` (create) | importable packages |
| `spikes/p5_viability/cases.py` (create) | `P5Case` + load/save/`reference_label` |
| `spikes/p5_viability/metrics.py` (create) | pure metric functions |
| `spikes/p5_viability/scoring.py` (create) | `ModelScore`, `score_run`, `render_review` |
| `spikes/p5_viability/generate_cases.py` (create) | live: frontier case generator CLI |
| `spikes/p5_viability/run_viability.py` (create) | live: judge runner CLI (+ `--dry-run`) |
| `spikes/p5_viability/README.md` (create) | how to run the spike |
| `tests/_helpers.py` (create) | `make_pico()` shared test fixture builder |
| `tests/test_p5_case_schema.py` (create) | `P5Case` round-trip + `reference_label` |
| `tests/test_p5_verify_citation.py` (create) | agent prompt+parse with a fake LLM |
| `tests/test_p5_metrics.py` (create) | metric correctness |
| `tests/test_p5_scoring.py` (create) | `score_run` + `render_review` over synthetic data |

---

## Task 1: Project wiring + schemas (`CitationJudgment`, `P5Case`)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/evidenceseeker/primitives/__init__.py`
- Create: `src/evidenceseeker/primitives/types.py`
- Create: `spikes/__init__.py`
- Create: `spikes/p5_viability/__init__.py`
- Create: `spikes/p5_viability/cases.py`
- Create: `tests/_helpers.py`
- Test: `tests/test_p5_case_schema.py`

**Interfaces:**
- Produces: `CitationJudgment(support: SupportJudgment, reason: str, pico_match_notes: str | None = None)` (frozen).
- Produces: `P5Case(id: str, pico: PicoFrame, claim: str, passage: str, intended_class: SupportJudgment, gold_label: SupportJudgment | None = None, notes: str | None = None)` (frozen).
- Produces: `load_cases(path: Path) -> list[P5Case]`, `save_cases(cases: list[P5Case], path: Path) -> None`, `reference_label(case: P5Case, frontier: SupportJudgment) -> SupportJudgment`.
- Produces: `tests/_helpers.py::make_pico(*, intervention: str = "DrugA", comparator: str = "DrugB", dose: str | None = None) -> PicoFrame`.

- [ ] **Step 1: Wire dependencies and tool config in `pyproject.toml`**

Replace the `dependencies` line and the `[tool.pytest.ini_options]` / `[tool.mypy]` blocks, and add a `[tool.uv.sources]` block:

```toml
dependencies = ["pydantic>=2.7", "bmlib[anthropic,ollama]"]

[tool.uv.sources]
bmlib = { path = "../../bmlib", editable = true }

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "."
files = ["src", "tests"]

[[tool.mypy.overrides]]
module = "bmlib.*"
ignore_missing_imports = true
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync --extra dev`
Expected: resolves and installs `bmlib` (editable, from `../../bmlib`) with `anthropic` + `ollama` extras; exit 0.

- [ ] **Step 3: Ignore the spike output dir**

Append to `.gitignore`:

```
spikes/p5_viability/out/
```

- [ ] **Step 4: Create the `primitives` package and `CitationJudgment`**

Create `src/evidenceseeker/primitives/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reasoning primitives (P1..P8). Only P5 is implemented so far."""
```

Create `src/evidenceseeker/primitives/types.py`:

```python
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
```

- [ ] **Step 5: Create the `spikes` packages and `P5Case`**

Create `spikes/__init__.py` and `spikes/p5_viability/__init__.py`, each:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

Create `spikes/p5_viability/cases.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 spike case schema and JSON persistence.

A case's reference label is its human ``gold_label`` if present, else the
frontier model's blind judgment — see ``reference_label``. This is the seam
that lets a future gold set slot in with no runner change.
"""

from __future__ import annotations

import json
from pathlib import Path

from evidenceseeker.contracts import PicoFrame, SupportJudgment, _Frozen


class P5Case(_Frozen):
    id: str
    pico: PicoFrame
    claim: str
    passage: str
    intended_class: SupportJudgment
    gold_label: SupportJudgment | None = None
    notes: str | None = None


def load_cases(path: Path) -> list[P5Case]:
    raw = json.loads(Path(path).read_text())
    return [P5Case.model_validate(item) for item in raw]


def save_cases(cases: list[P5Case], path: Path) -> None:
    payload = [c.model_dump(mode="json") for c in cases]
    Path(path).write_text(json.dumps(payload, indent=2))


def reference_label(case: P5Case, frontier: SupportJudgment) -> SupportJudgment:
    """Human gold label when present, else the frontier model's judgment."""
    return case.gold_label if case.gold_label is not None else frontier
```

- [ ] **Step 6: Create the shared test helper**

Create `tests/_helpers.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared builders for tests."""

from __future__ import annotations

from evidenceseeker.contracts import (
    Archetype,
    Outcome,
    OutcomeType,
    PicoFrame,
    Population,
    Therapy,
)


def make_pico(
    *,
    intervention: str = "DrugA",
    comparator: str = "DrugB",
    dose: str | None = None,
) -> PicoFrame:
    return PicoFrame(
        population=Population(age_band="65-74", sex="any", settings=["outpatient"]),
        intervention=Therapy(label=intervention, dose=dose),
        comparator=Therapy(label=comparator),
        outcomes=[Outcome(label="all-cause mortality", type=OutcomeType.EFFICACY)],
        archetype=Archetype.NON_INFERIORITY,
        question_text="Is DrugA non-inferior to DrugB for mortality?",
        applicability="community outpatients aged 65-74",
    )
```

- [ ] **Step 7: Write the failing schema test**

Create `tests/test_p5_case_schema.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from evidenceseeker.contracts import SupportJudgment
from spikes.p5_viability.cases import (
    P5Case,
    load_cases,
    reference_label,
    save_cases,
)
from tests._helpers import make_pico


def _case(**overrides: object) -> P5Case:
    base = dict(
        id="supports-0",
        pico=make_pico(),
        claim="DrugA is non-inferior to DrugB for mortality.",
        passage="In a non-inferiority RCT, DrugA was non-inferior to DrugB.",
        intended_class=SupportJudgment.SUPPORTS,
    )
    base.update(overrides)
    return P5Case(**base)  # type: ignore[arg-type]


def test_roundtrip_preserves_fields(tmp_path: Path) -> None:
    cases = [_case(), _case(id="partial-0", intended_class=SupportJudgment.PARTIAL)]
    path = tmp_path / "cases.json"
    save_cases(cases, path)
    loaded = load_cases(path)
    assert loaded == cases


def test_reference_label_prefers_gold() -> None:
    case = _case(gold_label=SupportJudgment.DOES_NOT)
    assert reference_label(case, SupportJudgment.SUPPORTS) is SupportJudgment.DOES_NOT


def test_reference_label_falls_back_to_frontier() -> None:
    case = _case(gold_label=None)
    assert reference_label(case, SupportJudgment.CONTRADICTS) is SupportJudgment.CONTRADICTS
```

- [ ] **Step 8: Run the test to verify it fails**

Run: `uv run pytest tests/test_p5_case_schema.py -v`
Expected: collection import succeeds; tests PASS (schema + helpers already written in this task). If any FAIL, fix the schema/helper until green.

- [ ] **Step 9: Run mypy**

Run: `uv run mypy`
Expected: `Success: no issues found`.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore src/evidenceseeker/primitives spikes tests/_helpers.py tests/test_p5_case_schema.py
git commit -m "$(cat <<'EOF'
feat: P5 spike wiring + CitationJudgment/P5Case schemas

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `P5VerifyCitation` agent + prompt template

**Files:**
- Create: `src/evidenceseeker/prompts/p5_verify_citation.jinja`
- Create: `src/evidenceseeker/primitives/p5_verify_citation.py`
- Test: `tests/test_p5_verify_citation.py`

**Interfaces:**
- Consumes: `CitationJudgment` (Task 1); `PicoFrame`, `SupportJudgment` (contracts); `make_pico()` (Task 1); `bmlib.agents.BaseAgent`, `bmlib.llm.LLMClient`, `bmlib.llm.LLMResponse`, `bmlib.templates.TemplateEngine`.
- Produces: `class P5VerifyCitation(BaseAgent)` with `verify(self, *, pico: PicoFrame, claim: str, passage: str) -> CitationJudgment`.
- Produces: `make_p5_agent(llm: LLMClient, model: str, *, temperature: float = 0.0) -> P5VerifyCitation`.

- [ ] **Step 1: Create the prompt template**

Create `src/evidenceseeker/prompts/p5_verify_citation.jinja`:

```jinja
You are judging whether ONE passage supports a clinical claim, conditioned on a PICO frame.

PICO FRAME
Population: age_band={{ pico.population.age_band }} sex={{ pico.population.sex }} settings={{ pico.population.settings }}
  comorbidities: {% for c in pico.population.comorbidities %}{{ c.display }}{% if not loop.last %}, {% endif %}{% endfor %}
Intervention: {{ pico.intervention.label }}{% if pico.intervention.dose %} (dose {{ pico.intervention.dose }}){% endif %}
Comparator: {{ pico.comparator.label }}{% if pico.comparator.dose %} (dose {{ pico.comparator.dose }}){% endif %}
Outcomes: {% for o in pico.outcomes %}{{ o.label }} [{{ o.type }}]{% if not loop.last %}; {% endif %}{% endfor %}
Applicability (population/setting the answer must generalize TO): {{ pico.applicability }}
Archetype: {{ pico.archetype }}

CLAIM
{{ claim }}

CANDIDATE PASSAGE
{{ passage }}

TASK
Decide how the passage relates to the claim, CONDITIONED ON THE PICO FRAME. A passage about a
different population, intervention, comparator, outcome, or dose than the PICO frame is at best
"partial" and usually "does_not" — it must NEVER be "supports". Labels:
  - "supports":    directly substantiates the claim for THIS PICO.
  - "partial":     relevant but mismatched on population/dose/outcome/comparator, or only indirect.
  - "does_not":    does not substantiate the claim (off-target).
  - "contradicts": evidence against the claim for this PICO.

Respond with ONLY a JSON object, no prose:
{"support": "supports|partial|does_not|contradicts", "reason": "<one sentence>", "pico_match_notes": "<which PICO elements matched/mismatched>"}
```

- [ ] **Step 2: Write the agent and factory**

Create `src/evidenceseeker/primitives/p5_verify_citation.py`:

```python
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


class P5VerifyCitation(BaseAgent):
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
```

- [ ] **Step 3: Write the failing agent test**

Create `tests/test_p5_verify_citation.py`:

```python
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
    agent = make_p5_agent(llm, "ollama:test", temperature=0.0)  # type: ignore[arg-type]

    result = agent.verify(
        pico=make_pico(intervention="DrugA", dose="10mg"),
        claim="DrugA is non-inferior to DrugB.",
        passage="A trial in children found DrugA non-inferior to DrugB.",
    )

    assert result.support is SupportJudgment.PARTIAL
    assert result.reason == "Right drug, wrong population."
    # The rendered prompt must actually carry the PICO frame.
    rendered = llm.last_messages[1].content  # type: ignore[index]
    assert "DrugA" in rendered and "10mg" in rendered
    assert "community outpatients aged 65-74" in rendered
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_p5_verify_citation.py -v`
Expected: PASS. If `chat_json` retries (it shouldn't on valid JSON), confirm the canned content is valid JSON.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/evidenceseeker/prompts src/evidenceseeker/primitives/p5_verify_citation.py tests/test_p5_verify_citation.py
git commit -m "$(cat <<'EOF'
feat: P5VerifyCitation agent + PICO-conditioned prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Metrics

**Files:**
- Create: `spikes/p5_viability/metrics.py`
- Test: `tests/test_p5_metrics.py`

**Interfaces:**
- Consumes: `SupportJudgment` (contracts).
- Produces: `false_support_rate(model, reference) -> float`, `missed_support_rate(model, reference) -> float`, `accuracy(model, reference) -> float`, `cohen_kappa(a, b) -> float`, `confusion(model, reference) -> dict[tuple[SupportJudgment, SupportJudgment], int]`. All take `list[SupportJudgment]` and require equal-length inputs (`zip(..., strict=True)`).

- [ ] **Step 1: Write the failing metrics test**

Create `tests/test_p5_metrics.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment as S
from spikes.p5_viability.metrics import (
    accuracy,
    cohen_kappa,
    confusion,
    false_support_rate,
    missed_support_rate,
)


def test_false_support_rate_counts_only_non_support_references() -> None:
    # references: 2 non-support (DOES_NOT, PARTIAL), 1 support.
    reference = [S.DOES_NOT, S.PARTIAL, S.SUPPORTS]
    # model calls the first non-support case "supports" -> 1/2.
    model = [S.SUPPORTS, S.PARTIAL, S.SUPPORTS]
    assert false_support_rate(model, reference) == 0.5


def test_false_support_rate_zero_when_no_negatives() -> None:
    assert false_support_rate([S.SUPPORTS], [S.SUPPORTS]) == 0.0


def test_missed_support_rate() -> None:
    reference = [S.SUPPORTS, S.SUPPORTS]
    model = [S.SUPPORTS, S.PARTIAL]
    assert missed_support_rate(model, reference) == 0.5


def test_accuracy() -> None:
    assert accuracy([S.SUPPORTS, S.PARTIAL], [S.SUPPORTS, S.DOES_NOT]) == 0.5


def test_cohen_kappa_perfect_agreement() -> None:
    a = [S.SUPPORTS, S.PARTIAL, S.DOES_NOT]
    assert cohen_kappa(a, a) == 1.0


def test_confusion_counts_pairs() -> None:
    c = confusion([S.SUPPORTS, S.SUPPORTS], [S.SUPPORTS, S.PARTIAL])
    assert c[(S.SUPPORTS, S.SUPPORTS)] == 1
    assert c[(S.SUPPORTS, S.PARTIAL)] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_p5_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: spikes.p5_viability.metrics`.

- [ ] **Step 3: Implement the metrics**

Create `spikes/p5_viability/metrics.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure scoring functions for the P5 viability spike.

All functions take equal-length lists of ``SupportJudgment`` and use
``zip(..., strict=True)`` so a length mismatch is a loud error, not a silent
truncation.
"""

from __future__ import annotations

from collections import Counter

from evidenceseeker.contracts import SupportJudgment

_SUPPORTS = SupportJudgment.SUPPORTS


def false_support_rate(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    """Of cases whose reference is NOT 'supports', the fraction the model
    called 'supports'. The trust headline."""
    negatives = [m for m, r in zip(model, reference, strict=True) if r is not _SUPPORTS]
    if not negatives:
        return 0.0
    return sum(1 for m in negatives if m is _SUPPORTS) / len(negatives)


def missed_support_rate(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    """Of cases whose reference IS 'supports', the fraction the model missed."""
    positives = [m for m, r in zip(model, reference, strict=True) if r is _SUPPORTS]
    if not positives:
        return 0.0
    return sum(1 for m in positives if m is not _SUPPORTS) / len(positives)


def accuracy(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> float:
    pairs = list(zip(model, reference, strict=True))
    if not pairs:
        return 0.0
    return sum(1 for m, r in pairs if m is r) / len(pairs)


def cohen_kappa(a: list[SupportJudgment], b: list[SupportJudgment]) -> float:
    pairs = list(zip(a, b, strict=True))
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in pairs if x is y) / n
    count_a = Counter(a)
    count_b = Counter(b)
    labels = set(a) | set(b)
    expected = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion(
    model: list[SupportJudgment], reference: list[SupportJudgment]
) -> dict[tuple[SupportJudgment, SupportJudgment], int]:
    return dict(Counter(zip(model, reference, strict=True)))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_p5_metrics.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Run mypy**

Run: `uv run mypy`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add spikes/p5_viability/metrics.py tests/test_p5_metrics.py
git commit -m "$(cat <<'EOF'
feat: P5 spike metrics (false-support rate, kappa, confusion)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Scoring + review artifact (pure)

**Files:**
- Create: `spikes/p5_viability/scoring.py`
- Test: `tests/test_p5_scoring.py`

**Interfaces:**
- Consumes: `P5Case`, `reference_label` (Task 1); `CitationJudgment` (Task 1); metrics (Task 3); `SupportJudgment` (contracts).
- Produces: `ModelScore(model: str, n: int, accuracy: float, false_support_rate: float, missed_support_rate: float, kappa_vs_frontier: float)` (frozen dataclass).
- Produces: `score_run(cases: list[P5Case], judgments: dict[str, list[CitationJudgment]], reference_model: str) -> list[ModelScore]` — `judgments` maps each model string to a per-case-aligned list; reference labels come from `reference_label(case, frontier_judgment.support)`; the frontier model itself is scored too (its kappa-vs-itself is 1.0).
- Produces: `render_review(cases: list[P5Case], judgments: dict[str, list[CitationJudgment]], reference_model: str) -> str` — markdown, one block per case, listing PICO/claim/passage/intended + every model's judgment and reason.

- [ ] **Step 1: Write the failing scoring test**

Create `tests/test_p5_scoring.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment as S
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability.cases import P5Case
from spikes.p5_viability.scoring import ModelScore, render_review, score_run
from tests._helpers import make_pico


def _case(cid: str, intended: S, gold: S | None = None) -> P5Case:
    return P5Case(
        id=cid,
        pico=make_pico(),
        claim="DrugA is non-inferior to DrugB.",
        passage="Some passage.",
        intended_class=intended,
        gold_label=gold,
    )


def _j(support: S) -> CitationJudgment:
    return CitationJudgment(support=support, reason="r")


def test_score_run_uses_frontier_as_reference_and_flags_false_support() -> None:
    cases = [_case("a", S.DOES_NOT), _case("b", S.SUPPORTS)]
    judgments = {
        "frontier": [_j(S.DOES_NOT), _j(S.SUPPORTS)],   # reference
        "local": [_j(S.SUPPORTS), _j(S.SUPPORTS)],      # false-supports case "a"
    }
    scores = score_run(cases, judgments, reference_model="frontier")
    by_model = {s.model: s for s in scores}

    assert isinstance(by_model["local"], ModelScore)
    assert by_model["local"].false_support_rate == 1.0   # 1 negative, called supports
    assert by_model["frontier"].false_support_rate == 0.0
    assert by_model["frontier"].kappa_vs_frontier == 1.0


def test_score_run_prefers_gold_label_over_frontier() -> None:
    # Frontier says SUPPORTS but the human gold says DOES_NOT -> "local" matching
    # frontier is now a false support against gold.
    cases = [_case("a", S.SUPPORTS, gold=S.DOES_NOT)]
    judgments = {
        "frontier": [_j(S.SUPPORTS)],
        "local": [_j(S.SUPPORTS)],
    }
    scores = {s.model: s for s in score_run(cases, judgments, "frontier")}
    assert scores["local"].false_support_rate == 1.0


def test_render_review_includes_claim_and_each_models_judgment() -> None:
    cases = [_case("a", S.DOES_NOT)]
    judgments = {"frontier": [_j(S.DOES_NOT)], "local": [_j(S.SUPPORTS)]}
    md = render_review(cases, judgments, "frontier")
    assert "DrugA is non-inferior to DrugB." in md
    assert "local" in md and "supports" in md
    assert "frontier" in md and "does_not" in md
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_p5_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: spikes.p5_viability.scoring`.

- [ ] **Step 3: Implement scoring + review**

Create `spikes/p5_viability/scoring.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score a P5 viability run and render a human-proofreadable review artifact.

Pure: no LLM, no I/O. Live judging lives in ``run_viability.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidenceseeker.contracts import SupportJudgment
from evidenceseeker.primitives.types import CitationJudgment
from spikes.p5_viability import metrics
from spikes.p5_viability.cases import P5Case, reference_label


@dataclass(frozen=True)
class ModelScore:
    model: str
    n: int
    accuracy: float
    false_support_rate: float
    missed_support_rate: float
    kappa_vs_frontier: float


def _reference_labels(
    cases: list[P5Case], frontier: list[CitationJudgment]
) -> list[SupportJudgment]:
    return [
        reference_label(case, fj.support)
        for case, fj in zip(cases, frontier, strict=True)
    ]


def score_run(
    cases: list[P5Case],
    judgments: dict[str, list[CitationJudgment]],
    reference_model: str,
) -> list[ModelScore]:
    frontier = judgments[reference_model]
    reference = _reference_labels(cases, frontier)
    frontier_supports = [j.support for j in frontier]

    scores: list[ModelScore] = []
    for model, judged in judgments.items():
        supports = [j.support for j in judged]
        scores.append(
            ModelScore(
                model=model,
                n=len(cases),
                accuracy=metrics.accuracy(supports, reference),
                false_support_rate=metrics.false_support_rate(supports, reference),
                missed_support_rate=metrics.missed_support_rate(supports, reference),
                kappa_vs_frontier=metrics.cohen_kappa(supports, frontier_supports),
            )
        )
    return scores


def render_review(
    cases: list[P5Case],
    judgments: dict[str, list[CitationJudgment]],
    reference_model: str,
) -> str:
    lines: list[str] = ["# P5 viability review\n"]
    for idx, case in enumerate(cases):
        ref = reference_label(case, judgments[reference_model][idx].support)
        lines.append(f"## {case.id}  (intended={case.intended_class.value}, reference={ref.value})")
        lines.append(f"- **claim:** {case.claim}")
        lines.append(f"- **passage:** {case.passage}")
        lines.append(f"- **applicability:** {case.pico.applicability}")
        lines.append("")
        lines.append("| model | judgment | reason |")
        lines.append("|---|---|---|")
        for model, judged in judgments.items():
            j = judged[idx]
            lines.append(f"| {model} | {j.support.value} | {j.reason} |")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_p5_scoring.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Run mypy**

Run: `uv run mypy`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add spikes/p5_viability/scoring.py tests/test_p5_scoring.py
git commit -m "$(cat <<'EOF'
feat: P5 spike scoring + markdown review artifact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Frontier case generator (live)

**Files:**
- Create: `spikes/p5_viability/generate_cases.py`
- Modify: `src/evidenceseeker/config.py` (add `P5SpikeConfig`)
- Test: `tests/test_p5_generate_parse.py`

**Interfaces:**
- Consumes: `P5Case`, `save_cases` (Task 1); `PicoFrame`, `SupportJudgment` (contracts); `bmlib.llm.LLMClient`; `bmlib.agents.BaseAgent.parse_json`.
- Produces: `P5SpikeConfig` in `config.py` with fields `judge_models: list[str]`, `reference_model: str`, `generator_model: str`, `cases_per_class: int = 5`, `temperature: float = 0.0`, `max_false_support_rate: float = 0.10`.
- Produces: `build_generation_prompt(target: SupportJudgment, n: int) -> str` and `parse_generated(data: dict, target: SupportJudgment) -> list[P5Case]` (pure, unit-tested). `data` shape: `{"cases": [{"pico": {...PicoFrame...}, "claim": str, "passage": str, "notes": str}]}`; ids are `f"{target.value}-{i}"`, `intended_class=target`.

- [ ] **Step 1: Add `P5SpikeConfig`**

Append to `src/evidenceseeker/config.py`:

```python
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
```

Add `ConfigDict` to the existing pydantic import if not already imported (the file already imports `BaseModel, ConfigDict, Field`).

- [ ] **Step 2: Write the failing parse test**

Create `tests/test_p5_generate_parse.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from evidenceseeker.contracts import SupportJudgment
from spikes.p5_viability.generate_cases import build_generation_prompt, parse_generated


def test_build_generation_prompt_names_target_and_count() -> None:
    prompt = build_generation_prompt(SupportJudgment.PARTIAL, 3)
    assert "partial" in prompt
    assert "3" in prompt


def test_parse_generated_attaches_id_and_intended_class() -> None:
    data = {
        "cases": [
            {
                "pico": {
                    "population": {"age_band": "65-74", "sex": "any", "settings": ["outpatient"]},
                    "intervention": {"label": "DrugA", "dose": "10mg"},
                    "comparator": {"label": "DrugB"},
                    "outcomes": [{"label": "mortality", "type": "efficacy"}],
                    "archetype": "non_inferiority",
                    "question_text": "Is DrugA non-inferior to DrugB?",
                    "applicability": "outpatients 65-74",
                },
                "claim": "DrugA is non-inferior to DrugB.",
                "passage": "A pediatric trial found DrugA non-inferior to DrugB.",
                "notes": "population mismatch (pediatric vs elderly)",
            }
        ]
    }
    cases = parse_generated(data, SupportJudgment.DOES_NOT)
    assert len(cases) == 1
    assert cases[0].id == "does_not-0"
    assert cases[0].intended_class is SupportJudgment.DOES_NOT
    assert cases[0].pico.intervention.label == "DrugA"
    assert cases[0].notes is not None
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_p5_generate_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: spikes.p5_viability.generate_cases`.

- [ ] **Step 4: Implement the generator**

Create `spikes/p5_viability/generate_cases.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frontier-authored P5 cases for the first (screen) spike.

The generator records the class it was asked to construct as ``intended_class``
but that label is NEVER shown to a judge. Hand-proofread the emitted JSON before
running the comparison. Disposable: Stage 2 replaces this with real literature.

    PYTHONPATH=src python spikes/p5_viability/generate_cases.py \
        --out spikes/p5_viability/cases/generated.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bmlib.agents import BaseAgent  # noqa: E402
from bmlib.llm import LLMClient, LLMMessage  # noqa: E402

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.contracts import PicoFrame, SupportJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, save_cases  # noqa: E402

_GUIDANCE = {
    SupportJudgment.SUPPORTS: "the passage directly substantiates the claim for the exact PICO",
    SupportJudgment.PARTIAL: "the passage is relevant but mismatched on population, dose, comparator, or outcome",
    SupportJudgment.DOES_NOT: "the passage is off-target and does not substantiate the claim",
    SupportJudgment.CONTRADICTS: "the passage provides evidence AGAINST the claim for this PICO",
}


def build_generation_prompt(target: SupportJudgment, n: int) -> str:
    return (
        f"Author {n} realistic clinical evidence cases where the relationship between the "
        f"CLAIM and the PASSAGE is '{target.value}': {_GUIDANCE[target]}.\n"
        "Make the near-misses subtle and adversarial (e.g. right drug but wrong population "
        "or dose). Each case needs a full PICO frame.\n\n"
        "Return ONLY JSON:\n"
        '{"cases": [{"pico": {"population": {"age_band": str, "sex": str, "settings": [str], '
        '"comorbidities": []}, "intervention": {"label": str, "dose": str|null}, '
        '"comparator": {"label": str}, "outcomes": [{"label": str, "type": '
        '"efficacy|harm|surrogate"}], "archetype": "non_inferiority", "question_text": str, '
        '"applicability": str}, "claim": str, "passage": str, "notes": str}]}'
    )


def parse_generated(data: dict, target: SupportJudgment) -> list[P5Case]:
    out: list[P5Case] = []
    for i, raw in enumerate(data["cases"]):
        out.append(
            P5Case(
                id=f"{target.value}-{i}",
                pico=PicoFrame.model_validate(raw["pico"]),
                claim=str(raw["claim"]),
                passage=str(raw["passage"]),
                intended_class=target,
                notes=raw.get("notes"),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = P5SpikeConfig()
    llm = LLMClient()
    agent = BaseAgent(llm=llm, model=cfg.generator_model, temperature=0.7)

    all_cases: list[P5Case] = []
    for target in SupportJudgment:
        prompt = build_generation_prompt(target, cfg.cases_per_class)
        data = agent.chat_json([LLMMessage(role="user", content=prompt)])
        all_cases.extend(parse_generated(data, target))
        print(f"generated {cfg.cases_per_class} cases for {target.value}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_cases(all_cases, args.out)
    print(f"wrote {len(all_cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_p5_generate_parse.py -v`
Expected: PASS (both). Note: the test imports only `build_generation_prompt`/`parse_generated`; the bmlib-using `main` is not exercised.

- [ ] **Step 6: Run the full suite and mypy**

Run: `uv run pytest && uv run mypy`
Expected: all tests PASS; mypy `Success`.

- [ ] **Step 7: Commit**

```bash
git add src/evidenceseeker/config.py spikes/p5_viability/generate_cases.py tests/test_p5_generate_parse.py
git commit -m "$(cat <<'EOF'
feat: P5SpikeConfig + frontier case generator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Comparison runner (live, with `--dry-run`) + README

**Files:**
- Create: `spikes/p5_viability/run_viability.py`
- Create: `spikes/p5_viability/README.md`
- Test: `tests/test_p5_run_dryrun.py`

**Interfaces:**
- Consumes: `load_cases`, `P5Case` (Task 1); `make_p5_agent` (Task 2); `score_run`, `render_review`, `ModelScore` (Task 4); `CitationJudgment` (Task 1); `P5SpikeConfig` (Task 5); `bmlib.llm.LLMClient`.
- Produces: `dry_run_judgments(cases: list[P5Case], models: list[str]) -> dict[str, list[CitationJudgment]]` — each judge echoes `case.intended_class` (proves the pipeline without any model).
- Produces: `verdict_lines(scores: list[ModelScore], cfg: P5SpikeConfig) -> list[str]` (pure, unit-tested) — a GO/NO-GO line per local judge vs `max_false_support_rate`, informational only.
- Produces: `main(argv) -> int` with `--cases`, `--format {text,json}`, `--out DIR`, `--dry-run`.

- [ ] **Step 1: Write the failing dry-run test**

Create `tests/test_p5_run_dryrun.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_p5_run_dryrun.py -v`
Expected: FAIL — `ModuleNotFoundError: spikes.p5_viability.run_viability`.

- [ ] **Step 3: Implement the runner**

Create `spikes/p5_viability/run_viability.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 viability runner: judge every case with each local model + the frontier
reference, score, and emit a comparison table plus a proofreadable review.

    # prove the machinery without any model:
    PYTHONPATH=src python spikes/p5_viability/run_viability.py \
        --cases spikes/p5_viability/cases/generated.json --dry-run

    # real screen:
    PYTHONPATH=src python spikes/p5_viability/run_viability.py \
        --cases spikes/p5_viability/cases/generated.json --out spikes/p5_viability/out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evidenceseeker.config import P5SpikeConfig  # noqa: E402
from evidenceseeker.primitives.types import CitationJudgment  # noqa: E402
from spikes.p5_viability.cases import P5Case, load_cases  # noqa: E402
from spikes.p5_viability.scoring import ModelScore, render_review, score_run  # noqa: E402


def dry_run_judgments(
    cases: list[P5Case], models: list[str]
) -> dict[str, list[CitationJudgment]]:
    """Each model echoes the intended class — proves the pipeline, no LLM."""
    return {
        model: [
            CitationJudgment(support=c.intended_class, reason="dry-run echo")
            for c in cases
        ]
        for model in models
    }


def live_judgments(
    cases: list[P5Case], models: list[str], cfg: P5SpikeConfig
) -> dict[str, list[CitationJudgment]]:
    from bmlib.llm import LLMClient

    from evidenceseeker.primitives.p5_verify_citation import make_p5_agent

    llm = LLMClient()
    out: dict[str, list[CitationJudgment]] = {}
    for model in models:
        agent = make_p5_agent(llm, model, temperature=cfg.temperature)
        out[model] = []
        for c in cases:
            out[model].append(agent.verify(pico=c.pico, claim=c.claim, passage=c.passage))
        print(f"judged {len(cases)} cases with {model}", file=sys.stderr)
    return out


def verdict_lines(scores: list[ModelScore], cfg: P5SpikeConfig) -> list[str]:
    lines: list[str] = []
    for s in scores:
        if s.model not in cfg.judge_models:
            continue
        flag = "GO" if s.false_support_rate <= cfg.max_false_support_rate else "NO-GO"
        lines.append(
            f"[{flag}] {s.model}: false_support={s.false_support_rate:.2f} "
            f"(threshold {cfg.max_false_support_rate:.2f}, informational)"
        )
    return lines


def _print_text(scores: list[ModelScore], cfg: P5SpikeConfig) -> None:
    print(f"\n{'model':<34} {'n':>3} {'acc':>5} {'false_sup':>10} {'missed':>7} {'kappa':>6}")
    print("-" * 70)
    for s in scores:
        print(
            f"{s.model:<34} {s.n:>3} {s.accuracy:>5.2f} "
            f"{s.false_support_rate:>10.2f} {s.missed_support_rate:>7.2f} "
            f"{s.kappa_vs_frontier:>6.2f}"
        )
    print()
    for line in verdict_lines(scores, cfg):
        print(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--out", type=Path, default=None, help="dir for the review artifact")
    ap.add_argument("--dry-run", action="store_true", help="echo intended class, no LLM")
    args = ap.parse_args(argv)

    cfg = P5SpikeConfig()
    cases = load_cases(args.cases)
    models = [*cfg.judge_models, cfg.reference_model]

    judgments = (
        dry_run_judgments(cases, models)
        if args.dry_run
        else live_judgments(cases, models, cfg)
    )
    scores = score_run(cases, judgments, cfg.reference_model)

    if args.format == "json":
        print(json.dumps({"scores": [s.__dict__ for s in scores]}, indent=2))
    else:
        _print_text(scores, cfg)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        review = render_review(cases, judgments, cfg.reference_model)
        (args.out / "review.md").write_text(review)
        print(f"\nreview -> {args.out / 'review.md'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_p5_run_dryrun.py -v`
Expected: PASS (both). The test imports only `dry_run_judgments` and `verdict_lines`; `live_judgments` (and its bmlib import) is not touched.

- [ ] **Step 5: Write the spike README**

Create `spikes/p5_viability/README.md`:

```markdown
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
```

- [ ] **Step 6: Verify the dry-run end-to-end with a tiny fixture**

Run:
```bash
PYTHONPATH=src uv run python -c "
from pathlib import Path
from evidenceseeker.contracts import SupportJudgment
from spikes.p5_viability.cases import P5Case, save_cases
from tests._helpers import make_pico
cases=[P5Case(id='does_not-0',pico=make_pico(),claim='c',passage='p',intended_class=SupportJudgment.DOES_NOT)]
Path('spikes/p5_viability/cases').mkdir(parents=True,exist_ok=True)
save_cases(cases, Path('spikes/p5_viability/cases/smoke.json'))
"
PYTHONPATH=src uv run python spikes/p5_viability/run_viability.py \
    --cases spikes/p5_viability/cases/smoke.json --dry-run
```
Expected: a comparison table printing all four model rows with `false_sup 0.00` and a `[GO]` line per local judge. Then remove the smoke file: `rm spikes/p5_viability/cases/smoke.json`.

- [ ] **Step 7: Run the full suite and mypy**

Run: `uv run pytest && uv run mypy`
Expected: all tests PASS; mypy `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add spikes/p5_viability/run_viability.py spikes/p5_viability/README.md tests/test_p5_run_dryrun.py
git commit -m "$(cat <<'EOF'
feat: P5 viability comparison runner (+ dry-run) and README

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- P5 reusable primitive (`P5VerifyCitation` + prompt) → Task 2 ✓
- `CitationJudgment` / `P5Case` schemas, `gold_label`-or-frontier seam → Task 1 (`reference_label`), used in Task 4 ✓
- Metrics incl. false-support headline, missed-support, κ, confusion → Task 3 ✓
- Frontier case generator emphasizing near-misses, intended_class hidden from judges → Task 5 ✓
- N-judges-plus-frontier runner, text/json table, review artifact, informational verdict, `--dry-run` → Task 6 ✓
- `P5SpikeConfig` with the exact model strings → Task 5 ✓
- `spikes/` outside pytest; pure modules strict-typed; live modules untested-by-import → Global Constraints + structure ✓
- bmlib path dependency + extras → Task 1 ✓
- Unit tests for agent (fake LLM), metrics, P5Case, scoring → Tasks 1–6 ✓
- Ramp path (Stage 2) needs no code change here — `gold_label` already honored ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; no "similar to Task N".

**Type consistency:** `CitationJudgment(support, reason, pico_match_notes)`, `P5Case(...)`, `ModelScore(...)`, `score_run(cases, judgments, reference_model)`, `make_p5_agent(llm, model, *, temperature)` are used identically across tasks. `reference_label(case, frontier)` consistent in cases.py/scoring.py. Metric signatures `(model, reference)` consistent.
