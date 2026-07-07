# Software Requirements Specifications

Lean-IEEE-style SRS documents for the three stages of this LLM fine-tuning pipeline. Each stage
is worked spec-first: write the SRS, then a failing pytest suite traced to its requirement IDs,
then (later) the implementation.

| Stage | Spec | Tests | Status |
|---|---|---|---|
| 1. Data Preparation | [data-preparation.md](./data-preparation.md) | `tests/test_data_preparation.py` (27 tests) | Implemented; `uv run pytest tests/test_data_preparation.py` green |
| 2. Model Selection & Training | [training.md](./training.md) | `tests/test_training.py` (18 tests) | Implemented; real Colab training run complete (`runs/training`) |
| 3. Evaluation & Analysis | [evaluation.md](./evaluation.md) | `tests/test_evaluation.py` (18 tests) | Implemented; baseline + post-fine-tuning reports in `runs/evaluation/` |

Each stage's dataset/model artifact is the next stage's input: Data Preparation's tokenized
datasets + `data_card.json` feed Training; Training's saved model + tokenizer directory feeds
Evaluation. See each spec's header table for its exact Consumes/Produces contract.

## Shared conventions

- **Section set** — each SRS uses the same lean subset of IEEE 29148/830 sections: Purpose &
  Scope, Definitions, Assumptions & Constraints, Functional Requirements, Non-Functional
  Requirements, Interface Requirements, and Open Questions/Risks. Personas, use-cases, and
  full documentation-requirements sections are omitted as overhead not worth it for a solo
  project.
- **Requirement IDs** — `FR-<STAGE>-#` (functional), `NFR-<STAGE>-#` (non-functional),
  `IR-<STAGE>-#` (interface), where `<STAGE>` is `DP`, `TR`, or `EV`. Assumptions are numbered
  `A3.#` within each spec's §3.
- **Traceability** — every test's docstring names the requirement ID(s) it covers, so coverage
  is greppable, e.g. `grep -o 'DP-[0-9]*' tests/test_data_preparation.py`.
- **Model/tokenizer decoupling** — Data Preparation and Training both defer the concrete base
  model/tokenizer choice (data-preparation.md A3.2, training.md A3.2), so specs and tests stay
  valid regardless of which Hugging Face model is eventually selected.
- **Local RNG, not global state** — every seeded function takes/returns a local
  `random.Random(seed)` instance rather than mutating global `random` state (data-preparation.md
  A3.6, training.md A3.5, evaluation.md A3.5), verified by dedicated non-mutation tests in each
  suite.
- **Test doubles over real models** — unit tests use fake model/tokenizer/optimizer objects
  duck-typing the relevant Hugging Face/PyTorch interfaces, so the full suite runs fast with no
  network access or GPU required. Real-model integration testing is a separate, later concern.

## Definition of done (per stage, spec + tests)

Before moving to the next stage: tests fail for the right reason (missing implementation, not
a broken test), `uv run ruff check .` and `uv run ruff format --check .` are clean, and
`CLAUDE.md`'s "Current state" reflects what was actually built.
