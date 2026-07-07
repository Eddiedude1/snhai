# SRS: Evaluation & Analysis Stage

| | |
|---|---|
| **Stage** | 3 of 3 — Evaluation and Analysis |
| **Status** | Draft |
| **Source** | `LLM Coding Challenge for AI Eng.pdf`, §2 "Evaluation and Analysis" |
| **Consumes** | Model + tokenizer directory + training log (see [training.md](./training.md), IR-TR-2); validation/test dataset + `data_card.json` (see [data-preparation.md](./data-preparation.md), IR-DP-2/IR-DP-3) |
| **Produces** | An evaluation report (metrics + sample dialogues) and a written strengths/weaknesses analysis — the final pipeline deliverable, feeding the challenge's optional brief report |

## 1. Purpose & Scope

Defines the requirements for evaluating the fine-tuned model's decision-making performance on
the validation set and for generating/analyzing sample dialogues, per the challenge brief's
"Evaluation and Analysis" task.

In scope: loading the fine-tuned model, generating decision+rationale completions, scoring them
against ground truth, sampling dialogues for qualitative review, and producing a persisted
report.

Out of scope: how the model was trained (see [training.md](./training.md)); how ground-truth
labels/examples were produced (see [data-preparation.md](./data-preparation.md)).

## 2. Definitions

- **Generated decision** — the decision label (`APPROVE` | `REJECT` | `FLAG_REVIEW`) parsed out
  of the model's generated completion for a given applicant-profile prompt.
- **Ground-truth decision** — the decision label attached to a validation/test example by Data
  Preparation's rule evaluation (the `labels` field in IR-DP-2's dataset artifact).
- **Dialogue** — one full generation round: the applicant-profile prompt plus the model's
  generated completion.
- **Unparseable output** — a generated completion with no recognizable
  `<|decision|>...<|/decision|>` span, or content inside it that doesn't match any known label.

## 3. Assumptions & Constraints

- A3.1: Evaluation consumes exactly the model+tokenizer artifact and dataset contract emitted
  by Training and Data Preparation (IR-TR-2, IR-DP-2/IR-DP-3) — it does not retrain the model
  or re-derive ground-truth labels. `--model-dir` MAY instead point at a raw, not-yet-fine-tuned
  base-model checkpoint identifier (e.g. the Hugging Face Hub id Training would otherwise fine-
  tune) to produce a pre-fine-tuning baseline report for comparison; see A3.7.
- A3.2: Data Preparation's rule-evaluation logic (`evaluate_profile`/`derive_decision`) is the
  ground-truth oracle for correctness metrics; Evaluation reuses it rather than re-implementing
  rule logic, consistent with Data Preparation's NFR-DP-2 (rule logic lives in one place).
- A3.3: Decision-label extraction from generated text is done via the special-token span
  (`<|decision|>...<|/decision|>`) recorded in the data card, not by assuming any particular
  model's output format — this keeps evaluation decoupled from which base model Training chose
  (mirrors data-preparation.md A3.2 / training.md A3.2).
- A3.4: The model is loaded from a configurable path rather than hard-coded, and generation is
  exercised in unit tests via a fake/stub model+tokenizer (duck-typing a minimal
  text-generation interface) rather than downloading real weights — mirrors
  data-preparation.md A3.5 and training.md A3.2/A3.4's test-double pattern.
- A3.5: Sampling-related randomness (stochastic decoding, sample-dialogue selection) SHALL use
  a local RNG (`make_rng(seed)`, see training.md A3.5) rather than global `random` state,
  consistent with data-preparation.md A3.6 and training.md A3.5.
- A3.6: Mirroring Data Preparation's A3.8 and training.md A3.6, this stage's CLI SHALL read its
  config from the `evaluation` section of the shared, namespaced `config.json` (via
  `config.py:load_config`), with precedence built-in script defaults < `config.json`'s
  `evaluation` section < explicit CLI flags. The `evaluation` section of `config.json` is a
  stub (`_todo`) until this stage's script exists.
- A3.7: Per A3.1's baseline-checkpoint case, `main()` unconditionally calls
  `training.align_tokenizer_and_model` (the same function Training's `main()` uses) on the
  loaded model/tokenizer, using `data_card.json`'s `special_tokens`, immediately after loading
  and before any decode/generate calls. A raw base-model checkpoint's tokenizer/embeddings don't
  yet include this project's special tokens (`<|decision|>`, `<|/applicant|>`, etc.) that the
  dataset's `input_ids` were tokenized with, so without this the stored ids can't be decoded and
  generation has no way to reference those tokens. The call is a no-op for an already-fine-tuned
  `model_dir` (Training already added the tokens and resized/saved both), so it's applied
  unconditionally rather than branched on which kind of checkpoint was passed. Real
  `model.generate()` also requires a batched tensor on the model's device rather than the plain
  `list[int]` this stage's fake-model unit tests use (FR-EV-3/A3.4); `_default_model_loader`
  wraps the real model so this conversion happens there, keeping `generate_completion`'s tested
  list-in/list-out contract unchanged. Neither adjustment is covered by the unit test suite,
  which never exercises the default (non-injected) loaders — consistent with training.md A3.2's
  same carve-out for its own default loaders, both being real-transformers/torch-only code paths
  that can't run without a torch install (see CLAUDE.md's environment notes).

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-EV-1 | The system SHALL load the fine-tuned model and tokenizer from the directory produced by the Training stage, or (A3.1/A3.7) from a raw base-model checkpoint identifier for a pre-fine-tuning baseline report, aligning the tokenizer/embeddings to the dataset's special tokens in the latter case. |
| FR-EV-2 | The system SHALL load the validation (and/or held-out test) dataset produced by the Data Preparation stage without re-deriving tokenization or ground-truth labels. |
| FR-EV-3 | The system SHALL generate a decision+rationale completion for each validation/test example from its applicant-profile prompt. |
| FR-EV-4 | The system SHALL parse the generated completion's decision label from its `<|decision|>...<|/decision|>` span, treating an unparseable or missing span as a distinct outcome rather than silently defaulting to a label. |
| FR-EV-5 | The system SHALL compute classification metrics (accuracy, per-label precision/recall/F1, confusion matrix) comparing parsed generated decisions against ground-truth decisions. |
| FR-EV-6 | For examples whose ground-truth decision is `REJECT` or `FLAG_REVIEW`, the system SHALL compute a rule-citation-accuracy metric: whether the generated rationale cites at least one of the rule id(s) that actually drove the ground-truth decision (A3.2). |
| FR-EV-7 | The system SHALL sample a configurable number of dialogues (prompt + generated completion) for qualitative inspection, drawn to include at least one correct and one incorrect prediction per decision label where available. |
| FR-EV-8 | The system SHALL persist evaluation results (metrics + sampled dialogues) to a report artifact rather than only printing to stdout. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-EV-1 | **Reproducibility** — sample-dialogue selection and any stochastic decoding SHALL be seeded via a local RNG (A3.5) so repeated runs with the same seed/config select identical samples. |
| NFR-EV-2 | **Documented analysis** — evaluation results SHALL be accompanied by a written strengths/weaknesses analysis referencing specific metrics and sample dialogues, per the challenge brief. |
| NFR-EV-3 | **Extensibility** — metrics and rule-citation-accuracy computation SHALL be data-driven off the ruleset/data card, not hard-coded to the specific 10 rules currently in `fine_tune_llm_credit_rules.json` (mirrors NFR-DP-2). |
| NFR-EV-4 | **Bounded orchestration overhead** — metric aggregation and report generation (excluding actual model inference cost) SHALL scale linearly with dataset size, not reprocess it more than a small constant number of passes. |
| NFR-EV-5 | **Robustness** — unparseable model outputs SHALL NOT crash the evaluation run; they SHALL be counted and reported as a distinct category (ties to FR-EV-4). |

## 6. Interface Requirements

| ID | Requirement |
|---|---|
| IR-EV-1 | **Input: model** — the model+tokenizer directory and training log produced per Training's IR-TR-2, or (A3.1/A3.7) a raw base-model checkpoint identifier for a pre-fine-tuning baseline report. |
| IR-EV-2 | **Input: data** — the validation/test dataset artifact and `data_card.json` produced per Data Preparation's IR-DP-2/IR-DP-3, used for prompts, ground-truth labels, and special-token parsing. |
| IR-EV-3 | **Output** — an evaluation report artifact (metrics + sample dialogues) in a durable, inspectable format (e.g. JSON/markdown), suitable for inclusion in the challenge's optional brief report. |
| IR-EV-4 | **Configuration** — the CLI SHALL accept an optional `--config` path to the shared namespaced JSON file (default `config.json`); it reads only this stage's `evaluation` section (via `config.py:load_config`) to supply CLI defaults, which explicit CLI flags then override (A3.6; mirrors data-preparation.md IR-DP-4). |

## 7. Open Questions / Risks

- OQ-1: The exact metric emphasis (overall accuracy vs. macro-averaged precision/recall/F1,
  given Data Preparation's label-imbalance allowance in NFR-DP-3) is deferred until real
  generation output is available to inspect (post FR-EV-3/FR-EV-4).
- OQ-2: FR-EV-6's rule-citation matching strategy (exact rule-id substring match vs. a more
  lenient/fuzzy match) is deferred pending how explicit Data Preparation's rendered rationale
  (data-preparation.md FR-DP-5) turns out to be in practice.
