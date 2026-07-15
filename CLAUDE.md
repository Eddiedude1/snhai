# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo implements an end-to-end pipeline for building and training a Large Language Model.
The goal is to fine-tune a Hugging Face LLM to adjudicate personal loan applications
(APPROVE / REJECT / FLAG_REVIEW) and explain its decision, using the rule set in
`fine_tune_llm_credit_rules.json` as the source of truth for what the model should learn to
apply.

Target deliverables:
- Data prep script(s): turn the credit rules into training examples (tokenization, special
  tokens, padding/truncation, train/val/test split), with rationale documented.
- Training script: fine-tune a chosen pretrained HF model with a full PyTorch/TensorFlow loop
  (forward pass, loss, backprop, optimizer step, validation, checkpointing), with justification
  for base model / optimizer / loss / hyperparameters.
- Evaluation script: metrics on the validation set plus sample generated dialogues with a
  strengths/weaknesses analysis.

**Current state:** each stage is worked in order (Data Preparation → Training → Evaluation),
spec-first: write a lean SRS under `docs/srs/`, then a failing pytest suite traced to that
spec's requirement IDs, then (later) the implementation. Status per stage:
- **Data Preparation**: spec (`docs/srs/data-preparation.md`), test suite
  (`tests/test_data_preparation.py`, 27 tests), and implementation
  (`src/snhai/data_preparation.py`) are done; `uv run pytest tests/test_data_preparation.py` is green (27 passed) and
  `uv run ruff check .` / `uv run ruff format --check .` are clean. Seeded functions
  (`generate_applicant_profiles`, `generate_balanced_profiles`, `generate_edge_case_profiles`,
  `split_dataset`) use a local `random.Random(seed)` instance rather than global `random`
  state (A3.6), enforced by dedicated non-mutation tests. Rule evaluation (`evaluate_profile`)
  is generic/data-driven off each rule's `field`/`operator`/`value` (NFR-DP-2); synthetic
  profile generation samples each field around its rule's threshold with a per-rule fail
  probability keyed off `action_on_fail`. Left alone, this naturally skews decisions toward
  REJECT (~75/15/12 REJECT/FLAG_REVIEW/APPROVE with the real ruleset, since REJECT fires if
  *any* of 8 REJECT-severity rules fails) — `generate_balanced_profiles` stratifies toward a
  configurable `target_label_ratios`, defaulting to a uniform 1/3 baseline (A3.7; open
  question on a better, diversity-weighted default is OQ-3 in the SRS) and is what `main()`
  now uses. Edge-case generation (`generate_edge_case_profiles`) perturbs 2-3
  numeric-threshold rules at once from an all-pass baseline, including exact-threshold cases
  (FR-DP-8). `data_preparation.py` also exposes a `main()` CLI that writes
  `train.jsonl`/`val.jsonl`/`test.jsonl` and `data_card.json` to `--output-dir`, defaulting to
  `WhitespaceTokenizer` but opting into the real base model's tokenizer
  (`load_real_tokenizer`, lazy `transformers` import mirroring Training's/Evaluation's
  pattern) whenever `tokenizer_model_id` is configured (not exercised by the unit tests, which
  inject their own tokenizer double per A3.5). Configuration is externalized to a single
  namespaced `config.json` (sections: `data_preparation`, `training`, `evaluation`) loaded via
  the shared `src/snhai/config.py:load_config(path, stage)` utility (`tests/test_config.py`, 3
  tests); precedence is built-in script defaults < `config.json`'s stage section < explicit CLI
  flags (A3.8, IR-DP-4). The `training` and `evaluation` sections are now populated. OQ-1 (§7
  of the SRS, `max_seq_len`/split-count calibration) is empirically closed:
  `scripts/measure_token_lengths.py` renders the real profile pool with `render_example` and
  tokenizes it with the actual selected base model's tokenizer (`Qwen/Qwen2.5-0.5B-Instruct`,
  per Training's A3.2) via the same `load_real_tokenizer` `main()` uses, not the naive
  `WhitespaceTokenizer` stand-in (which badly undercounts — render_example's `field=value`
  facts have no internal whitespace, so naive `.split()` treats each fact as one token).
  Measured result (`docs/analysis/token_length_measurement.json`): real max token length is 177
  (train/val pool) / 142 (edge-case pool), comfortably under the configured `max_seq_len=256`
  with zero truncation, so 256 was kept unchanged. The script also surfaced that
  `config.json`'s `split_counts["test"]=30` is dead configuration — the actual test file is
  sized by `generate_edge_case_profiles` (14 profiles for the current ruleset), not by
  `split_counts["test"]`, which only dilutes the train/val ratio denominator (actual resolved
  sizes at `n_profiles=400` are train=316/val=56/test=14, not the nominal 340/60/30); this is
  documented in the SRS rather than silently misleading.

  A second, more serious issue surfaced while wiring up a baseline (pre-fine-tuning)
  evaluation: the first generated `data/` used `WhitespaceTokenizer`'s ad-hoc local vocabulary
  for `input_ids`, but Training's `_batches()` feeds `input_ids` straight into the real model
  as token ids without re-tokenizing, and Evaluation's `main()` decodes them with the real
  tokenizer — both assume real-vocabulary-aligned ids, which whitespace-tokenizer ids aren't
  (same small integers, unrelated meaning under Qwen's vocabulary; not a crash, silently wrong
  training/eval). Fixed by adding `config.json`'s `data_preparation.tokenizer_model_id`
  (`Qwen/Qwen2.5-0.5B-Instruct`, matching Training's A3.2) and `load_real_tokenizer()` in
  `data_preparation.py`, so `main()`'s real run now tokenizes with the real, special-token-
  extended base-model tokenizer instead of `WhitespaceTokenizer` — verified by round-tripping a
  generated example's `input_ids` through `tokenizer.decode()` back to the original rendered
  text. `data/` was regenerated accordingly (splits/label distribution unchanged, since only
  the tokenizer changed, not profile generation; `data_card.json`'s `tokenizer_id` is now
  `Qwen/Qwen2.5-0.5B-Instruct` rather than `whitespace`) and committed as the reviewable
  deliverable — the data is small/synthetic/exactly reproducible from the seed (NFR-DP-1,
  verified by re-running and diffing both before and
  after this fix). `transformers` moved from a dev-only to a main runtime dependency as a
  result (`main()`'s real path needs it; the unit test suite still doesn't, since it injects
  its own doubles and never imports `transformers`).
- **Training**: spec (`docs/srs/training.md`), test suite (`tests/test_training.py`, 18
  tests), and implementation (`src/snhai/training.py`) are done; `uv run pytest tests/test_training.py`
  is green (18 passed) and `uv run ruff check .` / `uv run ruff format --check .` are clean.
  Tests exercise fake model/optimizer/tokenizer doubles throughout (duck-typing the relevant
  HF/torch interfaces), so this stage's test suite has no torch/transformers dependency.
  `load_base_model`/optimizer construction default to real `transformers`/`torch` imports, but
  those imports are lazy (inside the default-loader/optimizer functions and `main()`'s
  orchestration path only) rather than top-level — this machine (Intel macOS, Python 3.13) has
  no installable torch wheel at all (PyPI dropped Intel-macOS torch wheels after 2.2.2, which
  caps at cp312), so torch/transformers are deliberately *not* added to `pyproject.toml`; real
  training is expected to run in a separate GPU-enabled environment (Colab, per A3.2's
  rationale). Reproducibility is via a `make_rng(seed)` factory returning a local
  `random.Random`, not a global `set_seed` call (A3.5, mirroring Data Preparation's A3.6). Its
  CLI (`main()`, not exercised by the unit test suite, mirroring Data Preparation's `main()`)
  reads hyperparameters from `config.json`'s `training` section via `src/snhai/config.py:load_config`,
  same precedence pattern as Data Preparation (A3.6/IR-TR-4); that section is now populated
  (learning rate, batch size, epochs, optimizer, weight decay, warmup, eval/checkpoint cadence).

  Fine-tuning strategy is a **full fine-tune** (all parameters trainable, no LoRA/PEFT) — the
  SRS (`docs/srs/training.md` A3.2/A3.3/OQ-1) previously described LoRA, which was a
  documentation/implementation mismatch (`training.py` never had any `peft`/`LoraConfig` code)
  caught before the first real training run and resolved with the user rather than silently
  picked: at this run's scale (316 train examples, batch 8, 3 epochs = 120 total steps), full
  fine-tuning of a 0.5B model fits comfortably in a free-tier Colab T4's memory/time budget, and
  since Qwen2.5-0.5B ties `embed_tokens`/`lm_head` and that embedding table is ~28% of total
  params — which must stay fully trainable regardless, to learn this project's new special
  tokens — LoRA's usual parameter/memory savings would have been far more modest here than on
  larger models. Full fine-tuning was already built and unit-tested (zero new-code risk); LoRA
  is deferred to a planned follow-up comparison run (`docs/srs/training.md` OQ-3, not yet
  implemented — no `peft` dependency exists in `pyproject.toml`).

  A real bug surfaced the first time `align_tokenizer_and_model` actually ran against a real
  model (via Evaluation's baseline-eval path in Colab, not Training itself yet): it resized
  embeddings to `tokenizer.vocab_size`, but real HF tokenizers leave `vocab_size` at the base
  size after `add_special_tokens` — only `len(tokenizer)` reflects the added tokens — so the
  embedding table was never actually grown, and the newly added special-token ids (e.g.
  `<|decision|>`) indexed past its end, crashing with a CUDA device-side assert (embedding
  index out-of-bounds) on first real generation. Fixed by resizing to `len(tokenizer)`
  (`src/snhai/training.py`, A3.3). The unit test suite's `FakeTokenizer` double had (wrongly)
  mutated `vocab_size` itself on `add_special_tokens`, matching neither real tokenizer behavior
  nor exposing the bug; it's now fixed to mirror real semantics (`vocab_size` fixed, `__len__`
  reflects additions), and `tests/test_training.py`'s alignment test updated accordingly —
  `uv run pytest tests/test_training.py` still green (18 passed).

  A second real bug surfaced on the first real Colab training run itself (not just alignment):
  `training_step`/`evaluate` called `model(batch)`, passing the whole batch dict as a single
  positional argument. Real HF causal-LM `forward()` takes `input_ids`/`attention_mask`/`labels`
  as separate keyword arguments, so this crashed immediately (`TypeError: embedding(): argument
  'indices' ... must be Tensor, not dict` — the dict was being passed straight through as
  `input_ids`). Fixed by calling `model(**batch)` in both functions
  (`src/snhai/training.py`). The unit test suite's `FakeModel` double had `__call__(self,
  batch)` — a single positional dict — which matched the buggy call site instead of exposing
  it; fixed to `__call__(self, **kwargs)` to mirror real HF model call semantics, and
  `uv run pytest tests/test_training.py` still green (18 passed) since no test asserts on the
  double's argument-passing convention itself.

  A third real bug was caught by inspection before it could crash the same Colab run:
  `save_checkpoint` serialized the checkpoint payload with `json.dumps`, but a real model's
  `state_dict()` holds `torch.Tensor` values, which `json` cannot serialize — this would have
  thrown at the very first checkpoint (`config.json`'s `checkpoint_every_n_steps=50`). Fixed by
  switching `save_checkpoint`/`load_checkpoint` to `pickle` (`checkpoint.pkl`, not
  `checkpoint.json`), which serializes both real tensor state dicts and the unit test suite's
  plain-dict `FakeModel`/`FakeOptimizer` doubles generically, without this module needing a hard
  `torch` import to do it. `uv run pytest tests/test_training.py` still green (18 passed).

  A fourth real gap, caught while the user was mid-run and asked how to confirm the Colab T4
  was actually being used, was also an SRS gap, not just a code bug: `docs/srs/training.md`'s
  NFR-TR-2 mentioned GPU as a *possible* target ("single consumer GPU, or CPU as fallback") but
  never actually required using one when present, so nothing flagged that neither
  `_default_model_loader` nor `_batches` ever placed anything on a GPU device.
  `AutoModelForCausalLM.from_pretrained` loads onto CPU regardless of GPU availability, and
  `_batches`'s `torch.tensor(...)` calls had no `device=` at all — so the training run in
  progress was almost certainly running entirely on CPU despite the allocated T4, which also
  explains its slowness. NFR-TR-2 was tightened to require GPU execution when available ("silent
  CPU execution alongside an unused, allocated GPU is not acceptable"), and the code fixed to
  match by adding `_resolve_device()` (`cuda` if
  available, else `cpu`), moving the loaded model there in `_default_model_loader`, and
  threading a `device` parameter through `_batches`/`train_model` (inferred from
  `next(model.parameters()).device`, so it always matches wherever the model actually is).
  `main()` now also prints the resolved device and periodic per-step train/val loss, since the
  training loop previously had zero stdout output for its whole duration (only file-logged via
  `log_metrics`), which was indistinguishable from a hang. None of `_batches`/`train_model`/
  `main` are exercised by the unit test suite (real-transformers-only CLI orchestration, per
  this file's existing note), so this needed no test changes; `uv run pytest` still green (66
  passed).

  A fifth, minor real gap surfaced when committing `runs/training/metrics.log` as evidence
  alongside `eval_report.json`: `log_metrics` opens the log file in append mode, but
  `train_model` never truncated it at the start of a fresh (non-resumed) run, so the several
  earlier interrupted Colab attempts (the CPU-bound run before the device-placement fix, plus a
  couple of aborted retries) had all appended their own `step 1`/`step 2`/`step 3` entries onto
  the same `runs/training/metrics.log` before the actual successful 120-step run wrote its own —
  producing a handful of stray duplicate/overlapping leading step numbers with different loss
  values in the raw file. Fixed by truncating `log_path` at the start of `train_model` whenever
  `resume_checkpoint is None` (a genuine resume still appends, continuing the same log). The
  committed `runs/training/metrics.log` was manually cleaned to the single continuous 120-step
  run before this fix landed (its own step-50/step-100 val losses match the console output
  already reported in this file and `REPORT.md`); future fresh runs will produce a clean log
  automatically. `uv run pytest` still green (66 passed), since `train_model` isn't exercised by
  the unit test suite (same real-transformers-only carve-out as above).

  `main()`'s config-loading was later hardened by replacing the manual `stage_config.get(key,
  TrainingConfig.field)` bridging with a typed `TrainingSettings` (`pydantic.BaseModel`,
  `extra="forbid"`) that owns the *entire* `training` section of `config.json` — both
  hyperparameters and the run-level fields (`model_id`/`dataset_dir`/`output_dir`/`seed`/
  `resume_from`) that section also carries. A misspelled or unrecognized key (e.g. a typo'd
  `learning_rate`) now raises a `ValidationError` at startup instead of silently falling back to
  its built-in default, and `optimizer_name` is checked against `OPTIMIZER_REGISTRY` itself via a
  `field_validator` rather than a hardcoded literal list, so it can't drift out of sync with the
  registry. `TrainingConfig` (the dataclass `train_model` actually consumes) and every function
  below it in the module are unchanged — `TrainingSettings` only replaces how `main()` resolves
  `config.json` into argparse defaults — so `tests/test_training.py` needed no edits and stayed
  green (18 passed). This made `pydantic` a genuine top-level runtime dependency, unlike
  `torch`/`transformers`: it installs cleanly on this dev machine (no Intel-Mac wheel gap), so
  its import in `training.py` is a normal top-level import rather than the lazy,
  inside-a-function pattern used for `torch`/`transformers`.
- **Evaluation**: spec (`docs/srs/evaluation.md`), test suite (`tests/test_evaluation.py`, 18
  tests), and implementation (`src/snhai/evaluation.py`) are done; `uv run pytest tests/test_evaluation.py`
  is green (18 passed) and `uv run ruff check .` / `uv run ruff format --check .` are clean.
  Tests use fake model/tokenizer doubles and an injected `random.Random` (no global state)
  throughout, so this stage's test suite has no torch/transformers dependency; `main()`'s
  default loaders import `transformers` lazily, mirroring Training's A3.4 pattern.
  Decision-label parsing relies on the `<|decision|>...<|/decision|>` special-token span
  (A3.3), checked against a fixed `VALID_DECISION_LABELS` set (`APPROVE`/`REJECT`/
  `FLAG_REVIEW`), not any model-specific output format; an unrecognized or missing span is a
  distinct "unparseable" outcome rather than a crash (FR-EV-4/NFR-EV-5). Rule-citation accuracy
  (FR-EV-6) is plain substring matching of driving rule ids against the generated completion,
  so it stays generic across rulesets (NFR-EV-3) rather than hard-coding the real 10 rule ids.
  Since Data Preparation's dataset artifact only persists tokenized `input_ids` (not the raw
  applicant profile or an explicit driving-rule-ids field), `main()`'s orchestration path
  recovers both the eval prompt and the ground-truth driving rule ids by decoding the stored
  `input_ids` back to text and parsing data_preparation.render_example's rendered rationale
  (`"...failed rule(s): <ids>."`) and splitting at `<|decision|>` — a documented, not fully
  test-covered, implication of that dataset contract. Its CLI reads from `config.json`'s
  `evaluation` section via `src/snhai/config.py:load_config`, same precedence pattern as the other two
  stages (A3.6/IR-EV-4); that section is now populated (model dir, dataset dir, split, seed,
  samples-per-label, report path).

  A gap surfaced while preparing to run a real baseline (pre-fine-tuning) evaluation of the
  raw `Qwen/Qwen2.5-0.5B-Instruct` checkpoint in Colab: `main()`'s default loaders/generation
  path was only ever exercised against this stage's fake model/tokenizer doubles, which don't
  reflect two real-transformers constraints. First, `model.generate()` requires a batched
  tensor on the model's device, not the plain `list[int]` `generate_completion` passes in the
  fake-double tests — fixed by having `_default_model_loader` return a `_RealCausalLMAdapter`
  that does the tensor conversion/device placement internally, so `generate_completion`'s
  tested list-in/list-out contract is untouched. Second, a raw base-model checkpoint's
  tokenizer/embeddings don't yet include this project's special tokens
  (`<|decision|>`/`<|/applicant|>`/etc.) that the dataset's `input_ids` were tokenized with —
  fixed by having `main()` call `training.align_tokenizer_and_model` (same function Training's
  `main()` uses) right after loading, using `data_card.json`'s `special_tokens`. This call is a
  no-op for an already-fine-tuned `model_dir` (tokens/embeddings already match, since Training
  saves both via `save_pretrained`), so it's unconditional rather than baseline-only. Neither
  fix touches unit-tested behavior (`uv run pytest tests/test_evaluation.py` still 18 passed);
  both paths remain untested by the fake-double suite since they only run against a real
  transformers/torch install.

  A real parsing bug (not a model-quality issue) surfaced once a real fine-tuned checkpoint's
  evaluation came back 0% accuracy / 100% unparseable despite completions that visibly contained
  well-formed `<|decision|>\nDecision: APPROVE\nRationale: ...\n<|/decision|>` spans —
  `data_preparation.render_example` renders `"Decision: {decision}\nRationale: {rationale}"`
  *inside* the span, but `parse_decision` required the span's entire stripped content to
  exactly equal a bare label (`"APPROVE"`, not `"Decision: APPROVE\nRationale: ..."`), so it
  never matched. The unit test suite's `parse_decision` tests only ever used a bare-label span
  (`<|decision|>REJECT<|/decision|>`), which matched the buggy exact-equality check instead of
  exposing the mismatch against the real rendered format. Fixed by searching the span content
  for any of `VALID_DECISION_LABELS` as a whole word (`_LABEL_PATTERN`, word-boundary-anchored
  so e.g. `REJECTED` doesn't false-positive-match `REJECT`) rather than requiring exact equality
  — this also keeps the existing bare-label unit tests passing unchanged, since a bare label is
  a special case of "label appears as a whole word in the span." `uv run pytest
  tests/test_evaluation.py` still green (18 passed). Confirmed this bug did *not* affect the
  already-committed `baseline_eval_report.json`: the raw pre-fine-tuning model's completions
  never contained a closing `<|/decision|>` tag at all (it just rambled in generic chat style),
  so its "0% accuracy, all unparseable" result was independently correct and doesn't need
  re-running — only the post-fine-tuning evaluation was masked by this bug.

All three stages have specs + test suites, and all three are now implemented and green:
`uv run pytest tests/` passes (66 passed).

The three pipeline stages live in an installable `src/snhai/` package (`src/snhai/config.py`,
`src/snhai/data_preparation.py`, `src/snhai/training.py`, `src/snhai/evaluation.py`, plus an
empty `src/snhai/__init__.py`), built via a `hatchling` `[build-system]` in `pyproject.toml`;
`uv sync` editable-installs it, so `tests/` and the stages import each other as
`from snhai import data_preparation` etc. rather than as loose root-level scripts. Each stage's
CLI is run as `uv run python -m snhai.<stage>` — there are deliberately no `[project.scripts]`
entry points, to keep `pyproject.toml` minimal. `config.json` and
`fine_tune_llm_credit_rules.json` intentionally stay at the repo root: they're data/config
inputs, not package code, and moving them would only add path-updating churn for no benefit.
The old unrelated `main.py` (`uv init` stub) has been removed. One-off analysis tooling (not
part of the installable package or its test suite) lives outside `src/snhai/`: `scripts/` holds
`measure_token_lengths.py`, and its output report is written to `docs/analysis/`.

## Specifications & workflow

- `docs/srs/` holds one lean-IEEE SRS per stage (`data-preparation.md`, `training.md`,
  `evaluation.md`), each with Purpose/Scope, Assumptions & Constraints, Functional
  Requirements (`FR-<STAGE>-#`), Non-Functional Requirements (`NFR-<STAGE>-#`), and Interface
  Requirements (`IR-<STAGE>-#`).
- Each stage's pytest module (`tests/test_<stage>.py`) traces every test back to a requirement
  ID in that stage's docstring, so coverage is greppable, e.g.
  `grep -o 'DP-[0-9]*' tests/test_data_preparation.py`.
- Tests are written before the implementation and are expected to fail (red) until the stage's
  script is written — a collection-time `ModuleNotFoundError` for the not-yet-created module is
  the expected failure mode, not a bug in the test file.
- Fixtures start local to their test file; only promote a fixture to `tests/conftest.py` once a
  second stage's test module actually needs it.
- **Definition of done for a stage's spec+tests work**, before moving to the next stage: tests
  fail for the right reason (missing implementation, not a broken test), `uv run ruff check .`
  and `uv run ruff format --check .` are clean (or their diffs have been applied), and this
  file's "Current state" above reflects what was actually built.

## Environment and commands

This project uses `uv` (see `uv.lock`, `pyproject.toml`) with Python 3.13 (`.python-version`).

```bash
uv sync                                 # install deps + editable-install the snhai package
uv run python -m snhai.data_preparation # run a stage (also: snhai.training, snhai.evaluation)
uv add <package>                        # add a new dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>                   # add a dev-only dependency (e.g. test/lint tooling)
uv run jupyter lab                      # the `dev` group includes jupyter, which pulls in ipykernel transitively
uv run pytest                           # run the full test suite
uv run pytest tests/test_data_preparation.py -q   # run one stage's tests
uv run ruff check .                     # lint
uv run ruff format .                    # auto-format (check with `--check` first)
```

Runtime dependencies are `pydantic`, `transformers` (`transformers` is needed by Data
Preparation's `main()` real-tokenizer path — see the Data Preparation note above — and by
Training's/Evaluation's default model/tokenizer loaders; none of the three stages' unit test
suites import it, since all of them inject their own fake doubles). `pydantic` backs Training's
`TrainingSettings` (see the Training note above) and, unlike `transformers`/`torch`, is imported
top-level rather than lazily, since it installs on every platform this project runs on; it *is*
imported by `tests/test_training.py` transitively (via `import snhai.training`), so it's a real
test-suite dependency, not one of the lazily-imported real-model-only ones. Dev dependencies are
`jupyter`, `pytest`, `ruff`. `ipykernel` was likewise dropped from the runtime dependency
list — it was only ever needed for ad hoc notebook exploration, not by any script in `src/`, and
`jupyter` (a dev dependency) already pulls it in transitively, so an explicit top-level entry was
redundant. `pandas` and `seaborn` were declared as runtime dependencies from
the very first commit but never actually imported anywhere in `src/`/`tests/`/`scripts/` —
leftover from pre-spec-first exploratory notebook work (see the `.ipynb_checkpoints` data-card
note under Key data files) — and were removed, along with the 9 transitive packages
(`matplotlib`, `pillow`, etc.) they alone were pulling in. Actually running Training/Evaluation's
default model-loading paths will additionally require `torch`, which isn't installable on this
machine (see the Training status note above) — that's expected to happen in a separate
GPU-enabled environment (Colab).

## Key data files

- `fine_tune_llm_credit_rules.json` — the source ruleset (`personal_loan_credit_rules`). Each
  rule has: `id`, `name`, `description`, `field` (dotted path like `applicant.credit_score`),
  `operator` (`>=`, `<=`, `in`, `is`, or a field-relative comparison via
  `value_field_multiplier`/`multiplier_value`), `value`, `action_on_fail`
  (`REJECT` | `FLAG_REVIEW`), `severity` (`CRITICAL` | `MAJOR` | `MINOR`), and `group`
  (e.g. `Eligibility`, `Creditworthiness`, `FinancialStability`, `LoanSuitability`). Any data
  generation/labeling logic must evaluate applications against *all* rules and should respect
  the fact that a `REJECT`-severity rule failing should dominate the final decision.
- `.ipynb_checkpoints/data_card-checkpoint.json` — a draft data card from earlier
  exploration (not tracked in git; `.ipynb_checkpoints` is gitignored). It documents an intended
  dataset shape worth reusing as a starting point: 10 rules, whitespace-fallback tokenizer,
  special tokens (`<|begin|>`, `<|end|>`, `<|pad|>`, `<|applicant|>`/`<|/applicant|>`,
  `<|decision|>`/`<|/decision|>`), `max_seq_len` 256, a 340/60/30 train/val/test split, decision
  label distribution (APPROVE/REJECT/FLAG_REVIEW), and a test set deliberately composed of
  held-out multi-rule edge cases (2-3 simultaneous rule perturbations, some at exact numeric
  thresholds) — useful for stress-testing threshold logic.
