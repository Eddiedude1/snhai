# SRS: Data Preparation Stage

| | |
|---|---|
| **Stage** | 1 of 3 — Data Preparation |
| **Status** | Draft |
| **Source** | Project brief, "Data Preparation" task; `fine_tune_llm_credit_rules.json`; `.ipynb_checkpoints/data_card-checkpoint.json` (prior exploration) |
| **Consumes** | `fine_tune_llm_credit_rules.json` |
| **Produces** | Tokenized train/val/test datasets + `data_card.json` (contract consumed by the Training stage, see [training.md](./training.md)) |

## 1. Purpose & Scope

Defines the requirements for the script(s) that turn `fine_tune_llm_credit_rules.json` into
tokenized, split training data for fine-tuning an LLM to adjudicate personal loan applications
(APPROVE / REJECT / FLAG_REVIEW) and explain its decision.

In scope: rule ingestion, synthetic applicant/application example generation, per-rule
evaluation, decision derivation, natural-language example rendering, tokenization, and
train/val/test splitting.

Out of scope: model architecture, training loop, and evaluation metrics (see
[training.md](./training.md), [evaluation.md](./evaluation.md)).

## 2. Definitions

- **Rule** — one entry in `personal_loan_credit_rules.rules`, each with `id`, `name`,
  `description`, `field` (dotted path, e.g. `applicant.credit_score`), an `operator`
  (`>=`, `<=`, `in`, `is`, or a `value_field_multiplier`/`multiplier_value` field-relative
  comparison), a comparison value, `action_on_fail` (`REJECT` | `FLAG_REVIEW`), `severity`
  (`CRITICAL` | `MAJOR` | `MINOR`), and a `group`.
- **Applicant profile** — a synthetic record providing a value for every `field` referenced
  by the ruleset (e.g. `applicant.age`, `loan_application.requested_amount_usd`).
- **Decision** — the single overall label (`APPROVE` | `REJECT` | `FLAG_REVIEW`) derived for
  a profile after evaluating it against *all* rules.
- **Example** — one rendered, tokenized training instance: an applicant profile, its
  per-rule evaluation results, and its derived decision + rationale.

## 3. Assumptions & Constraints

- A3.1: The ruleset (`fine_tune_llm_credit_rules.json`) is the single source of truth for
  correct labeling; no external labeled data is assumed to exist.
- A3.2: The Data Preparation stage runs before a base model is finalized (Training stage,
  §Model Selection). Tokenization must therefore be parameterized by tokenizer choice rather
  than hard-coded to one model's tokenizer, so this stage isn't blocked on / redone after
  model selection.
- A3.3: Per CLAUDE.md, a rule whose `action_on_fail` is `REJECT` failing must dominate the
  final decision regardless of any `FLAG_REVIEW` rules also failing.
- A3.4: `max_seq_len` and split ratios are configurable. `max_seq_len=256` has now been
  empirically confirmed (see §7, OQ-1) against real rendered-example token lengths under the
  base model's own tokenizer — it was carried over from prior exploration as a placeholder,
  but measurement showed it already has ~45% headroom over the longest real example (256 vs.
  a measured max of 177 tokens), so no change was needed. The 340/60/30 split counts remain
  configured as-is; §7 documents what those counts actually resolve to and a scope note on
  `split_counts["test"]`.
- A3.5: The tokenizer is injected as a parameter (any HF `PreTrainedTokenizer`-compatible
  object); a trivial whitespace tokenizer is the default/test double so this stage's tests
  don't require downloading a real model checkpoint (see §7, OQ-2). `main()`'s CLI defaults
  to this whitespace fallback too, but opts into the real base model's tokenizer
  (`load_real_tokenizer`) instead when `tokenizer_model_id` is configured — required, not
  optional, once real training is intended: Training's SRS (`training.md`) has the batching
  logic feed Data Preparation's `input_ids` directly into the model as token ids, without
  re-tokenizing, so those ids must already be aligned to the real base model's vocabulary.
  `config.json`'s `data_preparation.tokenizer_model_id` is set to the same
  `Qwen/Qwen2.5-0.5B-Instruct` chosen in Training's A3.2, and the committed `data/` dataset
  was generated with it (verified: `tokenizer.decode` of a generated example's `input_ids`
  round-trips to the original rendered text — see `docs/analysis/token_length_measurement.json`
  and OQ-1 below for the token-length numbers measured against this same real tokenizer).
- A3.6: Every seeded function (`generate_applicant_profiles`, `generate_edge_case_profiles`,
  `split_dataset`) SHALL derive randomness from a local `random.Random(seed)` instance created
  within the call, and SHALL NOT read or mutate the global `random` module's state. This keeps
  generation calls side-effect-free for callers and prevents cross-test state leakage in the
  unit test suite (NFR-DP-1).
- A3.7: The decision-label balance of generated train/val examples SHALL be a configurable
  `target_label_ratios` parameter (`generate_balanced_profiles`), rather than whatever
  distribution falls out of independent per-rule sampling. The default is a uniform 1/3 per
  label baseline — a deliberately simple starting point beyond the ≥5% floor in NFR-DP-3, not
  a claim that uniform is the optimal target. See OQ-3.
- A3.8: Configuration is externalized to a single `config.json` with one namespaced section
  per pipeline stage (`data_preparation`, `training`, `evaluation`), loaded via the shared
  `config.py:load_config(path, stage)` utility (IR-DP-4) rather than three separate per-stage
  config files or one flat unnamespaced file. This keeps each stage's config independently
  editable — changing a training hyperparameter can't accidentally touch data prep's section —
  while still living in one file a reviewer can read end-to-end. Precedence is layered:
  built-in script defaults < `config.json`'s stage section < explicit CLI flags. A missing
  config file, or a config file missing the requested stage's section, is not an error; it
  just means "use the script's built-in defaults," so this is backward compatible with running
  any stage's script with no config file at all.

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-DP-1 | The system SHALL load `fine_tune_llm_credit_rules.json` and validate every rule against the expected schema (required fields present; `value_field_multiplier`/`multiplier_value` present together or not at all), raising a descriptive error on malformed input. |
| FR-DP-2 | The system SHALL generate synthetic applicant/loan-application profiles supplying a value for every distinct `field` referenced across all rules. |
| FR-DP-3 | The system SHALL evaluate each profile against **every** rule in the ruleset (no short-circuiting on first failure) and record a per-rule pass/fail result. |
| FR-DP-4 | The system SHALL derive one overall decision per profile using this precedence: (a) if any failed rule has `action_on_fail == REJECT`, decision = `REJECT`; else (b) if any failed rule has `action_on_fail == FLAG_REVIEW`, decision = `FLAG_REVIEW`; else (c) decision = `APPROVE`. |
| FR-DP-5 | The system SHALL render each (profile, evaluation result) pair as a natural-language example that states the applicant's relevant facts, the decision, and a rationale naming the specific rule(s) (by `id`) that drove the decision. |
| FR-DP-6 | The system SHALL wrap structural sections of each rendered example in delimiting special tokens (at minimum: sequence begin/end, pad, applicant-block open/close, decision-block open/close). |
| FR-DP-7 | The system SHALL tokenize each rendered example with the configurable, injected tokenizer (A3.5), applying padding or truncation to a configurable fixed `max_seq_len` (A3.4). |
| FR-DP-8 | The system SHALL generate a held-out test set composed of multi-rule edge cases: profiles perturbing 2-3 rules simultaneously, including cases at exact numeric thresholds (e.g. `credit_score == 670`), distinct from the train/val generation logic. |
| FR-DP-9 | The system SHALL split generated examples into non-overlapping train, validation, and test sets and persist each as a separate artifact. |
| FR-DP-10 | The system SHALL emit a `data_card.json` documenting: rule count, `max_seq_len`, tokenizer identifier, special token map, split sizes, and decision label distribution per split. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-DP-1 | **Reproducibility** — Given the same ruleset and a fixed random seed, dataset generation SHALL be deterministic (identical examples, splits, and ordering across runs), and SHALL do so via a local RNG instance without mutating global `random` module state (A3.6). |
| NFR-DP-2 | **Extensibility** — Adding, removing, or modifying a rule in the JSON ruleset SHALL NOT require code changes to the generation pipeline (rule evaluation must be data-driven off the `operator`/`field`/`value` schema, not per-rule-id special-cased logic). |
| NFR-DP-3 | **Label coverage** — The generated train and validation sets SHALL each contain a non-trivial proportion (minimum threshold, e.g. ≥5%) of all three decision labels, to avoid degenerate single-class datasets. |
| NFR-DP-4 | **Documented rationale** — Preprocessing choices (tokenization strategy, special tokens, split ratios, edge-case test set design) SHALL be documented alongside the code, per the project brief's requirement to "discuss your rationale." |
| NFR-DP-5 | **Performance** — Full dataset generation and tokenization SHALL complete on a single CPU without a GPU dependency, in bounded time proportional to configured example count. |

## 6. Interface Requirements

| ID | Requirement |
|---|---|
| IR-DP-1 | **Input** — a file path to a JSON document matching the `personal_loan_credit_rules` schema described in §2. |
| IR-DP-2 | **Output: datasets** — tokenized train/val/test examples persisted in a format directly loadable by the Training stage (e.g. Hugging Face `Dataset` saved to disk, or JSONL of input_ids/attention_mask/labels). |
| IR-DP-3 | **Output: data card** — `data_card.json` is the stable contract read by the Training stage to align tokenizer, special tokens, and vocabulary/embedding resizing; its schema (see §4, FR-DP-10) SHALL NOT change without a corresponding update to [training.md](./training.md). |
| IR-DP-4 | **Configuration** — the CLI SHALL accept an optional `--config` path to a namespaced JSON file (default `config.json`); it reads only this stage's `data_preparation` section (via `config.py:load_config`) to supply CLI defaults, which explicit CLI flags then override (A3.8). |

## 7. Open Questions / Risks

- OQ-1 (resolved — measured, not just carried over): `max_seq_len` and the split counts were
  re-measured against real rendered examples once FR-DP-5 (`render_example`) existed, via
  `scripts/measure_token_lengths.py`; full numbers in
  `docs/analysis/token_length_measurement.json`.
  - **`max_seq_len`**: whitespace-tokenizer counts (the pipeline's default injected tokenizer,
    A3.5) turned out to be a poor proxy for this — `render_example`'s facts line has no
    internal whitespace within a fact (`applicant.credit_score=792`), so naive `.split()`
    collapses each fact into one giant "token" (measured mean ~26 tokens, clearly
    unrepresentative). The real question is what the *base model's* tokenizer produces, since
    that's what Training will actually use — Training's A3.2 has already chosen
    `Qwen/Qwen2.5-0.5B-Instruct`, so the script tokenizes with that model's real tokenizer
    (special tokens registered via `add_special_tokens`, mirroring
    `training.py:align_tokenizer_and_model`). Result at `n_profiles=400`, `seed=42`: the
    balanced train/val pool is min 125 / mean 131.7 / p99 157 / **max 177** tokens; the FR-DP-8
    edge-case pool is min 132 / mean 135 / max 142 tokens. Both are well under the configured
    `max_seq_len=256` (max real example uses 177/256 = 69% of the budget, zero truncation in
    either pool) — note that truncation here would be especially costly, since
    `tokenize_example` truncates by keeping the head and dropping the tail, and
    `render_example` puts the decision + rationale at the very end, so a truncated example
    would silently lose its label. **Decision: keep `max_seq_len=256` unchanged** — it already
    has comfortable headroom, confirmed empirically rather than assumed.
  - **Split counts**: at the configured `split_counts={340, 60, 30}` / `n_profiles=400`, the
    actual resolved sizes are **train=316, val=56** (`split_dataset`'s ratio arithmetic, which
    divides by all three configured counts including `test`, yields fewer than the nominal 340
    and 60; the remaining 28 generated profiles are computed but discarded by
    `main()`'s `train, val, _ = split_dataset(...)`). The **test** file is not sized by
    `split_counts["test"]` at all — it comes entirely from `generate_edge_case_profiles`,
    which deterministically produces **14** profiles for the current 10-rule ruleset (6
    numeric-threshold rules → 6 single-threshold cases + 4 pair-combos + 4 triple-combos, each
    capped at 4 by `main()`'s combo sampling). So `split_counts["test"]=30` is effectively
    unused for sizing — it only dilutes the train/val ratio denominator. NFR-DP-3's ≥5%-per-
    label floor was re-verified against the *actual* sizes (not the nominal 340/60): train is
    APPROVE 33.2% / REJECT 34.2% / FLAG_REVIEW 32.6%, val is 32.1% / 35.7% / 32.1% — both
    comfortably clear the floor. **Decision: no code or config changes to `split_counts`** —
    the resulting sizes already satisfy every enforced requirement; making
    `split_counts["test"]` actually govern test-set sizing would be a design change to
    FR-DP-8/FR-DP-9/`main()` (new tests required) and is out of scope here, but is now
    documented rather than silently misleading.
- OQ-2 (resolved — A3.5): Tokenizer is injected rather than hard-coded, avoiding coupling
  this stage to Training's base-model decision; a whitespace tokenizer is the default/test
  double. Once a base model was chosen (Training's A3.2), `main()` was wired to opt into that
  model's real tokenizer via `tokenizer_model_id` — see A3.5's fuller note on why this is
  required, not just an enhancement, for `input_ids` that Training/Evaluation can actually use.
- OQ-3 (open — A3.7): Independent per-rule sampling naturally yields a skewed decision-label
  distribution (~75% REJECT / ~15% FLAG_REVIEW / ~12% APPROVE with the real ruleset, since
  REJECT fires if *any* of 8 REJECT-severity rules fails, while APPROVE requires all 10 to
  pass). `generate_balanced_profiles` makes the target ratio configurable and defaults it to
  a uniform 1/3 per label. Whether uniform is actually the right target — versus weighting by
  each label's downstream rationale diversity (REJECT can be driven by any of 8 rules alone or
  in combination; APPROVE's rationale is close to invariant, "all rules passed") — is left for
  future exploration rather than decided here.
