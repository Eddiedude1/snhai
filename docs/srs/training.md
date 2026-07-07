# SRS: Model Selection & Training Stage

| | |
|---|---|
| **Stage** | 2 of 3 — Model Selection and Training |
| **Status** | Draft |
| **Source** | `LLM Coding Challenge for AI Eng.pdf`, §2 "Model Selection and Training" |
| **Consumes** | Tokenized train/val datasets + `data_card.json` (see [data-preparation.md](./data-preparation.md), IR-DP-2/IR-DP-3) |
| **Produces** | Fine-tuned model + tokenizer directory + training log/metrics (contract consumed by the Evaluation stage, see [evaluation.md](./evaluation.md)) |

## 1. Purpose & Scope

Defines the requirements for selecting a pretrained Hugging Face base model and fine-tuning it
on the dataset produced by the Data Preparation stage, via an explicit training loop (forward
pass, loss, backpropagation, optimizer step, validation, checkpointing).

In scope: base model loading, tokenizer/embedding alignment with the data card's special
tokens, the training loop itself, validation, checkpointing/resumption, and hyperparameter
configuration.

Out of scope: how training examples are generated or tokenized (see
[data-preparation.md](./data-preparation.md)); evaluation metrics and dialogue generation (see
[evaluation.md](./evaluation.md)).

## 2. Definitions

- **Base model** — the pretrained Hugging Face causal-language-model checkpoint used as the
  fine-tuning starting point.
- **Checkpoint** — a saved snapshot of model, optimizer, and scheduler state (plus step/epoch
  counters) sufficient to resume training.
- **Training run** — one execution of the training loop over a configured number of
  epochs/steps against the prepared dataset.

## 3. Assumptions & Constraints

- A3.1: This stage consumes exactly the dataset contract emitted by Data Preparation
  (tokenized datasets + `data_card.json`, IR-DP-2/IR-DP-3) — it does not re-derive or duplicate
  tokenization logic.
- A3.2: The base model identifier and fine-tuning strategy remain configurable rather than
  hard-coded (FR-TR-1, NFR-TR-5), so the requirements below are still stated model-agnostically.
  The concrete choice is **`Qwen/Qwen2.5-0.5B-Instruct`, fine-tuned with LoRA** (full rationale:
  `model_selection.md`) — selected for its Apache 2.0 license, ~0.5B size (fits a free-tier
  Colab T4 without quantization), strong existing structured-I/O/JSON behavior, and
  proportionality to a 10-rule single-domain task (see §7, OQ-1, resolved).
- A3.3: The base model's tokenizer SHALL be extended with the special-token set recorded in
  the data card, and the model's token embeddings resized accordingly, before training begins
  — otherwise special tokens (e.g. `<|decision|>`) would map to the tokenizer's default unknown
  token and be unlearnable. Because Qwen2.5-0.5B ties input and output embeddings, and LoRA
  freezes the base model by default, resizing alone is not sufficient: `embed_tokens` and
  `lm_head` SHALL be added to the LoRA config's `modules_to_save` (fully fine-tuned, not
  LoRA-adapted) so the newly added token rows actually receive gradient updates, rather than
  remaining at their random initialization. `align_tokenizer_and_model` resizes to
  `len(tokenizer)`, not `tokenizer.vocab_size`: real HF tokenizers leave `vocab_size` at the
  base size after `add_special_tokens` and only reflect added tokens via `__len__`, so resizing
  to `vocab_size` would silently leave the embedding table too small and the newly added
  token ids out of range — surfaced as a CUDA device-side assert (embedding index
  out-of-bounds) the first time this was run against a real model in Colab, since the unit
  test suite's `FakeTokenizer` originally (incorrectly) mutated `vocab_size` itself and so
  didn't catch the discrepancy.
- A3.4: PyTorch is assumed as the training framework (the more idiomatic path through Hugging
  Face Transformers); TensorFlow is not pursued unless a specific reason emerges to prefer it.
- A3.5: Reproducibility SHALL be provided by a local RNG instance created via a `make_rng(seed)`
  factory and passed explicitly to whatever needs it (e.g. data-order shuffling), rather than a
  global `set_seed`-style call that mutates process-wide random state — mirroring Data
  Preparation's A3.6. Seeding of framework-level global state (torch/CUDA) needed for full
  end-to-end determinism is out of scope for unit-level testing and is addressed at
  implementation time, not asserted by this spec.
- A3.6: Mirroring Data Preparation's A3.8, this stage's CLI SHALL read hyperparameters from the
  `training` section of the shared, namespaced `config.json` (via `config.py:load_config`),
  with precedence built-in script defaults < `config.json`'s `training` section < explicit CLI
  flags. The `training` section of `config.json` is a stub (`_todo`) until this stage's script
  exists; populating it (learning rate, batch size, epochs, optimizer, etc. — FR-TR-8) is part
  of this stage's implementation work, not a separate config-file task.

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-TR-1 | The system SHALL load a pretrained causal-language-model checkpoint from Hugging Face Transformers by a configurable model identifier. |
| FR-TR-2 | The system SHALL extend the base model's tokenizer with the special tokens recorded in Data Preparation's `data_card.json` and resize the model's token embeddings accordingly before training. |
| FR-TR-3 | The system SHALL load the tokenized train/validation datasets produced by the Data Preparation stage without re-deriving tokenization logic. |
| FR-TR-4 | The system SHALL implement a training loop performing, per step: forward pass, loss computation, backpropagation, and an optimizer step. |
| FR-TR-5 | The system SHALL run validation-set loss evaluation at a configurable cadence (e.g. every N steps or once per epoch) during training. |
| FR-TR-6 | The system SHALL checkpoint model, optimizer, and scheduler state at a configurable cadence, and SHALL retain the checkpoint with the best observed validation loss. |
| FR-TR-7 | The system SHALL support resuming training from a saved checkpoint, restoring model, optimizer, scheduler, and step/epoch counters. |
| FR-TR-8 | The system SHALL expose training hyperparameters (learning rate, batch size, epochs/max steps, optimizer type, weight decay, warmup schedule) as configurable inputs rather than hard-coded constants. |
| FR-TR-9 | The system SHALL log per-step or per-epoch training loss and per-evaluation validation loss for post-hoc inspection. |
| FR-TR-10 | The system SHALL persist the final (or best) fine-tuned model together with its tokenizer in a self-contained directory loadable by the Evaluation stage. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-TR-1 | **Reproducibility** — training SHALL be seeded (data ordering/shuffling, any stochastic init) via a local RNG instance (A3.5) so that, given the same seed/config/data, results are repeatable modulo hardware/floating-point nondeterminism, without mutating global `random` module state. |
| NFR-TR-2 | **Resource boundedness** — the chosen base model and configuration SHALL be able to complete at least one full epoch over the prepared dataset on commodity hardware (single consumer GPU, or CPU as fallback) in bounded wall-clock time; parameter-efficient fine-tuning (e.g. LoRA) is an acceptable way to satisfy this. |
| NFR-TR-3 | **Fault tolerance** — an interrupted training run SHALL be resumable from the last checkpoint with no loss beyond the interval since that checkpoint (ties to FR-TR-6/FR-TR-7). |
| NFR-TR-4 | **Documented rationale** — the choice of base model, optimizer, loss function, and hyperparameters SHALL be documented with justification, per the challenge brief. |
| NFR-TR-5 | **Extensibility** — swapping the base model identifier or optimizer type SHALL NOT require changes to the core training-loop logic (config-driven, not hard-coded per-model branching). |

## 6. Interface Requirements

| ID | Requirement |
|---|---|
| IR-TR-1 | **Input** — the tokenized dataset artifact and `data_card.json` produced per Data Preparation's IR-DP-2/IR-DP-3. |
| IR-TR-2 | **Output** — a saved model+tokenizer directory (e.g. Hugging Face `save_pretrained` layout) plus a training log/metrics artifact; this is the contract consumed by the Evaluation stage. |
| IR-TR-3 | **Checkpoint format** — checkpoints SHALL be self-describing enough (step/epoch, optimizer state, config) to resume without external bookkeeping. |
| IR-TR-4 | **Configuration** — the CLI SHALL accept an optional `--config` path to the shared namespaced JSON file (default `config.json`); it reads only this stage's `training` section (via `config.py:load_config`) to supply CLI defaults, which explicit CLI flags then override (A3.6; mirrors data-preparation.md IR-DP-4). |

## 7. Open Questions / Risks

- OQ-1 (resolved — A3.2): base model is `Qwen/Qwen2.5-0.5B-Instruct`, fine-tuned via LoRA
  with `embed_tokens`/`lm_head` in `modules_to_save` to keep new special tokens learnable
  despite tied embeddings (A3.3). Optimizer/loss/hyperparameter rationale (NFR-TR-4) is not
  yet documented — still open.
- OQ-2: PyTorch vs. TensorFlow — assumed PyTorch (A3.4); revisit only if a specific reason to
  prefer TensorFlow emerges.
