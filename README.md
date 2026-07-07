# SNH-AI Technical Coding Challenge: Building and Training an LLM

An end-to-end pipeline that fine-tunes a Hugging Face LLM to adjudicate personal loan
applications (`APPROVE` / `REJECT` / `FLAG_REVIEW`) and explain its decision, using the rule set
in `fine_tune_llm_credit_rules.json` as the source of truth for what the model should learn to
apply. Built for the challenge described in `LLM Coding Challenge for AI Eng.pdf`.

The pipeline has three stages, each with its own spec, test suite, and script:

| Stage | Spec | Script | Tests |
|---|---|---|---|
| 1. Data Preparation | [docs/srs/data-preparation.md](docs/srs/data-preparation.md) | `src/snhai/data_preparation.py` | `tests/test_data_preparation.py` (27) |
| 2. Model Selection & Training | [docs/srs/training.md](docs/srs/training.md) | `src/snhai/training.py` | `tests/test_training.py` (18) |
| 3. Evaluation & Analysis | [docs/srs/evaluation.md](docs/srs/evaluation.md) | `src/snhai/evaluation.py` | `tests/test_evaluation.py` (18) |

Each stage's output is the next stage's input: Data Preparation produces tokenized
train/val/test splits + `data_card.json`; Training produces a fine-tuned model + tokenizer
directory; Evaluation scores that model and writes a report. See `CLAUDE.md` for a detailed,
running account of design decisions and real bugs found/fixed while building this out, and
`REPORT.md` for the results/analysis writeup.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13 (`.python-version`).

```bash
uv sync                                 # installs deps + editable-installs the snhai package
uv run pytest                           # run the full test suite (66 tests, no GPU/torch needed)
uv run ruff check .                     # lint
uv run ruff format --check .            # format check
```

Runtime dependencies: `pandas`, `seaborn`, `ipykernel`, `transformers`. Dev dependencies:
`jupyter`, `pytest`, `ruff`. **None of the three stages' unit test suites require `torch`** —
they're tested against fake model/tokenizer/optimizer doubles that duck-type the relevant
Hugging Face/PyTorch interfaces, so `uv sync && uv run pytest` works on any machine, including
one where `torch` isn't installable at all (this was developed on an Intel Mac, where PyPI no
longer ships a `torch` wheel).

Actually *running* Training or Evaluation's real model-loading paths does need `torch` and,
practically, a GPU — see below.

## Running a stage locally (Data Preparation only)

Data Preparation has no GPU dependency and can be run directly:

```bash
uv run python -m snhai.data_preparation
```

This reads `fine_tune_llm_credit_rules.json`, generates synthetic applicant profiles, evaluates
them against every rule, tokenizes the rendered examples with the real base model's tokenizer
(`Qwen/Qwen2.5-0.5B-Instruct`, extended with this project's special tokens), and writes
`train.jsonl` / `val.jsonl` / `test.jsonl` / `data_card.json` to `data/` (already committed and
reproducible from the fixed seed — re-running should not change these files).

## Running Training and Evaluation (needs a GPU)

`training.py`'s and `evaluation.py`'s default model loaders use real `transformers`/`torch`,
and full fine-tuning of even a 0.5B model on CPU is impractically slow. Both scripts were run in
Google Colab against a free-tier T4 GPU:

```bash
# In a Colab notebook, after cloning this repo and cd'ing into it:
pip install -q -e . --no-deps && pip install -q "transformers>=5.13.0"
# --no-deps avoids upgrading Colab's own pinned pandas/ipykernel, which this repo's
# pyproject.toml doesn't actually need at runtime for these two stages.

python -m snhai.training      # writes runs/training/final_model + runs/training/metrics.log
python -m snhai.evaluation    # writes runs/evaluation/eval_report.json
```

Both scripts print progress (resolved device, per-step train/val loss for training; nothing
long-running for evaluation) and read all hyperparameters from `config.json` — no flags are
required for a default run. Every CLI also accepts `--config <path>` plus per-field overrides
(e.g. `--learning-rate`, `--model-dir`) that take precedence over `config.json`, which in turn
takes precedence over the script's built-in defaults.

`training.py` also accepts `--resume-from <checkpoint-dir>`; `evaluation.py` also accepts
`--model-dir`/`--split`/`--n-per-label`/etc. Run either script with `-h` for the full flag list.

### Producing a pre-fine-tuning baseline

`evaluation.py --model-dir Qwen/Qwen2.5-0.5B-Instruct` (or setting `evaluation.model_dir` in
`config.json` to the same) evaluates the *raw*, not-yet-fine-tuned base checkpoint instead of a
`runs/training/final_model` directory — used to produce
`runs/evaluation/baseline_eval_report.json`, the "before" half of the before/after comparison in
`REPORT.md`.

## Configuration

All three stages read defaults from `config.json`'s matching top-level section
(`data_preparation` / `training` / `evaluation`) via `src/snhai/config.py:load_config`.
Precedence: built-in script defaults < `config.json`'s section < explicit CLI flags. This is the
single place to change the base model, hyperparameters, dataset size, or file paths without
touching code.

## Key files

- `fine_tune_llm_credit_rules.json` — the source ruleset. Any data generation/labeling logic
  evaluates applications against *all* rules; a failing `REJECT`-severity rule dominates the
  final decision.
- `data/` — the committed, reproducible dataset artifact (train/val/test splits + data card).
- `runs/evaluation/baseline_eval_report.json` — pre-fine-tuning baseline evaluation report.
- `runs/evaluation/eval_report.json` — post-fine-tuning evaluation report.
- `runs/training/metrics.log` — per-step train/val loss from the training run that produced the
  model behind `eval_report.json`.
- `docs/srs/` — one lean SRS per stage; `docs/analysis/` — supporting empirical analysis (e.g.
  token-length measurement used to size `max_seq_len`).
- `REPORT.md` — approach, results, and strengths/weaknesses analysis.

Note: the fine-tuned model checkpoint itself (`runs/training/final_model/`) is **not** committed
to this repo — at ~1-2GB it exceeds GitHub's file-size limits, and it's cheaply reproducible from
the committed dataset, `config.json`'s hyperparameters, and the fixed seed. The evaluation
reports and metrics log are the durable evidence of what it produced.
