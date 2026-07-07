"""Training stage: fine-tunes a Hugging Face causal-language-model checkpoint on the
tokenized dataset produced by Data Preparation, via an explicit training loop (forward pass,
loss, backprop, optimizer step, validation, checkpointing).

Spec: docs/srs/training.md. Function names/signatures mirror the FR-TR-#/NFR-TR-#/IR-TR-#
requirements exercised by tests/test_training.py. Model/tokenizer loading (FR-TR-1) and
optimizer construction (NFR-TR-5) are injectable, with default implementations that import
torch/transformers lazily (only when actually invoked, e.g. from `main()`). This keeps the
orchestration logic unit-testable against fake model/optimizer/tokenizer doubles without a
torch/transformers install, since actual training runs in a separate GPU-enabled environment
(A3.2 targets a free-tier Colab T4, not this dev machine) rather than this repo's local venv.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from snhai.config import load_config

# --- FR-TR-1: base model loading by configurable identifier ----------------------------------


def _default_model_loader(model_id: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id)


def _default_tokenizer_loader(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load_base_model(
    model_id: str,
    model_loader: Callable[[str], Any] = _default_model_loader,
    tokenizer_loader: Callable[[str], Any] = _default_tokenizer_loader,
) -> tuple[Any, Any]:
    model = model_loader(model_id)
    tokenizer = tokenizer_loader(model_id)
    return model, tokenizer


# --- FR-TR-2 / A3.3: tokenizer + embedding alignment ------------------------------------------


def align_tokenizer_and_model(model, tokenizer, special_tokens: dict) -> int:
    added = tokenizer.add_special_tokens(special_tokens)
    # len(tokenizer), not tokenizer.vocab_size: real HF tokenizers leave vocab_size at the base
    # size after add_special_tokens and only reflect added tokens in __len__, so resizing to
    # vocab_size is a no-op that leaves newly added token ids out of the embedding table's range.
    model.resize_token_embeddings(len(tokenizer))
    return added


# --- FR-TR-3 / IR-TR-1: loading Data Preparation's dataset artifact ---------------------------


def _load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_training_data(dataset_dir: str | Path) -> tuple[list[dict], list[dict]]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    train = _load_jsonl(dataset_dir / "train.jsonl")
    val = _load_jsonl(dataset_dir / "val.jsonl")
    return train, val


# --- FR-TR-8: configurable hyperparameters ----------------------------------------------------


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    batch_size: int = 8
    max_epochs: int = 3
    optimizer_name: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 0
    eval_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 50


# --- NFR-TR-5: config-driven optimizer selection, not per-model branching --------------------


def _adamw_constructor(params, lr: float, weight_decay: float):
    from torch.optim import AdamW

    return AdamW(params, lr=lr, weight_decay=weight_decay)


def _sgd_constructor(params, lr: float, weight_decay: float):
    from torch.optim import SGD

    return SGD(params, lr=lr, weight_decay=weight_decay)


OPTIMIZER_REGISTRY: dict[str, Callable[..., Any]] = {
    "adamw": _adamw_constructor,
    "sgd": _sgd_constructor,
}


def get_optimizer_constructor(name: str) -> Callable[..., Any]:
    try:
        return OPTIMIZER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer {name!r}; available: {sorted(OPTIMIZER_REGISTRY)}"
        ) from exc


# --- FR-TR-4: forward pass, loss, backprop, optimizer step ------------------------------------


def training_step(model, batch, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    return loss.item()


# --- FR-TR-5: validation loss evaluation -------------------------------------------------------


def evaluate(model, batches: list) -> float:
    model.eval()
    losses = [model(**batch).loss.item() for batch in batches]
    return sum(losses) / len(losses)


# --- FR-TR-6 / FR-TR-7 / IR-TR-3: checkpointing and resumption --------------------------------


@dataclass
class Checkpoint:
    step: int
    epoch: int
    model_state: dict
    optimizer_state: dict


def save_checkpoint(model, optimizer, step: int, epoch: int, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    # pickle, not json: a real model/optimizer state_dict() holds torch.Tensor values, which
    # json.dumps cannot serialize. pickle handles both real tensors and the fake doubles' plain
    # dicts generically, without this module needing a hard torch import to do it.
    (path / "checkpoint.pkl").write_bytes(pickle.dumps(payload))


def load_checkpoint(path: str | Path) -> Checkpoint:
    checkpoint_file = Path(path) / "checkpoint.pkl"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = pickle.loads(checkpoint_file.read_bytes())
    return Checkpoint(
        step=payload["step"],
        epoch=payload["epoch"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
    )


def best_checkpoint_step(history: dict[int, float]) -> int:
    return min(history, key=history.get)


# --- FR-TR-9: metrics logging -------------------------------------------------------------------


def log_metrics(
    log_path: str | Path, step: int, train_loss: float, val_loss: float | None
) -> None:
    record = {"step": step, "train_loss": train_loss, "val_loss": val_loss}
    with Path(log_path).open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_metrics_log(log_path: str | Path) -> list[dict]:
    return _load_jsonl(Path(log_path))


# --- FR-TR-10 / IR-TR-2: persisting the fine-tuned model + tokenizer --------------------------


def save_model_and_tokenizer(model, tokenizer, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


# --- NFR-TR-1 / A3.5: reproducibility via a local RNG, not global state -----------------------


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


# --- CLI orchestration (not exercised by the unit test suite) ---------------------------------


def _batches(examples: list[dict], batch_size: int, rng: random.Random):
    import torch

    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start : start + batch_size]]
        input_ids = torch.tensor([e["input_ids"] for e in chunk])
        attention_mask = torch.tensor([e["attention_mask"] for e in chunk])
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def train_model(
    model,
    tokenizer,
    train_examples: list[dict],
    val_examples: list[dict],
    config: TrainingConfig,
    checkpoint_dir: str | Path,
    log_path: str | Path,
    final_model_dir: str | Path,
    rng: random.Random,
    resume_checkpoint: str | Path | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = get_optimizer_constructor(config.optimizer_name)(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    step = 0
    start_epoch = 0
    if resume_checkpoint is not None:
        restored = load_checkpoint(resume_checkpoint)
        model.load_state_dict(restored.model_state)
        optimizer.load_state_dict(restored.optimizer_state)
        step = restored.step
        start_epoch = restored.epoch

    val_loss_by_step: dict[int, float] = {}
    for epoch in range(start_epoch, config.max_epochs):
        for batch in _batches(train_examples, config.batch_size, rng):
            train_loss = training_step(model, batch, optimizer)
            step += 1

            val_loss = None
            if step % config.eval_every_n_steps == 0:
                val_batches = list(_batches(val_examples, config.batch_size, rng))
                val_loss = evaluate(model, val_batches)
                val_loss_by_step[step] = val_loss

            log_metrics(log_path, step=step, train_loss=train_loss, val_loss=val_loss)

            if step % config.checkpoint_every_n_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch,
                    path=checkpoint_dir / f"checkpoint-{step}",
                )

    if val_loss_by_step:
        best_step = best_checkpoint_step(val_loss_by_step)
        best = load_checkpoint(checkpoint_dir / f"checkpoint-{best_step}")
        model.load_state_dict(best.model_state)

    save_model_and_tokenizer(model, tokenizer, final_model_dir)


def main(argv: list[str] | None = None) -> None:
    # Phase 1: find --config (if any) before the rest of argparse needs its values as defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.json"))
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    stage_config = load_config(pre_args.config, "training")

    # Phase 2: full parser, defaults layered as config.json < CLI flags (A3.6, IR-TR-4).
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument(
        "--model-id",
        type=str,
        default=stage_config.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(stage_config.get("dataset_dir", "data")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(stage_config.get("output_dir", "runs/training")),
    )
    parser.add_argument("--seed", type=int, default=stage_config.get("seed", 42))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=stage_config.get("learning_rate", TrainingConfig.learning_rate),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=stage_config.get("batch_size", TrainingConfig.batch_size),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=stage_config.get("max_epochs", TrainingConfig.max_epochs),
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default=stage_config.get("optimizer_name", TrainingConfig.optimizer_name),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=stage_config.get("weight_decay", TrainingConfig.weight_decay),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=stage_config.get("warmup_steps", TrainingConfig.warmup_steps),
    )
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=stage_config.get(
            "eval_every_n_steps", TrainingConfig.eval_every_n_steps
        ),
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=stage_config.get(
            "checkpoint_every_n_steps", TrainingConfig.checkpoint_every_n_steps
        ),
    )
    parser.add_argument(
        "--resume-from", type=Path, default=stage_config.get("resume_from")
    )
    args = parser.parse_args(remaining_argv)

    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        optimizer_name=args.optimizer_name,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_every_n_steps=args.eval_every_n_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
    )

    rng = make_rng(args.seed)
    train_examples, val_examples = load_training_data(args.dataset_dir)
    data_card = json.loads((args.dataset_dir / "data_card.json").read_text())

    model, tokenizer = load_base_model(args.model_id)
    align_tokenizer_and_model(model, tokenizer, data_card["special_tokens"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        model,
        tokenizer,
        train_examples,
        val_examples,
        config,
        checkpoint_dir=args.output_dir / "checkpoints",
        log_path=args.output_dir / "metrics.log",
        final_model_dir=args.output_dir / "final_model",
        rng=rng,
        resume_checkpoint=args.resume_from,
    )


if __name__ == "__main__":
    main()
