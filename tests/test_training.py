"""Failing unit tests for the Training stage.

Spec: docs/srs/training.md. Each test's docstring names the FR-TR-#/NFR-TR-#/IR-TR-#
requirement(s) it covers. These tests import `training`, which does not exist yet — they are
expected to fail (collection error) until that stage is implemented.

These tests deliberately avoid real torch/transformers models: they exercise training.py's
orchestration logic (steps, checkpointing, logging, config, model loading) against small fake
model/optimizer/tokenizer objects that duck-type the relevant HF/torch interfaces (`__call__`,
`.state_dict()`, `.save_pretrained()`, etc.) — the same test-double pattern used for the
tokenizer in tests/test_data_preparation.py (data-preparation.md A3.5). FR-TR-1 (loading a real
pretrained checkpoint) is exercised through injectable loader callables rather than a real
network/model download, for the same reason.
"""

import json
import random
import time
from pathlib import Path

import pytest

from snhai import training as tr

REPO_ROOT = Path(__file__).resolve().parent.parent

SPECIAL_TOKENS = {
    "bos_token": "<|begin|>",
    "eos_token": "<|end|>",
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|applicant|>",
        "<|/applicant|>",
        "<|decision|>",
        "<|/decision|>",
    ],
}


class FakeLoss:
    def __init__(self, value: float):
        self.value = value
        self.backward_called = False

    def backward(self):
        self.backward_called = True

    def item(self) -> float:
        return self.value


class FakeModelOutput:
    def __init__(self, loss: "FakeLoss"):
        self.loss = loss


class FakeModel:
    """Duck-types the bits of a HF model used by the training loop."""

    def __init__(self, loss_value: float = 1.0, loss_values: list[float] | None = None):
        self.loss_value = loss_value
        self._loss_values = list(loss_values) if loss_values is not None else None
        self._call_index = 0
        self.train_mode = None
        self.last_loss = None
        self.embedding_size = 100
        self.weight = "untrained"

    def __call__(self, batch):
        if self._loss_values is not None:
            value = self._loss_values[self._call_index]
            self._call_index += 1
        else:
            value = self.loss_value
        self.last_loss = FakeLoss(value)
        return FakeModelOutput(self.last_loss)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def resize_token_embeddings(self, new_size: int):
        self.embedding_size = new_size

    def state_dict(self):
        return {"weight": self.weight, "embedding_size": self.embedding_size}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.marker").write_text("saved")


class FakeOptimizer:
    def __init__(self):
        self.step_calls = 0
        self.zero_grad_calls = 0
        self.momentum = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1

    def state_dict(self):
        return {"momentum": self.momentum}


class FakeTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.added_tokens: list[str] = []

    def add_special_tokens(self, special_tokens: dict) -> int:
        new_tokens = []
        for value in special_tokens.values():
            values = value if isinstance(value, list) else [value]
            for token in values:
                if token not in self.added_tokens and token not in new_tokens:
                    new_tokens.append(token)
        self.added_tokens.extend(new_tokens)
        self.vocab_size += len(new_tokens)
        return len(new_tokens)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.marker").write_text("saved")


@pytest.fixture
def fake_optimizer() -> FakeOptimizer:
    return FakeOptimizer()


# --- FR-TR-1: base model loading by configurable identifier -----------------------------------


def test_load_base_model_delegates_to_injected_loaders():
    """FR-TR-1: the base model/tokenizer are loaded by a configurable identifier via injectable
    loader callables, so this is unit-testable without a network call or a real download."""
    calls = {}

    def fake_model_loader(model_id):
        calls["model_id"] = model_id
        return FakeModel()

    def fake_tokenizer_loader(model_id):
        return FakeTokenizer()

    model, tokenizer = tr.load_base_model(
        "sshleifer/tiny-gpt2",
        model_loader=fake_model_loader,
        tokenizer_loader=fake_tokenizer_loader,
    )
    assert calls["model_id"] == "sshleifer/tiny-gpt2"
    assert isinstance(model, FakeModel)
    assert isinstance(tokenizer, FakeTokenizer)


# --- FR-TR-2 / A3.3: tokenizer + embedding alignment -------------------------------------------


def test_align_tokenizer_and_model_adds_special_tokens_and_resizes_embeddings():
    """FR-TR-2, A3.3: special tokens are added to the tokenizer and the model's embeddings
    resized to match, before training begins."""
    tokenizer = FakeTokenizer(vocab_size=100)
    model = FakeModel()
    model.embedding_size = 100
    tr.align_tokenizer_and_model(model, tokenizer, SPECIAL_TOKENS)
    assert tokenizer.vocab_size == 107  # 3 single tokens + 4 additional_special_tokens
    assert model.embedding_size == tokenizer.vocab_size


# --- FR-TR-3 / IR-TR-1: loading Data Preparation's dataset artifact ----------------------------


def test_load_training_data_reads_train_and_val_jsonl(tmp_path):
    """FR-TR-3, IR-TR-1: loads the tokenized train/val examples produced by Data Preparation,
    without re-deriving tokenization logic."""
    train_examples = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": "APPROVE"}
    ]
    val_examples = [{"input_ids": [3, 4], "attention_mask": [1, 1], "labels": "REJECT"}]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(e) for e in train_examples)
    )
    (tmp_path / "val.jsonl").write_text("\n".join(json.dumps(e) for e in val_examples))
    loaded_train, loaded_val = tr.load_training_data(tmp_path)
    assert loaded_train == train_examples
    assert loaded_val == val_examples


def test_load_training_data_missing_directory_raises():
    """IR-TR-1: a nonexistent dataset directory raises FileNotFoundError, not a silent empty result."""
    with pytest.raises(FileNotFoundError):
        tr.load_training_data(REPO_ROOT / "does_not_exist_dataset_dir")


# --- FR-TR-8: configurable hyperparameters ------------------------------------------------------


def test_training_config_holds_configurable_hyperparameters():
    """FR-TR-8: hyperparameters are configurable constructor inputs, not hard-coded constants."""
    config = tr.TrainingConfig(
        learning_rate=5e-5,
        batch_size=8,
        max_epochs=3,
        optimizer_name="adamw",
        weight_decay=0.01,
        warmup_steps=10,
    )
    assert config.learning_rate == 5e-5
    assert config.batch_size == 8
    assert config.max_epochs == 3
    assert config.optimizer_name == "adamw"
    assert config.weight_decay == 0.01
    assert config.warmup_steps == 10


def test_training_config_has_usable_defaults():
    """FR-TR-8: a default config is constructible without specifying every hyperparameter."""
    config = tr.TrainingConfig()
    assert config.learning_rate > 0
    assert config.batch_size > 0
    assert config.max_epochs > 0


# --- NFR-TR-5: config-driven optimizer selection, not per-model branching -----------------------


def test_optimizer_registry_supports_lookup_by_name():
    """NFR-TR-5: optimizer construction is config-driven via a name registry, not hard-coded
    per-model branching in the training loop."""
    assert "adamw" in tr.OPTIMIZER_REGISTRY
    constructor = tr.get_optimizer_constructor("adamw")
    assert callable(constructor)


def test_optimizer_registry_rejects_unknown_name():
    """NFR-TR-5: an unrecognized optimizer name fails loudly instead of silently defaulting."""
    with pytest.raises(ValueError):
        tr.get_optimizer_constructor("not_a_real_optimizer")


# --- FR-TR-4: forward pass, loss, backprop, optimizer step ---------------------------------------


def test_training_step_runs_forward_loss_backward_and_optimizer_step(fake_optimizer):
    """FR-TR-4: one training step performs forward pass, loss calculation, backprop, and an
    optimizer step."""
    model = FakeModel(loss_value=2.5)
    batch = {"input_ids": [[1, 2, 3]]}
    loss_value = tr.training_step(model, batch, fake_optimizer)
    assert loss_value == 2.5
    assert model.last_loss.backward_called
    assert fake_optimizer.step_calls == 1
    assert fake_optimizer.zero_grad_calls == 1
    assert model.train_mode is True


# --- FR-TR-5: validation loss evaluation ---------------------------------------------------------


def test_evaluate_averages_loss_over_validation_batches_without_updating_weights():
    """FR-TR-5: validation runs forward-only over each batch and reports the average loss,
    switching the model into eval mode."""
    model = FakeModel(loss_values=[1.0, 2.0, 3.0])
    avg_loss = tr.evaluate(model, [{}, {}, {}])
    assert avg_loss == pytest.approx(2.0)
    assert model.train_mode is False


# --- FR-TR-6 / FR-TR-7 / IR-TR-3: checkpointing and resumption -----------------------------------


def test_checkpoint_round_trips_model_optimizer_step_and_epoch(
    tmp_path, fake_optimizer
):
    """FR-TR-6, FR-TR-7, IR-TR-3: a saved checkpoint restores model/optimizer state and the
    step/epoch counters needed to resume."""
    model = FakeModel()
    model.weight = "trained-weights"
    fake_optimizer.momentum = 7
    ckpt_path = tmp_path / "checkpoint-100"
    tr.save_checkpoint(model, fake_optimizer, step=100, epoch=2, path=ckpt_path)
    restored = tr.load_checkpoint(ckpt_path)
    assert restored.step == 100
    assert restored.epoch == 2
    assert restored.model_state == model.state_dict()
    assert restored.optimizer_state == fake_optimizer.state_dict()


def test_load_checkpoint_missing_path_raises():
    """NFR-TR-3, FR-TR-7: resuming from a nonexistent checkpoint raises FileNotFoundError
    rather than silently starting fresh."""
    with pytest.raises(FileNotFoundError):
        tr.load_checkpoint(REPO_ROOT / "does_not_exist_checkpoint")


def test_best_checkpoint_step_selects_minimum_validation_loss():
    """FR-TR-6: the checkpoint retained as best is the one with the lowest validation loss."""
    history = {50: 0.9, 100: 0.4, 150: 0.6}
    assert tr.best_checkpoint_step(history) == 100


# --- FR-TR-9: metrics logging ---------------------------------------------------------------------


def test_log_metrics_records_and_reads_back_train_and_val_loss(tmp_path):
    """FR-TR-9: per-step train loss and per-evaluation val loss are logged for post-hoc inspection."""
    log_path = tmp_path / "metrics.log"
    tr.log_metrics(log_path, step=1, train_loss=1.5, val_loss=None)
    tr.log_metrics(log_path, step=2, train_loss=1.2, val_loss=1.1)
    records = tr.read_metrics_log(log_path)
    assert records[0] == {"step": 1, "train_loss": 1.5, "val_loss": None}
    assert records[1] == {"step": 2, "train_loss": 1.2, "val_loss": 1.1}


# --- FR-TR-10 / IR-TR-2: persisting the fine-tuned model + tokenizer -------------------------------


def test_save_model_and_tokenizer_writes_a_self_contained_directory(tmp_path):
    """FR-TR-10, IR-TR-2: the fine-tuned model and tokenizer are saved together in a directory
    loadable by the Evaluation stage."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    out_dir = tmp_path / "final_model"
    tr.save_model_and_tokenizer(model, tokenizer, out_dir)
    assert (out_dir / "model.marker").exists()
    assert (out_dir / "tokenizer.marker").exists()


# --- NFR-TR-1 / A3.5: reproducibility via a local RNG, not global state ------------------------


def test_make_rng_produces_reproducible_sequence():
    """NFR-TR-1, A3.5: the same seed produces an RNG yielding an identical draw sequence."""
    first_sequence = [tr.make_rng(42).random() for _ in range(5)]
    second_sequence = [tr.make_rng(42).random() for _ in range(5)]
    assert first_sequence == second_sequence


def test_make_rng_does_not_mutate_global_random_state():
    """NFR-TR-1, A3.5: make_rng SHALL return a local `random.Random` instance and SHALL NOT
    read or mutate the global `random` module's state, so it can't leak into other tests."""
    state_before = random.getstate()
    rng = tr.make_rng(42)
    rng.random()
    assert random.getstate() == state_before


# --- NFR-TR-2: bounded orchestration overhead -------------------------------------------------------
# Real-model resource boundedness (can the chosen base model train in bounded wall-clock time on
# commodity hardware) is a model-selection/rationale concern documented per NFR-TR-4 once a base
# model is chosen (docs/srs/training.md, OQ-1) — not something a fake-model unit test can verify.
# This test only guards against the orchestration code itself adding pathological overhead.


def test_training_step_overhead_is_negligible_with_fakes(fake_optimizer):
    """NFR-TR-2: the training-step orchestration itself adds no pathological overhead beyond
    the (fake) model call."""
    model = FakeModel(loss_value=1.0)
    start = time.monotonic()
    for _ in range(500):
        tr.training_step(model, {}, fake_optimizer)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
