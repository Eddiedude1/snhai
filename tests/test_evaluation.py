"""Failing unit tests for the Evaluation stage.

Spec: docs/srs/evaluation.md. Each test's docstring names the FR-EV-#/NFR-EV-#/IR-EV-#
requirement(s) it covers. These tests import `evaluation`, which does not exist yet — they are
expected to fail (collection error) until that stage is implemented.

These tests avoid real torch/transformers models and generation: model loading and text
generation are exercised through injectable loader callables and small fake
model/tokenizer objects that duck-type a minimal encode/generate/decode interface — the same
test-double pattern used in tests/test_data_preparation.py (A3.5) and tests/test_training.py
(A3.2/A3.4).
"""

import json
import random
import time
from pathlib import Path

import pytest

from snhai import evaluation as ev

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeEvalTokenizer:
    """Minimal encode/decode double; real tokenization is Data Preparation's concern (A3.3)."""

    def encode(self, text: str) -> list[int]:
        return [len(text)]

    def decode(self, ids: list[int]) -> str:
        return f"decoded:{ids}"


class FakeGenModel:
    def __init__(self, output_ids: list[int]):
        self.output_ids = output_ids
        self.last_input_ids = None

    def generate(self, input_ids: list[int]) -> list[int]:
        self.last_input_ids = input_ids
        return self.output_ids


# --- FR-EV-1 / IR-EV-1: loading the fine-tuned model + tokenizer -------------------------------


def test_load_finetuned_model_delegates_to_injected_loaders(tmp_path):
    """FR-EV-1, IR-EV-1: the model/tokenizer are loaded from the Training stage's output
    directory via injectable loader callables, so this is unit-testable without a real model."""
    calls = {}

    def fake_model_loader(model_dir):
        calls["model_dir"] = model_dir
        return "FAKE_MODEL"

    def fake_tokenizer_loader(model_dir):
        return "FAKE_TOKENIZER"

    model, tokenizer = ev.load_finetuned_model(
        tmp_path, model_loader=fake_model_loader, tokenizer_loader=fake_tokenizer_loader
    )
    assert calls["model_dir"] == tmp_path
    assert model == "FAKE_MODEL"
    assert tokenizer == "FAKE_TOKENIZER"


# --- FR-EV-2 / IR-EV-2: loading Data Preparation's validation/test dataset ----------------------


def test_load_eval_dataset_reads_requested_split(tmp_path):
    """FR-EV-2, IR-EV-2: loads the requested split's examples produced by Data Preparation,
    without re-deriving tokenization or ground-truth labels."""
    val_example = {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": "APPROVE"}
    (tmp_path / "val.jsonl").write_text(json.dumps(val_example))
    loaded = ev.load_eval_dataset(tmp_path, split="val")
    assert loaded == [val_example]


def test_load_eval_dataset_missing_directory_raises():
    """IR-EV-2: a nonexistent dataset directory raises FileNotFoundError, not a silent empty result."""
    with pytest.raises(FileNotFoundError):
        ev.load_eval_dataset(REPO_ROOT / "does_not_exist_dataset_dir", split="val")


# --- FR-EV-3: generating a completion for a prompt -----------------------------------------------


def test_generate_completion_encodes_generates_and_decodes():
    """FR-EV-3: generation composes tokenizer.encode -> model.generate -> tokenizer.decode."""
    tokenizer = FakeEvalTokenizer()
    model = FakeGenModel(output_ids=[99])
    completion = ev.generate_completion(model, tokenizer, prompt="hello")
    assert model.last_input_ids == tokenizer.encode("hello")
    assert completion == "decoded:[99]"


# --- FR-EV-4: parsing the decision label out of a completion --------------------------------------


def test_parse_decision_extracts_valid_label_from_tagged_completion():
    """FR-EV-4, A3.3: the decision is parsed from the <|decision|>...<|/decision|> span."""
    completion = "some rationale <|decision|>REJECT<|/decision|> trailing text"
    assert ev.parse_decision(completion) == "REJECT"


def test_parse_decision_returns_none_when_tags_missing():
    """FR-EV-4, NFR-EV-5: a completion with no decision span is an unparseable outcome, not a crash."""
    assert ev.parse_decision("the model rambled without ever deciding") is None


def test_parse_decision_returns_none_when_tag_content_unrecognized():
    """FR-EV-4, NFR-EV-5: content inside the span that isn't a known label is unparseable."""
    assert ev.parse_decision("<|decision|>MAYBE<|/decision|>") is None


# --- FR-EV-5: classification metrics ---------------------------------------------------------------


def test_compute_classification_metrics_matches_hand_computed_values():
    """FR-EV-5: accuracy, per-label precision/recall/F1, and confusion matrix match a
    hand-computed example, including one unparseable (None) prediction."""
    ground_truth = ["APPROVE", "APPROVE", "REJECT", "REJECT", "FLAG_REVIEW"]
    predictions = ["APPROVE", "REJECT", "REJECT", "APPROVE", None]
    metrics = ev.compute_classification_metrics(predictions, ground_truth)

    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["unparseable_count"] == 1

    approve = metrics["per_label"]["APPROVE"]
    assert approve["precision"] == pytest.approx(0.5)
    assert approve["recall"] == pytest.approx(0.5)
    assert approve["f1"] == pytest.approx(0.5)

    reject = metrics["per_label"]["REJECT"]
    assert reject["precision"] == pytest.approx(0.5)
    assert reject["recall"] == pytest.approx(0.5)
    assert reject["f1"] == pytest.approx(0.5)

    flag_review = metrics["per_label"]["FLAG_REVIEW"]
    assert flag_review["precision"] == pytest.approx(0.0)
    assert flag_review["recall"] == pytest.approx(0.0)
    assert flag_review["f1"] == pytest.approx(0.0)


def test_compute_classification_metrics_handles_all_unparseable_without_crashing():
    """NFR-EV-5: even if every prediction is unparseable, metrics computation completes
    without raising, reporting zero accuracy and the correct unparseable count."""
    metrics = ev.compute_classification_metrics([None, None], ["APPROVE", "REJECT"])
    assert metrics["unparseable_count"] == 2
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_compute_classification_metrics_completes_quickly_on_large_input():
    """NFR-EV-4: metrics aggregation scales linearly and doesn't reprocess the dataset
    pathologically, even for a few thousand examples."""
    ground_truth = ["APPROVE", "REJECT", "FLAG_REVIEW"] * 1000
    predictions = ["APPROVE", "REJECT", "FLAG_REVIEW"] * 1000
    start = time.monotonic()
    ev.compute_classification_metrics(predictions, ground_truth)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0


# --- FR-EV-6 / NFR-EV-3: rule-citation accuracy -----------------------------------------------------


def test_rule_citation_accuracy_computes_expected_ratio():
    """FR-EV-6: measures whether the rationale cites at least one actual driving rule id, over
    REJECT/FLAG_REVIEW examples only."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion="Approved, all checks passed.",
        ),
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["RULE-AGE-001"],
            generated_completion="Rejected due to RULE-AGE-001 (minimum age).",
        ),
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["RULE-CREDIT-001"],
            generated_completion="Rejected for unrelated reasons.",
        ),
        ev.GenerationResult(
            ground_truth_decision="FLAG_REVIEW",
            driving_rule_ids=["RULE-EMPLOY-002", "RULE-LOANAMT-001"],
            generated_completion="Flagged per RULE-LOANAMT-001.",
        ),
    ]
    assert ev.rule_citation_accuracy(results) == pytest.approx(2 / 3)


def test_rule_citation_accuracy_is_generic_not_tied_to_specific_rule_ids():
    """NFR-EV-3: citation matching is plain string matching against whatever rule ids appear,
    not a hard-coded list of the real 10 rule ids."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["RULE-CUSTOM-999"],
            generated_completion="Rejected per RULE-CUSTOM-999.",
        ),
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["RULE-CUSTOM-998"],
            generated_completion="Rejected for unrelated reasons.",
        ),
    ]
    assert ev.rule_citation_accuracy(results) == pytest.approx(0.5)


def test_rule_citation_accuracy_returns_none_when_no_reject_or_flag_examples():
    """FR-EV-6: with no REJECT/FLAG_REVIEW examples the metric is undefined (None), not a
    division-by-zero crash."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion="Approved.",
        )
    ]
    assert ev.rule_citation_accuracy(results) is None


# --- FR-EV-7 / NFR-EV-1: sampling dialogues for qualitative review --------------------------------


def test_sample_dialogues_includes_correct_and_incorrect_per_label():
    """FR-EV-7: sampling draws at least one correct and one incorrect prediction per label,
    where available."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion="approve-correct",
            predicted_decision="APPROVE",
        ),
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion="approve-incorrect",
            predicted_decision="REJECT",
        ),
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["R1"],
            generated_completion="reject-correct",
            predicted_decision="REJECT",
        ),
        ev.GenerationResult(
            ground_truth_decision="REJECT",
            driving_rule_ids=["R1"],
            generated_completion="reject-incorrect",
            predicted_decision="APPROVE",
        ),
    ]
    sampled = ev.sample_dialogues(results, n_per_label=1, rng=random.Random(0))
    sampled_texts = {r.generated_completion for r in sampled}
    assert sampled_texts == {
        "approve-correct",
        "approve-incorrect",
        "reject-correct",
        "reject-incorrect",
    }


def test_sample_dialogues_is_reproducible_given_same_rng_seed():
    """NFR-EV-1, A3.5: sampling is deterministic given the same seeded rng."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion=f"approve-correct-{i}",
            predicted_decision="APPROVE",
        )
        for i in range(5)
    ]
    first = ev.sample_dialogues(results, n_per_label=1, rng=random.Random(42))
    second = ev.sample_dialogues(results, n_per_label=1, rng=random.Random(42))
    assert [r.generated_completion for r in first] == [
        r.generated_completion for r in second
    ]


def test_sample_dialogues_does_not_mutate_global_random_state():
    """NFR-EV-1, A3.5: sampling SHALL use the passed-in rng, not the global `random` module."""
    results = [
        ev.GenerationResult(
            ground_truth_decision="APPROVE",
            driving_rule_ids=[],
            generated_completion=f"approve-correct-{i}",
            predicted_decision="APPROVE",
        )
        for i in range(5)
    ]
    state_before = random.getstate()
    ev.sample_dialogues(results, n_per_label=1, rng=random.Random(42))
    assert random.getstate() == state_before


# --- FR-EV-8 / IR-EV-3: persisting the evaluation report ------------------------------------------


def test_save_and_load_report_round_trips(tmp_path):
    """FR-EV-8, IR-EV-3: the evaluation report (metrics + sample dialogues) round-trips through
    a durable, inspectable artifact."""
    report = {
        "metrics": {"accuracy": 0.8},
        "sample_dialogues": [{"prompt": "...", "completion": "..."}],
    }
    report_path = tmp_path / "eval_report.json"
    ev.save_report(report, report_path)
    loaded = ev.load_report(report_path)
    assert loaded == report


# --- NFR-EV-2: documented strengths/weaknesses analysis (documentation presence, not code) -------


def test_evaluation_spec_documents_analysis_requirement():
    """NFR-EV-2: the SRS records the requirement for a written strengths/weaknesses analysis."""
    spec = (REPO_ROOT / "docs" / "srs" / "evaluation.md").read_text()
    assert "NFR-EV-2" in spec
