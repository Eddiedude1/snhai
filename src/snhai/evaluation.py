"""Evaluation stage: generates decision+rationale completions from the fine-tuned model on
Data Preparation's validation/test set, scores them against ground truth, samples dialogues
for qualitative review, and persists a report.

Spec: docs/srs/evaluation.md. Function names/signatures mirror the FR-EV-#/NFR-EV-#/IR-EV-#
requirements exercised by tests/test_evaluation.py. Model/tokenizer loading (FR-EV-1) mirrors
training.py's `load_base_model` injectable-loader pattern (A3.4): default loaders import
transformers lazily so this module's test suite has no torch/transformers dependency.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from snhai.config import load_config
from snhai.data_preparation import DECISION_CLOSE, DECISION_OPEN

# --- A3.3: decision labels are a closed, project-wide enum (unlike rule ids, which NFR-EV-3
# keeps generic) --------------------------------------------------------------------------

VALID_DECISION_LABELS = {"APPROVE", "REJECT", "FLAG_REVIEW"}

_DECISION_SPAN_PATTERN = re.compile(
    re.escape(DECISION_OPEN) + r"(.*?)" + re.escape(DECISION_CLOSE), re.DOTALL
)
_DRIVING_RULES_PATTERN = re.compile(r"failed rule\(s\):\s*(.*?)\.")


# --- FR-EV-1 / IR-EV-1: loading the fine-tuned model + tokenizer ---------------------------


def _default_model_loader(model_dir):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir)


def _default_tokenizer_loader(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def load_finetuned_model(
    model_dir: str | Path,
    model_loader: Callable[[Any], Any] = _default_model_loader,
    tokenizer_loader: Callable[[Any], Any] = _default_tokenizer_loader,
) -> tuple[Any, Any]:
    model = model_loader(model_dir)
    tokenizer = tokenizer_loader(model_dir)
    return model, tokenizer


# --- FR-EV-2 / IR-EV-2: loading Data Preparation's validation/test dataset -----------------


def load_eval_dataset(dataset_dir: str | Path, split: str) -> list[dict]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    examples = []
    with (dataset_dir / f"{split}.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# --- FR-EV-3: generating a completion for a prompt -----------------------------------------


def generate_completion(model, tokenizer, prompt: str) -> str:
    input_ids = tokenizer.encode(prompt)
    output_ids = model.generate(input_ids)
    return tokenizer.decode(output_ids)


# --- FR-EV-4 / A3.3 / NFR-EV-5: parsing the decision label out of a completion -------------


def parse_decision(completion: str) -> str | None:
    match = _DECISION_SPAN_PATTERN.search(completion)
    if not match:
        return None
    label = match.group(1).strip()
    return label if label in VALID_DECISION_LABELS else None


# --- FR-EV-5 / NFR-EV-4 / NFR-EV-5: classification metrics ---------------------------------


def compute_classification_metrics(
    predictions: list[str | None], ground_truth: list[str]
) -> dict:
    n = len(ground_truth)
    unparseable_count = sum(1 for p in predictions if p is None)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / n if n else 0.0

    labels = sorted(set(ground_truth) | {p for p in predictions if p is not None})
    tp = dict.fromkeys(labels, 0)
    fp = dict.fromkeys(labels, 0)
    fn = dict.fromkeys(labels, 0)
    confusion: dict[str, dict[str, int]] = {label: {} for label in labels}

    for p, g in zip(predictions, ground_truth):
        pred_key = p if p is not None else "UNPARSEABLE"
        confusion[g][pred_key] = confusion[g].get(pred_key, 0) + 1
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            if p is not None:
                fp[p] += 1

    per_label = {}
    for label in labels:
        predicted_count = tp[label] + fp[label]
        actual_count = tp[label] + fn[label]
        precision = tp[label] / predicted_count if predicted_count else 0.0
        recall = tp[label] / actual_count if actual_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "accuracy": accuracy,
        "unparseable_count": unparseable_count,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


# --- FR-EV-6 / NFR-EV-3: rule-citation accuracy --------------------------------------------


@dataclass
class GenerationResult:
    """One dialogue: an applicant-profile prompt, the model's completion, and enough
    ground-truth context (decision + driving rule ids, A3.2) to score it."""

    ground_truth_decision: str
    driving_rule_ids: list[str]
    generated_completion: str
    predicted_decision: str | None = None
    prompt: str = ""


def rule_citation_accuracy(results: list[GenerationResult]) -> float | None:
    relevant = [
        r for r in results if r.ground_truth_decision in ("REJECT", "FLAG_REVIEW")
    ]
    if not relevant:
        return None
    cited = sum(
        1
        for r in relevant
        if any(rule_id in r.generated_completion for rule_id in r.driving_rule_ids)
    )
    return cited / len(relevant)


# --- FR-EV-7 / NFR-EV-1: sampling dialogues for qualitative review -------------------------


def _sample_up_to(items: list, k: int, rng: random.Random) -> list:
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def sample_dialogues(
    results: list[GenerationResult], n_per_label: int, rng: random.Random
) -> list[GenerationResult]:
    labels = sorted({r.ground_truth_decision for r in results})
    sampled: list[GenerationResult] = []
    for label in labels:
        label_results = [r for r in results if r.ground_truth_decision == label]
        correct = [r for r in label_results if r.predicted_decision == label]
        incorrect = [r for r in label_results if r.predicted_decision != label]
        sampled.extend(_sample_up_to(correct, n_per_label, rng))
        sampled.extend(_sample_up_to(incorrect, n_per_label, rng))
    return sampled


# --- FR-EV-8 / IR-EV-3: persisting the evaluation report -----------------------------------


def save_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# --- CLI orchestration (not exercised by the unit test suite) ------------------------------


def _extract_driving_rule_ids(full_text: str) -> list[str]:
    """Ground-truth driving rule ids are baked into the training text's rationale (see
    data_preparation.render_example) rather than persisted as a separate dataset field."""
    match = _DRIVING_RULES_PATTERN.search(full_text)
    if not match:
        return []
    ids_part = match.group(1).strip()
    return [rule_id.strip() for rule_id in ids_part.split(",") if rule_id.strip()]


def _extract_prompt(full_text: str) -> str:
    idx = full_text.find(DECISION_OPEN)
    return full_text[: idx + len(DECISION_OPEN)] if idx != -1 else full_text


def main(argv: list[str] | None = None) -> None:
    # Phase 1: find --config (if any) before the rest of argparse needs its values as defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.json"))
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    stage_config = load_config(pre_args.config, "evaluation")

    # Phase 2: full parser, defaults layered as config.json < CLI flags (A3.6, IR-EV-4).
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(stage_config.get("model_dir", "runs/training/final_model")),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(stage_config.get("dataset_dir", "data")),
    )
    parser.add_argument("--split", type=str, default=stage_config.get("split", "test"))
    parser.add_argument("--seed", type=int, default=stage_config.get("seed", 42))
    parser.add_argument(
        "--n-per-label", type=int, default=stage_config.get("n_per_label", 2)
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            stage_config.get("report_path", "runs/evaluation/eval_report.json")
        ),
    )
    args = parser.parse_args(remaining_argv)

    rng = random.Random(args.seed)
    model, tokenizer = load_finetuned_model(args.model_dir)
    examples = load_eval_dataset(args.dataset_dir, split=args.split)

    results = []
    for example in examples:
        full_text = tokenizer.decode(example["input_ids"])
        prompt = _extract_prompt(full_text)
        completion = generate_completion(model, tokenizer, prompt)
        results.append(
            GenerationResult(
                ground_truth_decision=example["labels"],
                driving_rule_ids=_extract_driving_rule_ids(full_text),
                generated_completion=completion,
                predicted_decision=parse_decision(completion),
                prompt=prompt,
            )
        )

    predictions = [r.predicted_decision for r in results]
    ground_truth = [r.ground_truth_decision for r in results]
    metrics = compute_classification_metrics(predictions, ground_truth)
    metrics["rule_citation_accuracy"] = rule_citation_accuracy(results)

    sampled = sample_dialogues(results, n_per_label=args.n_per_label, rng=rng)

    report = {
        "metrics": metrics,
        "sample_dialogues": [
            {
                "prompt": r.prompt,
                "completion": r.generated_completion,
                "ground_truth_decision": r.ground_truth_decision,
                "predicted_decision": r.predicted_decision,
            }
            for r in sampled
        ],
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, args.report_path)


if __name__ == "__main__":
    main()
