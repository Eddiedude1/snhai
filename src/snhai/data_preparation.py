"""Data Preparation stage: turns `fine_tune_llm_credit_rules.json` into tokenized,
split training data for fine-tuning an LLM to adjudicate personal loan applications.

Spec: docs/srs/data-preparation.md. Function names/signatures mirror the FR-DP-#/NFR-DP-#
requirements exercised by tests/test_data_preparation.py.

Rationale (NFR-DP-4):
- Rule evaluation (`evaluate_profile`) is entirely data-driven off each rule's `field` /
  `operator` / `value` (or `value_field_multiplier`/`multiplier_value`) — no rule id is ever
  special-cased, so the ruleset can be edited without touching this module (NFR-DP-2).
- Synthetic profile generation samples each field around its own rule's threshold, with a
  per-field "should this rule fail" coin flip whose probability is keyed off the rule's
  `action_on_fail` (REJECT rules fail less often than FLAG_REVIEW rules). This keeps the
  generator generic while still producing a healthy mix of all three decision labels
  (NFR-DP-3) — REJECT dominates (since any one of several REJECT rules failing is enough),
  so REJECT rules are given a lower per-rule failure probability than FLAG_REVIEW rules to
  leave enough probability mass for APPROVE and FLAG_REVIEW outcomes.
- Edge-case generation (FR-DP-8) is deliberately a *different* code path from normal profile
  generation: it starts from an all-rules-pass baseline and perturbs 2-3 numeric-threshold
  rules at a time (exact threshold, threshold+1, threshold-1), so the held-out test set
  stresses boundary conditions the train/val generator only hits by chance.
- Special tokens delimit structural sections (applicant block, decision block) so a model
  can learn to locate the facts and the verdict independently of surrounding prose; `pad`
  is part of the vocabulary but only appears at tokenization time, not in rendered text.
- The tokenizer is a constructor parameter everywhere (A3.5), so this stage never assumes a
  specific base model; `WhitespaceTokenizer` below is only the default/local test double.
  `main()` opts into the real base model's tokenizer instead (`load_real_tokenizer`) when
  `--tokenizer-model-id` / config's `tokenizer_model_id` is set — required for the produced
  `input_ids` to be real-vocabulary-aligned, since Training's batching feeds them straight
  into the model without re-tokenizing (see docs/srs/data-preparation.md A3.5).
- CLI defaults are layered: built-in constants below < this stage's section of `config.json`
  (via the shared `config.load_config`) < explicit CLI flags (A3.8, IR-DP-4). This keeps the
  script runnable with zero setup while still letting a run's full configuration live in one
  reviewable file instead of scattered module constants.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snhai.config import load_config

# --- Special tokens (FR-DP-6) -----------------------------------------------------------

BOS_TOKEN = "<|begin|>"
EOS_TOKEN = "<|end|>"
PAD_TOKEN = "<|pad|>"
APPLICANT_OPEN = "<|applicant|>"
APPLICANT_CLOSE = "<|/applicant|>"
DECISION_OPEN = "<|decision|>"
DECISION_CLOSE = "<|/decision|>"

SPECIAL_TOKENS = {
    "bos_token": BOS_TOKEN,
    "eos_token": EOS_TOKEN,
    "pad_token": PAD_TOKEN,
    "additional_special_tokens": [
        APPLICANT_OPEN,
        APPLICANT_CLOSE,
        DECISION_OPEN,
        DECISION_CLOSE,
    ],
}

# Defaults carried over from prior exploration (A3.4).
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_SPLIT_COUNTS = {"train": 340, "val": 60, "test": 30}
DEFAULT_SPLIT_RATIOS = (
    DEFAULT_SPLIT_COUNTS["train"] / sum(DEFAULT_SPLIT_COUNTS.values()),
    DEFAULT_SPLIT_COUNTS["val"] / sum(DEFAULT_SPLIT_COUNTS.values()),
    DEFAULT_SPLIT_COUNTS["test"] / sum(DEFAULT_SPLIT_COUNTS.values()),
)

REQUIRED_RULE_FIELDS = {
    "id",
    "name",
    "description",
    "field",
    "operator",
    "action_on_fail",
    "severity",
    "group",
}

# Per-rule failure probability used by the synthetic generator, keyed by the rule's
# `action_on_fail` (data-driven, not rule-id-specific — see NFR-DP-2 and module docstring).
_FAIL_PROBABILITY_BY_ACTION = {"REJECT": 0.16, "FLAG_REVIEW": 0.3}
_DEFAULT_FAIL_PROBABILITY = 0.2

# Default decision-label balance target for `generate_balanced_profiles` — a uniform
# baseline (A3.7), deliberately simple; see docs/srs/data-preparation.md OQ-3 for the
# open question on weighting this by each label's downstream rationale diversity instead.
DEFAULT_LABEL_RATIOS = {"APPROVE": 1 / 3, "REJECT": 1 / 3, "FLAG_REVIEW": 1 / 3}


@dataclass
class RuleResult:
    """The outcome of evaluating one profile against one rule (FR-DP-3)."""

    rule_id: str
    passed: bool
    action_on_fail: str
    severity: str
    rule_name: str = ""
    field: str = ""
    detail: str = ""


# --- FR-DP-1: load & validate ruleset schema ---------------------------------------------


def load_ruleset(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ruleset file not found: {path}")
    with path.open() as f:
        ruleset = json.load(f)
    _validate_ruleset(ruleset)
    return ruleset


def _validate_ruleset(ruleset: dict) -> None:
    try:
        rules = ruleset["personal_loan_credit_rules"]["rules"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Malformed ruleset: expected a 'personal_loan_credit_rules.rules' list"
        ) from exc

    for rule in rules:
        missing = REQUIRED_RULE_FIELDS - rule.keys()
        if missing:
            raise ValueError(
                f"Rule {rule.get('id', '<unknown>')!r} is missing required field(s): "
                f"{sorted(missing)}"
            )
        has_value = "value" in rule
        has_multiplier_field = "value_field_multiplier" in rule
        has_multiplier_value = "multiplier_value" in rule
        if has_multiplier_field != has_multiplier_value:
            raise ValueError(
                f"Rule {rule['id']!r} must define 'value_field_multiplier' and "
                "'multiplier_value' together, or neither"
            )
        if not has_value and not has_multiplier_field:
            raise ValueError(
                f"Rule {rule['id']!r} must define either 'value' or a "
                "'value_field_multiplier'/'multiplier_value' pair"
            )


# --- Field access helpers -----------------------------------------------------------------


def _get_field(record: dict, dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        value = value[part]
    return value


def _set_field(record: dict, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = record
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _rules_of(ruleset: dict) -> list[dict]:
    return ruleset["personal_loan_credit_rules"]["rules"]


def _threshold_of(rule: dict, profile: dict) -> Any:
    if "value_field_multiplier" in rule:
        other_value = _get_field(profile, rule["value_field_multiplier"])
        return other_value * rule["multiplier_value"]
    return rule["value"]


# --- FR-DP-3 / NFR-DP-2: generic, data-driven rule evaluation ------------------------------


def _evaluate_rule(rule: dict, profile: dict) -> tuple[bool, str]:
    field_value = _get_field(profile, rule["field"])
    operator = rule["operator"]
    threshold = _threshold_of(rule, profile)

    if operator == ">=":
        passed = field_value >= threshold
    elif operator == "<=":
        passed = field_value <= threshold
    elif operator == "in":
        passed = field_value in threshold
    elif operator == "is":
        passed = field_value is threshold
    else:
        raise ValueError(
            f"Unsupported operator {operator!r} in rule {rule.get('id')!r}"
        )

    detail = f"{rule['field']}={field_value!r} {operator} {threshold!r} -> {'PASS' if passed else 'FAIL'}"
    return passed, detail


def evaluate_profile(ruleset: dict, profile: dict) -> list[RuleResult]:
    results = []
    for rule in _rules_of(ruleset):
        passed, detail = _evaluate_rule(rule, profile)
        results.append(
            RuleResult(
                rule_id=rule["id"],
                passed=passed,
                action_on_fail=rule["action_on_fail"],
                severity=rule["severity"],
                rule_name=rule.get("name", ""),
                field=rule["field"],
                detail=detail,
            )
        )
    return results


# --- FR-DP-4 / A3.3: decision precedence ---------------------------------------------------


def derive_decision(results: list[RuleResult]) -> str:
    if any(not r.passed and r.action_on_fail == "REJECT" for r in results):
        return "REJECT"
    if any(not r.passed and r.action_on_fail == "FLAG_REVIEW" for r in results):
        return "FLAG_REVIEW"
    return "APPROVE"


# --- FR-DP-2 / NFR-DP-1 / NFR-DP-3: synthetic profile generation ---------------------------


def _sample_numeric(
    threshold: float, operator: str, fail_prob: float, rng: random.Random
) -> int:
    spread = max(abs(threshold) * 0.3, 5)
    should_fail = rng.random() < fail_prob
    if operator == ">=":
        value = (
            threshold - rng.uniform(0.1, spread)
            if should_fail
            else threshold + rng.uniform(0, spread)
        )
    else:  # "<="
        value = (
            threshold + rng.uniform(0.1, spread)
            if should_fail
            else threshold - rng.uniform(0, spread)
        )
    return round(value)


def _sample_categorical(allowed: list, fail_prob: float, rng: random.Random):
    if rng.random() < fail_prob:
        return f"invalid_{rng.randint(1, 1_000_000)}"
    return rng.choice(allowed)


def _sample_boolean(pass_value: bool, fail_prob: float, rng: random.Random) -> bool:
    should_fail = rng.random() < fail_prob
    return (not pass_value) if should_fail else pass_value


def _sample_field_value(rule: dict, profile: dict, rng: random.Random):
    fail_prob = _FAIL_PROBABILITY_BY_ACTION.get(
        rule["action_on_fail"], _DEFAULT_FAIL_PROBABILITY
    )
    operator = rule["operator"]
    threshold = _threshold_of(rule, profile)

    if operator in (">=", "<="):
        return _sample_numeric(threshold, operator, fail_prob, rng)
    if operator == "in":
        return _sample_categorical(list(threshold), fail_prob, rng)
    if operator == "is":
        return _sample_boolean(bool(threshold), fail_prob, rng)
    raise ValueError(f"Unsupported operator {operator!r} in rule {rule.get('id')!r}")


def generate_applicant_profiles(ruleset: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rules = _rules_of(ruleset)
    simple_rules = [r for r in rules if "value_field_multiplier" not in r]
    multiplier_rules = [r for r in rules if "value_field_multiplier" in r]

    profiles = []
    for _ in range(n):
        profile: dict = {}
        for rule in simple_rules:
            _set_field(profile, rule["field"], _sample_field_value(rule, profile, rng))
        for rule in multiplier_rules:
            _set_field(profile, rule["field"], _sample_field_value(rule, profile, rng))
        profiles.append(profile)
    return profiles


def generate_balanced_profiles(
    ruleset: dict,
    n: int,
    seed: int,
    target_label_ratios: dict[str, float] = DEFAULT_LABEL_RATIOS,
    oversample_factor: int = 20,
) -> list[dict]:
    """Stratified-resample `generate_applicant_profiles` output toward `target_label_ratios`
    (A3.7). Default is a uniform 1/3-per-label baseline rather than the natural ~75/15/12
    REJECT/FLAG_REVIEW/APPROVE split `generate_applicant_profiles` otherwise produces; see
    OQ-3 for the open question on a diversity-weighted (non-uniform) target.

    Best-effort: if the oversampled pool doesn't contain enough examples of a label to fill
    its quota, that label's bucket is short and the returned list is smaller than `n`.
    """
    rng = random.Random(seed)
    pool = generate_applicant_profiles(ruleset, n=n * oversample_factor, seed=seed)

    buckets: dict[str, list[dict]] = {label: [] for label in target_label_ratios}
    for profile in pool:
        decision = derive_decision(evaluate_profile(ruleset, profile))
        if decision in buckets:
            buckets[decision].append(profile)

    quotas = {label: round(n * ratio) for label, ratio in target_label_ratios.items()}
    remainder = n - sum(quotas.values())
    if remainder:
        largest_label = max(quotas, key=quotas.get)
        quotas[largest_label] += remainder

    selected: list[dict] = []
    for label, quota in quotas.items():
        selected.extend(buckets[label][:quota])

    rng.shuffle(selected)
    return selected


# --- FR-DP-8: held-out multi-rule edge case test set ----------------------------------------


def _baseline_pass_value(rule: dict, profile: dict):
    operator = rule["operator"]
    threshold = _threshold_of(rule, profile)
    if operator == ">=":
        return threshold + max(abs(threshold) * 0.1, 1)
    if operator == "<=":
        return threshold - max(abs(threshold) * 0.1, 1)
    if operator == "in":
        return threshold[0]
    if operator == "is":
        return threshold
    raise ValueError(f"Unsupported operator {operator!r} in rule {rule.get('id')!r}")


def _build_baseline_profile(rules: list[dict]) -> dict:
    simple_rules = [r for r in rules if "value_field_multiplier" not in r]
    multiplier_rules = [r for r in rules if "value_field_multiplier" in r]
    profile: dict = {}
    for rule in simple_rules:
        _set_field(profile, rule["field"], _baseline_pass_value(rule, profile))
    for rule in multiplier_rules:
        _set_field(profile, rule["field"], _baseline_pass_value(rule, profile))
    return profile


def generate_edge_case_profiles(ruleset: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rules = _rules_of(ruleset)
    numeric_rules = [r for r in rules if r["operator"] in (">=", "<=")]

    profiles = []

    # Single-rule cases pinned to the exact numeric threshold (e.g. credit_score == 670).
    for rule in numeric_rules:
        profile = _build_baseline_profile(rules)
        _set_field(profile, rule["field"], _threshold_of(rule, profile))
        profiles.append(profile)

    # Multi-rule (2-3 simultaneous) perturbations around each rule's threshold.
    for combo_size in (2, 3):
        combos = list(itertools.combinations(numeric_rules, combo_size))
        rng.shuffle(combos)
        for combo in combos[: min(4, len(combos))]:
            profile = _build_baseline_profile(rules)
            for rule in combo:
                shift = rng.choice([0, 1, -1])
                _set_field(profile, rule["field"], _threshold_of(rule, profile) + shift)
            profiles.append(profile)

    return profiles


# --- FR-DP-5 / FR-DP-6: natural-language rendering with special tokens ---------------------


def _flatten_profile(record: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in record.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_profile(value, path))
        else:
            items.append((path, value))
    return items


def render_example(profile: dict, results: list[RuleResult], decision: str) -> str:
    facts = "; ".join(f"{path}={value}" for path, value in _flatten_profile(profile))

    if decision == "APPROVE":
        rationale = (
            "All eligibility, creditworthiness, and suitability rules were satisfied."
        )
    else:
        driving = [r for r in results if not r.passed and r.action_on_fail == decision]
        rule_ids = ", ".join(r.rule_id for r in driving)
        rationale = f"Decision driven by failed rule(s): {rule_ids}."

    lines = [
        BOS_TOKEN,
        APPLICANT_OPEN,
        facts,
        APPLICANT_CLOSE,
        DECISION_OPEN,
        f"Decision: {decision}",
        f"Rationale: {rationale}",
        DECISION_CLOSE,
        EOS_TOKEN,
    ]
    return "\n".join(lines)


# --- FR-DP-7: tokenization with padding/truncation ------------------------------------------


def tokenize_example(text: str, tokenizer, max_seq_len: int) -> dict:
    token_ids = list(tokenizer.encode(text))
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_seq_len:
        pad_len = max_seq_len - len(token_ids)
        token_ids = token_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
    else:
        token_ids = token_ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]

    return {"input_ids": token_ids, "attention_mask": attention_mask}


class WhitespaceTokenizer:
    """Deterministic whitespace tokenizer; the default/fallback injected tokenizer (A3.5)."""

    def __init__(self, special_tokens: list[str] | None = None):
        self.vocab: dict[str, int] = {}
        self.pad_token = PAD_TOKEN
        for token in special_tokens or []:
            self._register(token)
        self._register(self.pad_token)

    def _register(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    def encode(self, text: str) -> list[int]:
        return [self._register(tok) for tok in text.split()]


class _RealTokenizerAdapter:
    """Wraps a real HF tokenizer to match WhitespaceTokenizer's .encode/.pad_token_id
    contract, so it's a drop-in injected tokenizer for tokenize_example (A3.5)."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.pad_token_id = hf_tokenizer.pad_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


def load_real_tokenizer(model_id: str) -> _RealTokenizerAdapter:
    """Loads `model_id`'s real tokenizer and registers this pipeline's special tokens on
    it, mirroring training.py:align_tokenizer_and_model's tokenizer.add_special_tokens(...)
    call so measured/produced token ids reflect the vocabulary that will actually exist at
    training time. Training's batching feeds Data Preparation's `input_ids` straight into
    the model without re-tokenizing (see training.py's `_batches`), so those ids must
    already be real-tokenizer-aligned -- this is what lets `main()` opt into that instead of
    the WhitespaceTokenizer default (A3.5) once a base model has been chosen."""
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return _RealTokenizerAdapter(hf_tokenizer)


# --- FR-DP-9: non-overlapping train/val/test split -------------------------------------------


def split_dataset(
    examples: list,
    seed: int,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_frac, val_frac, _ = split_ratios
    n_train = min(round(n * train_frac), n)
    n_val = min(round(n * val_frac), n - n_train)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


# --- FR-DP-10 / IR-DP-3: data card ------------------------------------------------------------


def build_data_card(
    n_rules: int,
    max_seq_len: int,
    tokenizer_id: str,
    special_tokens: dict,
    splits: dict,
    decision_label_distribution: dict,
) -> dict:
    return {
        "n_rules": n_rules,
        "max_seq_len": max_seq_len,
        "tokenizer_id": tokenizer_id,
        "special_tokens": special_tokens,
        "splits": splits,
        "decision_label_distribution": decision_label_distribution,
    }


# --- IR-DP-2: dataset persistence -------------------------------------------------------------


def save_dataset(examples: list[dict], path: str | Path) -> None:
    path = Path(path)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def load_dataset(path: str | Path) -> list[dict]:
    path = Path(path)
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# --- CLI orchestration (not exercised by the unit test suite) --------------------------------


def _build_examples(
    profiles: list[dict], ruleset: dict, tokenizer, max_seq_len: int
) -> list[dict]:
    examples = []
    for profile in profiles:
        results = evaluate_profile(ruleset, profile)
        decision = derive_decision(results)
        text = render_example(profile, results, decision)
        tokenized = tokenize_example(text, tokenizer=tokenizer, max_seq_len=max_seq_len)
        examples.append({**tokenized, "labels": decision})
    return examples


def _label_distribution(examples: list[dict]) -> dict:
    counts = {"APPROVE": 0, "REJECT": 0, "FLAG_REVIEW": 0}
    for example in examples:
        counts[example["labels"]] = counts.get(example["labels"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> None:
    # Phase 1: find --config (if any) before the rest of argparse needs its values as defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.json"))
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)
    stage_config = load_config(pre_args.config, "data_preparation")

    # Phase 2: full parser, defaults layered as config.json < CLI flags (A3.8).
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path(stage_config.get("rules_path", "fine_tune_llm_credit_rules.json")),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(stage_config.get("output_dir", "data"))
    )
    parser.add_argument("--seed", type=int, default=stage_config.get("seed", 42))
    parser.add_argument(
        "--n-profiles", type=int, default=stage_config.get("n_profiles", 400)
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=stage_config.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default=stage_config.get("tokenizer_model_id"),
    )
    args = parser.parse_args(remaining_argv)

    target_label_ratios = stage_config.get("target_label_ratios", DEFAULT_LABEL_RATIOS)
    split_counts = stage_config.get("split_counts", DEFAULT_SPLIT_COUNTS)
    split_ratios = tuple(
        split_counts[key] / sum(split_counts.values())
        for key in ("train", "val", "test")
    )

    ruleset = load_ruleset(args.rules_path)
    if args.tokenizer_model_id:
        tokenizer = load_real_tokenizer(args.tokenizer_model_id)
        tokenizer_id = args.tokenizer_model_id
    else:
        tokenizer = WhitespaceTokenizer(
            special_tokens=list(SPECIAL_TOKENS.values())[:3]
            + SPECIAL_TOKENS["additional_special_tokens"]
        )
        tokenizer_id = "whitespace"

    profiles = generate_balanced_profiles(
        ruleset,
        n=args.n_profiles,
        seed=args.seed,
        target_label_ratios=target_label_ratios,
    )
    normal_examples = _build_examples(profiles, ruleset, tokenizer, args.max_seq_len)
    train, val, _ = split_dataset(
        normal_examples, seed=args.seed, split_ratios=split_ratios
    )

    edge_profiles = generate_edge_case_profiles(ruleset, seed=args.seed)
    test = _build_examples(edge_profiles, ruleset, tokenizer, args.max_seq_len)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(train, args.output_dir / "train.jsonl")
    save_dataset(val, args.output_dir / "val.jsonl")
    save_dataset(test, args.output_dir / "test.jsonl")

    card = build_data_card(
        n_rules=len(_rules_of(ruleset)),
        max_seq_len=args.max_seq_len,
        tokenizer_id=tokenizer_id,
        special_tokens=SPECIAL_TOKENS,
        splits={"train": len(train), "val": len(val), "test": len(test)},
        decision_label_distribution={
            "train": _label_distribution(train),
            "val": _label_distribution(val),
            "test": _label_distribution(test),
        },
    )
    (args.output_dir / "data_card.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
