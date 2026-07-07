"""Failing unit tests for the Data Preparation stage.

Spec: docs/srs/data-preparation.md. Each test's docstring names the FR-DP-#/NFR-DP-#/IR-DP-#
requirement(s) it covers, so `grep -o 'DP-[0-9]*' tests/test_data_preparation.py` against the
spec shows coverage. These tests import `data_preparation`, which does not exist yet — they
are expected to fail (collection error) until that stage is implemented.
"""

import json
import random
import time
from pathlib import Path

import pytest

from snhai import data_preparation as dp

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "fine_tune_llm_credit_rules.json"

# A minimal, deliberately different ruleset (different ids/fields/operators) used to prove
# rule evaluation is generic rather than hard-coded against the real 10 rules (NFR-DP-2).
CUSTOM_RULESET = {
    "personal_loan_credit_rules": {
        "version": "0.1",
        "description": "Synthetic ruleset for genericity testing.",
        "rules": [
            {
                "id": "RULE-TEST-001",
                "name": "Minimum Widget Count",
                "description": "Applicant must own at least 3 widgets.",
                "field": "applicant.widget_count",
                "operator": ">=",
                "value": 3,
                "action_on_fail": "REJECT",
                "severity": "CRITICAL",
                "group": "Widgets",
            },
            {
                "id": "RULE-TEST-002",
                "name": "Widget Type Allowed",
                "description": "Applicant's widget type must be allowed.",
                "field": "applicant.widget_type",
                "operator": "in",
                "value": ["red", "blue"],
                "action_on_fail": "FLAG_REVIEW",
                "severity": "MINOR",
                "group": "Widgets",
            },
        ],
    }
}


def _get_by_path(record: dict, dotted_path: str):
    value = record
    for part in dotted_path.split("."):
        value = value[part]
    return value


def _all_field_paths(ruleset: dict) -> set[str]:
    return {rule["field"] for rule in ruleset["personal_loan_credit_rules"]["rules"]}


@pytest.fixture
def ruleset() -> dict:
    """The real ruleset, loaded fresh per test (cheap: 10 rules). Local to this file until
    a second test module (training/evaluation) needs it too — see docs/srs/data-preparation.md."""
    return dp.load_ruleset(RULES_PATH)


@pytest.fixture
def tokenizer() -> "_WhitespaceTokenizer":
    """Cheap test-double tokenizer standing in for the injected tokenizer (A3.5)."""
    return _WhitespaceTokenizer()


# --- FR-DP-1: load & validate ruleset schema ---------------------------------------------


def test_load_ruleset_returns_all_rules_with_required_fields():
    """FR-DP-1: loading the real ruleset yields every rule with its required schema fields."""
    ruleset = dp.load_ruleset(RULES_PATH)
    rules = ruleset["personal_loan_credit_rules"]["rules"]
    assert len(rules) == 10
    required = {
        "id",
        "name",
        "description",
        "field",
        "operator",
        "action_on_fail",
        "severity",
        "group",
    }
    for rule in rules:
        assert required.issubset(rule.keys())


def test_load_ruleset_rejects_malformed_schema(tmp_path):
    """FR-DP-1: a rule missing a required field (action_on_fail) is rejected with a descriptive error."""
    bad = {
        "personal_loan_credit_rules": {
            "version": "1.0",
            "description": "bad",
            "rules": [
                {
                    "id": "X",
                    "name": "x",
                    "field": "a.b",
                    "operator": ">=",
                    "value": 1,
                    "severity": "MINOR",
                    "group": "g",
                }
            ],
        }
    }
    bad_path = tmp_path / "bad_rules.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        dp.load_ruleset(bad_path)


def test_load_ruleset_missing_file_raises():
    """IR-DP-1: a nonexistent ruleset path raises FileNotFoundError, not a silent empty result."""
    with pytest.raises(FileNotFoundError):
        dp.load_ruleset(REPO_ROOT / "does_not_exist.json")


# --- FR-DP-2: synthetic profile generation -----------------------------------------------


def test_generated_profiles_cover_every_referenced_field(ruleset):
    """FR-DP-2: every profile supplies a value for every field referenced by the ruleset."""
    profiles = dp.generate_applicant_profiles(ruleset, n=20, seed=42)
    assert len(profiles) == 20
    field_paths = _all_field_paths(ruleset)
    for profile in profiles:
        for path in field_paths:
            _get_by_path(profile, path)  # raises KeyError if missing


# --- FR-DP-3: evaluate against ALL rules, no short-circuit --------------------------------


def test_evaluate_profile_reports_every_rule_even_with_multiple_failures(ruleset):
    """FR-DP-3: a profile failing several rules still gets a result recorded for every rule."""
    rules = ruleset["personal_loan_credit_rules"]["rules"]
    profile = {
        "applicant": {
            "age": 16,  # fails RULE-AGE-001
            "credit_score": 500,  # fails RULE-CREDIT-001
            "annual_income_usd": 10000,  # fails RULE-INCOME-001
            "debt_to_income_ratio_percent": 80,  # fails RULE-DTI-001
            "employment_status": "unemployed",  # fails RULE-EMPLOY-001
            "current_employment_duration_months": 0,  # fails RULE-EMPLOY-002
            "residency_status": "non_resident",  # fails RULE-RESIDENCY-001
            "has_bankruptcy_recent": True,  # fails RULE-BANKRUPTCY-001
            "has_verifiable_bank_account": False,  # fails RULE-BANKACCTS-001
        },
        "loan_application": {"requested_amount_usd": 50000},  # fails RULE-LOANAMT-001
    }
    results = dp.evaluate_profile(ruleset, profile)
    assert len(results) == len(rules)
    assert all(not result.passed for result in results)


# --- FR-DP-4 / A3.3: decision precedence ---------------------------------------------------


def test_decision_reject_dominates_flag_review():
    """FR-DP-4, A3.3: if a REJECT-action rule and a FLAG_REVIEW-action rule both fail, decision is REJECT."""
    results = [
        dp.RuleResult(
            rule_id="R1", passed=False, action_on_fail="REJECT", severity="CRITICAL"
        ),
        dp.RuleResult(
            rule_id="R2", passed=False, action_on_fail="FLAG_REVIEW", severity="MINOR"
        ),
    ]
    assert dp.derive_decision(results) == "REJECT"


def test_decision_flag_review_when_only_flag_rules_fail():
    """FR-DP-4: if only FLAG_REVIEW-action rules fail, decision is FLAG_REVIEW."""
    results = [
        dp.RuleResult(
            rule_id="R1", passed=True, action_on_fail="REJECT", severity="CRITICAL"
        ),
        dp.RuleResult(
            rule_id="R2", passed=False, action_on_fail="FLAG_REVIEW", severity="MINOR"
        ),
    ]
    assert dp.derive_decision(results) == "FLAG_REVIEW"


def test_decision_approve_when_all_rules_pass():
    """FR-DP-4: if every rule passes, decision is APPROVE."""
    results = [
        dp.RuleResult(
            rule_id="R1", passed=True, action_on_fail="REJECT", severity="CRITICAL"
        ),
        dp.RuleResult(
            rule_id="R2", passed=True, action_on_fail="FLAG_REVIEW", severity="MINOR"
        ),
    ]
    assert dp.derive_decision(results) == "APPROVE"


# --- FR-DP-5 / FR-DP-6: rendering with special tokens --------------------------------------


def test_render_example_wraps_sections_in_special_tokens_and_names_driving_rule(
    ruleset,
):
    """FR-DP-5, FR-DP-6: rendered text delimits applicant/decision blocks and names the driving rule id."""
    profile = {"applicant": {"age": 16}, "loan_application": {}}
    # fill remaining required fields with passing values so RULE-AGE-001 is the sole failure
    for path in _all_field_paths(ruleset):
        parts = path.split(".")
        d = profile.setdefault(parts[0], {})
        d.setdefault(parts[1], True)
    profile["applicant"]["age"] = 16
    results = dp.evaluate_profile(ruleset, profile)
    decision = dp.derive_decision(results)
    text = dp.render_example(profile, results, decision)
    assert "<|applicant|>" in text and "<|/applicant|>" in text
    assert "<|decision|>" in text and "<|/decision|>" in text
    assert "RULE-AGE-001" in text


# --- FR-DP-7: tokenization with padding/truncation ------------------------------------------


class _WhitespaceTokenizer:
    """Minimal test-double tokenizer standing in for the injected tokenizer (A3.5)."""

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [hash(tok) % 1000 + 1 for tok in text.split()]


def test_tokenize_example_pads_short_sequences_to_max_seq_len(tokenizer):
    """FR-DP-7: a short example is padded up to max_seq_len."""
    tokenized = dp.tokenize_example("short text", tokenizer=tokenizer, max_seq_len=16)
    assert len(tokenized["input_ids"]) == 16
    assert len(tokenized["attention_mask"]) == 16
    assert tokenized["attention_mask"][-1] == 0


def test_tokenize_example_truncates_long_sequences_to_max_seq_len(tokenizer):
    """FR-DP-7: a long example is truncated down to max_seq_len."""
    long_text = " ".join(f"word{i}" for i in range(100))
    tokenized = dp.tokenize_example(long_text, tokenizer=tokenizer, max_seq_len=16)
    assert len(tokenized["input_ids"]) == 16


# --- FR-DP-8: held-out multi-rule edge case test set ----------------------------------------


def test_edge_case_profiles_perturb_multiple_rules_and_hit_exact_thresholds(ruleset):
    """FR-DP-8: edge-case profiles perturb 2-3 rules at once, including exact numeric thresholds."""
    edge_profiles = dp.generate_edge_case_profiles(ruleset, seed=42)
    assert len(edge_profiles) > 0
    assert any(profile["applicant"]["credit_score"] == 670 for profile in edge_profiles)


def test_edge_case_profile_generation_does_not_mutate_global_random_state(ruleset):
    """A3.6: edge-case generation SHALL use a local RNG, not the global `random` module."""
    state_before = random.getstate()
    dp.generate_edge_case_profiles(ruleset, seed=42)
    assert random.getstate() == state_before


# --- FR-DP-9: non-overlapping train/val/test split ------------------------------------------


def test_split_dataset_produces_non_overlapping_partitions():
    """FR-DP-9: train/val/test partitions are disjoint and cover all examples."""
    examples = [{"id": i} for i in range(100)]
    train, val, test = dp.split_dataset(examples, seed=42)
    train_ids = {e["id"] for e in train}
    val_ids = {e["id"] for e in val}
    test_ids = {e["id"] for e in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == {e["id"] for e in examples}


def test_split_dataset_does_not_mutate_global_random_state():
    """A3.6: dataset splitting SHALL use a local RNG, not the global `random` module."""
    examples = [{"id": i} for i in range(100)]
    state_before = random.getstate()
    dp.split_dataset(examples, seed=42)
    assert random.getstate() == state_before


# --- FR-DP-10 / IR-DP-3: data card schema ----------------------------------------------------


def test_data_card_contains_required_keys():
    """FR-DP-10, IR-DP-3: the data card documents rule count, tokenizer config, splits, and label distribution."""
    card = dp.build_data_card(
        n_rules=10,
        max_seq_len=256,
        tokenizer_id="whitespace",
        special_tokens={
            "bos_token": "<|begin|>",
            "eos_token": "<|end|>",
            "pad_token": "<|pad|>",
        },
        splits={"train": 340, "val": 60, "test": 30},
        decision_label_distribution={"APPROVE": 228, "REJECT": 164, "FLAG_REVIEW": 38},
    )
    required = {
        "n_rules",
        "max_seq_len",
        "tokenizer_id",
        "special_tokens",
        "splits",
        "decision_label_distribution",
    }
    assert required.issubset(card.keys())


# --- NFR-DP-1: reproducibility ---------------------------------------------------------------


def test_profile_generation_is_deterministic_given_same_seed(ruleset):
    """NFR-DP-1: the same seed produces identical profiles across runs."""
    first = dp.generate_applicant_profiles(ruleset, n=10, seed=42)
    second = dp.generate_applicant_profiles(ruleset, n=10, seed=42)
    assert first == second


def test_profile_generation_does_not_mutate_global_random_state(ruleset):
    """NFR-DP-1, A3.6: generation SHALL use a local RNG, not the global `random` module, so
    concurrent/adjacent calls (and other tests) aren't affected by its seeding."""
    state_before = random.getstate()
    dp.generate_applicant_profiles(ruleset, n=10, seed=42)
    assert random.getstate() == state_before


# --- A3.7: configurable decision-label balance target (default: uniform 1/3 baseline) ---------


def test_default_label_ratios_are_uniform():
    """A3.7: the documented default target is an equal 1/3 per label, not a tuned split."""
    assert dp.DEFAULT_LABEL_RATIOS == pytest.approx(
        {"APPROVE": 1 / 3, "REJECT": 1 / 3, "FLAG_REVIEW": 1 / 3}
    )


def test_generate_balanced_profiles_respects_target_label_ratios(ruleset):
    """A3.7: a non-uniform target_label_ratios is honored, proving the split is configurable."""
    profiles = dp.generate_balanced_profiles(
        ruleset,
        n=60,
        seed=42,
        target_label_ratios={"APPROVE": 0.5, "REJECT": 0.3, "FLAG_REVIEW": 0.2},
    )
    decisions = [dp.derive_decision(dp.evaluate_profile(ruleset, p)) for p in profiles]
    counts = {
        label: decisions.count(label) for label in ("APPROVE", "REJECT", "FLAG_REVIEW")
    }
    assert counts["APPROVE"] >= counts["REJECT"] >= counts["FLAG_REVIEW"]


def test_generate_balanced_profiles_is_deterministic_given_same_seed(ruleset):
    """NFR-DP-1: the same seed produces identical balanced profiles across runs."""
    first = dp.generate_balanced_profiles(ruleset, n=30, seed=7)
    second = dp.generate_balanced_profiles(ruleset, n=30, seed=7)
    assert first == second


def test_generate_balanced_profiles_does_not_mutate_global_random_state(ruleset):
    """A3.6: balanced generation SHALL use a local RNG, not the global `random` module."""
    state_before = random.getstate()
    dp.generate_balanced_profiles(ruleset, n=30, seed=7)
    assert random.getstate() == state_before


# --- NFR-DP-2: rule evaluation is generic, not id-specific ------------------------------------


def test_rule_evaluation_works_on_an_unfamiliar_custom_ruleset():
    """NFR-DP-2: evaluation logic is data-driven — it must work on rules/ids it has never seen."""
    profile_pass = {"applicant": {"widget_count": 5, "widget_type": "red"}}
    profile_fail = {"applicant": {"widget_count": 1, "widget_type": "green"}}
    results_pass = dp.evaluate_profile(CUSTOM_RULESET, profile_pass)
    results_fail = dp.evaluate_profile(CUSTOM_RULESET, profile_fail)
    assert all(r.passed for r in results_pass)
    assert all(not r.passed for r in results_fail)


# --- NFR-DP-3: label coverage -----------------------------------------------------------------


def test_generated_dataset_contains_all_three_labels_above_minimum_threshold(ruleset):
    """NFR-DP-3: train/val generation yields all three decision labels, each at >=5% of examples."""
    profiles = dp.generate_applicant_profiles(ruleset, n=200, seed=42)
    decisions = [dp.derive_decision(dp.evaluate_profile(ruleset, p)) for p in profiles]
    counts = {
        label: decisions.count(label) for label in ("APPROVE", "REJECT", "FLAG_REVIEW")
    }
    total = len(decisions)
    for label, count in counts.items():
        assert count / total >= 0.05, f"{label} under-represented: {count}/{total}"


# --- NFR-DP-4: documented rationale (documentation presence, not code behavior) ---------------


def test_data_preparation_spec_documents_rationale():
    """NFR-DP-4: the SRS records rationale for preprocessing choices alongside the code."""
    spec = (REPO_ROOT / "docs" / "srs" / "data-preparation.md").read_text()
    assert "NFR-DP-4" in spec


# --- NFR-DP-5: bounded performance -------------------------------------------------------------


def test_dataset_generation_completes_within_time_budget(ruleset):
    """NFR-DP-5: generating 200 examples completes quickly on CPU (no GPU dependency)."""
    start = time.monotonic()
    dp.generate_applicant_profiles(ruleset, n=200, seed=42)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0


# --- IR-DP-2: output dataset artifact is loadable in a training-ready format --------------------


def test_saved_dataset_round_trips_with_training_ready_keys(tmp_path):
    """IR-DP-2: persisted examples reload with input_ids/attention_mask/labels for the Training stage."""
    tokenized_examples = [
        {
            "input_ids": [1, 2, 3, 0],
            "attention_mask": [1, 1, 1, 0],
            "labels": "APPROVE",
        },
        {"input_ids": [4, 5, 6, 0], "attention_mask": [1, 1, 1, 0], "labels": "REJECT"},
    ]
    out_path = tmp_path / "train.jsonl"
    dp.save_dataset(tokenized_examples, out_path)
    loaded = dp.load_dataset(out_path)
    assert loaded == tokenized_examples
    for example in loaded:
        assert {"input_ids", "attention_mask", "labels"}.issubset(example.keys())
