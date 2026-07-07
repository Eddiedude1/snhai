"""One-off analysis script: measures real token-length distributions of rendered Data
Preparation examples to resolve docs/srs/data-preparation.md OQ-1 (max_seq_len / split-count
calibration).

Not part of the `snhai` package or its test suite: this is the only place in the repo (besides
training.py's/evaluation.py's own lazy default-loader functions) that imports `transformers`,
and it lives outside `src/snhai/` so the installable package and its unit tests stay
torch/transformers-free. `data_preparation.py`'s pipeline itself is unchanged and keeps using
`WhitespaceTokenizer` as its default injected tokenizer (A3.5) — this script only measures
against the real base-model tokenizer to calibrate `max_seq_len`.

Usage: uv run python scripts/measure_token_lengths.py [--config config.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from snhai import data_preparation as dp
from snhai.config import load_config


class _RealTokenizerAdapter:
    """Wraps a HF tokenizer to match WhitespaceTokenizer's .encode/.pad_token_id contract."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.pad_token_id = hf_tokenizer.pad_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


def _load_real_tokenizer(model_id: str) -> _RealTokenizerAdapter:
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Mirrors training.py:align_tokenizer_and_model's tokenizer.add_special_tokens(...) call,
    # so measured lengths reflect the vocabulary that will actually exist at training time.
    hf_tokenizer.add_special_tokens(dp.SPECIAL_TOKENS)
    return _RealTokenizerAdapter(hf_tokenizer)


def _render_all(ruleset: dict, profiles: list[dict]) -> list[tuple[str, str]]:
    rendered = []
    for profile in profiles:
        results = dp.evaluate_profile(ruleset, profile)
        decision = dp.derive_decision(results)
        text = dp.render_example(profile, results, decision)
        rendered.append((text, decision))
    return rendered


def _length_stats(lengths: list[int], max_seq_len: int) -> dict:
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def pct(p: float) -> int:
        idx = min(n - 1, int(p * n))
        return lengths_sorted[idx]

    truncated = [length for length in lengths_sorted if length > max_seq_len]
    return {
        "n": n,
        "min": lengths_sorted[0],
        "mean": round(statistics.mean(lengths_sorted), 2),
        "median": statistics.median(lengths_sorted),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": lengths_sorted[-1],
        "n_truncated": len(truncated),
        "truncation_rate": round(len(truncated) / n, 4),
    }


def _label_distribution(rendered: list[tuple[str, str]]) -> dict:
    counts = {"APPROVE": 0, "REJECT": 0, "FLAG_REVIEW": 0}
    for _, decision in rendered:
        counts[decision] = counts.get(decision, 0) + 1
    total = len(rendered) or 1
    return {
        label: {"count": count, "pct": round(count / total, 4)}
        for label, count in counts.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/analysis/token_length_measurement.json"),
    )
    args = parser.parse_args(argv)

    dp_cfg = load_config(args.config, "data_preparation")
    tr_cfg = load_config(args.config, "training")

    model_id = tr_cfg.get("model_id")
    if not model_id:
        raise SystemExit(
            "training.model_id must be set in config.json to measure real token lengths"
        )

    rules_path = dp_cfg.get("rules_path", "fine_tune_llm_credit_rules.json")
    seed = dp_cfg.get("seed", 42)
    n_profiles = dp_cfg.get("n_profiles", 400)
    target_label_ratios = dp_cfg.get("target_label_ratios", dp.DEFAULT_LABEL_RATIOS)
    configured_max_seq_len = dp_cfg.get("max_seq_len", dp.DEFAULT_MAX_SEQ_LEN)
    split_counts = dp_cfg.get("split_counts", dp.DEFAULT_SPLIT_COUNTS)

    ruleset = dp.load_ruleset(rules_path)
    profiles = dp.generate_balanced_profiles(
        ruleset, n=n_profiles, seed=seed, target_label_ratios=target_label_ratios
    )
    edge_profiles = dp.generate_edge_case_profiles(ruleset, seed=seed)

    pool_rendered = _render_all(ruleset, profiles)
    edge_rendered = _render_all(ruleset, edge_profiles)

    special_tokens = (
        list(dp.SPECIAL_TOKENS.values())[:3]
        + dp.SPECIAL_TOKENS["additional_special_tokens"]
    )
    whitespace_tok = dp.WhitespaceTokenizer(special_tokens=special_tokens)
    real_tok = _load_real_tokenizer(model_id)

    tokenizers = {
        "whitespace_naive_proxy": whitespace_tok,
        "real_model_tokenizer": real_tok,
    }
    pools = {"balanced_pool": pool_rendered, "edge_case_pool": edge_rendered}

    length_report: dict = {}
    for tok_name, tokenizer in tokenizers.items():
        length_report[tok_name] = {}
        for pool_name, rendered in pools.items():
            lengths = [len(tokenizer.encode(text)) for text, _ in rendered]
            length_report[tok_name][pool_name] = _length_stats(
                lengths, configured_max_seq_len
            )

    # Split-count reality check: main() derives split_ratios from split_counts the same way.
    split_ratios = tuple(
        split_counts[key] / sum(split_counts.values())
        for key in ("train", "val", "test")
    )
    train, val, discarded = dp.split_dataset(
        profiles, seed=seed, split_ratios=split_ratios
    )
    train_rendered = _render_all(ruleset, train)
    val_rendered = _render_all(ruleset, val)

    split_report = {
        "configured_split_counts": split_counts,
        "actual_n_train": len(train),
        "actual_n_val": len(val),
        "actual_n_discarded_from_pool": len(discarded),
        "actual_n_test_edge_cases": len(edge_profiles),
        "note": (
            "main() sizes train/val from split_dataset(profiles, split_ratios) but sizes the "
            "test file from generate_edge_case_profiles(), NOT from split_counts['test'] or "
            "the 'discarded' third of split_dataset above. split_counts['test'] only "
            "participates as a ratio denominator diluting train/val below their nominal counts."
        ),
        "train_label_distribution": _label_distribution(train_rendered),
        "val_label_distribution": _label_distribution(val_rendered),
    }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "model_id": model_id,
        "seed": seed,
        "n_profiles": n_profiles,
        "configured_max_seq_len": configured_max_seq_len,
        "token_length_stats": length_report,
        "split_check": split_report,
        "notes": {
            "whitespace_naive_proxy": (
                "Naive whitespace .split() token count. render_example's facts line has no "
                "internal whitespace within a fact (e.g. 'applicant.credit_score=792'), so "
                "this severely undercounts real subword tokenization. Kept only as a contrast "
                "data point -- do not use it to size max_seq_len."
            ),
            "tokenizer_id_in_data_card": (
                "data_preparation.main() still uses WhitespaceTokenizer to produce the "
                "committed dataset's input_ids (A3.5); real_model_tokenizer above is used only "
                "for this calibration measurement, not for producing the dataset artifact."
            ),
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")

    real_stats = length_report["real_model_tokenizer"]
    overall_truncated = (
        real_stats["balanced_pool"]["n_truncated"]
        + real_stats["edge_case_pool"]["n_truncated"]
    )

    print(f"Report written to {args.report_path}")
    print(
        f"Real tokenizer ({model_id}) stats vs configured max_seq_len={configured_max_seq_len}:"
    )
    for pool_name in ("balanced_pool", "edge_case_pool"):
        s = real_stats[pool_name]
        print(
            f"  {pool_name}: n={s['n']} min={s['min']} mean={s['mean']} median={s['median']} "
            f"p90={s['p90']} p95={s['p95']} p99={s['p99']} max={s['max']} "
            f"truncated={s['n_truncated']} ({s['truncation_rate']:.2%})"
        )
    print(
        f"Split reality: train={split_report['actual_n_train']} "
        f"val={split_report['actual_n_val']} test(edge)={split_report['actual_n_test_edge_cases']} "
        f"discarded_from_pool={split_report['actual_n_discarded_from_pool']}"
    )

    if overall_truncated:
        print(
            f"FAIL: {overall_truncated} example(s) would be truncated under the real tokenizer "
            f"at max_seq_len={configured_max_seq_len}. Raise data_preparation.max_seq_len in "
            "config.json to cover the measured max."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
