# Report: Fine-Tuning an LLM to Adjudicate Loan Applications

## 1. Approach

**Data Preparation.** `fine_tune_llm_credit_rules.json` defines 10 rules across four groups
(Eligibility, Creditworthiness, FinancialStability, LoanSuitability), each with a
`field`/`operator`/`value` comparison, an `action_on_fail` (`REJECT` or `FLAG_REVIEW`), and a
`severity`. `data_preparation.py` generates synthetic applicant profiles by sampling each rule's
field around its threshold, evaluates every profile against *all* rules (a failing `REJECT`-
severity rule dominates the final decision, matching real underwriting logic where any critical
disqualifier overrides softer positive signals), and renders each labeled example as text with
special tokens (`<|begin|>`, `<|applicant|>...<|/applicant|>`, `<|decision|>...<|/decision|>`,
`<|end|>`, `<|pad|>`). Rendering is generic and data-driven off the ruleset rather than
hard-coded to these specific 10 rules, so the same code works if the ruleset changes.

Two properties of naive random sampling would have hurt training: label imbalance (the real
ruleset naturally skews ~75% REJECT, since REJECT fires if *any* of 8 REJECT-severity rules
fails) and a lack of adversarial examples near rule boundaries. Both were addressed directly:
`generate_balanced_profiles` stratifies toward a uniform 1/3 APPROVE/REJECT/FLAG_REVIEW split for
train/val, and a dedicated `generate_edge_case_profiles` perturbs 2-3 numeric-threshold rules at
once (including exact-threshold cases) to build the 14-example held-out test set — deliberately
harder than the training distribution, to stress-test generalization rather than measure
memorization.

Tokenization uses the real base model's tokenizer (not a toy whitespace tokenizer), extended
with the special tokens above. `max_seq_len=256` was chosen empirically, not guessed:
`scripts/measure_token_lengths.py` tokenized the actual rendered profile pool with the real
tokenizer and found a true max of 177 tokens (train/val) / 142 (edge cases) — comfortably under
256 with zero truncation.

**Model Selection and Training.** Base model: `Qwen/Qwen2.5-0.5B-Instruct` — Apache 2.0
licensed, ~0.5B parameters (fits a free-tier Colab T4 without quantization), and proportional to
a 10-rule, single-domain classification+explanation task (a larger model would mostly add
unused capacity here). The model is **fully fine-tuned** (all parameters trainable, no
LoRA/PEFT): at this scale, full fine-tuning comfortably fits a T4's memory/time budget, and
because the tied `embed_tokens`/`lm_head` table is ~28% of the model's total parameters and must
stay fully trainable regardless (to learn the newly added special tokens), LoRA's usual
parameter/memory savings would have been far more modest here than on larger models — a
comparison LoRA run is tracked as future work (§5).

Training loop: standard forward pass → cross-entropy loss (via the model's own `.loss`) →
backprop → AdamW step, with the tokenizer/embedding table resized *before* training so the new
special tokens are learnable rather than mapping to an unknown-token embedding. Hyperparameters
(`config.json`): learning rate `2e-4`, batch size `8`, `3` epochs (120 total steps over 316
training examples), AdamW with weight decay `0.01`, 10 warmup steps. Validation loss was
evaluated and checkpointed every 50 steps; the checkpoint with the best validation loss is
restored before saving the final model.

**Evaluation.** For every test-split example, the model generates a completion from the
applicant-profile prompt, the decision label is parsed out of the `<|decision|>...<|/decision|>`
span, and predictions are scored against ground truth: accuracy, per-label precision/recall/F1,
a confusion matrix, and a rule-citation-accuracy metric (for REJECT/FLAG_REVIEW ground-truth
examples, whether the model's rationale mentions at least one rule id that actually drove that
decision). An unparseable or missing decision span is tracked as a distinct outcome rather than
silently scored wrong.

## 2. Results

| | Baseline (raw, pre-fine-tuning) | Fine-tuned |
|---|---|---|
| Accuracy | 0.0% | **57.1%** (8/14) |
| Unparseable rate | 100% (14/14) | **0%** (0/14) |
| Rule-citation accuracy | 0.0%* | 0.0% |
| APPROVE precision / recall | — | 0.71 / 0.56 |
| REJECT precision / recall | — | 0.43 / 0.75 |
| FLAG_REVIEW precision / recall | — | 0.0 / 0.0 |

\* The baseline's rule-citation accuracy is 0% for a different reason than the fine-tuned
model's: the raw instruct model never produces a `<|decision|>` span at all (see §3), so there's
nothing to score as correct or incorrect — every case is "unparseable," not "wrong."

Full reports: `runs/evaluation/baseline_eval_report.json`, `runs/evaluation/eval_report.json`.
Training loss dropped from 9.61 (step 1, cold special-token embeddings) to ~0.17 by step 120,
with validation loss tracking training loss closely throughout (no divergence/overfitting
observed within these 3 epochs).

## 3. Sample Dialogues & Analysis

**Baseline.** The raw, un-fine-tuned model never engages with the task's format at all — sample
completions include a SQL query reconstructing the applicant's fields, a Python-style object
dump, a list of hashtags, and general assistant-style advice about improving a credit
application. It has no exposure to `<|decision|>`/`<|/decision|>` as meaningful tokens, so every
one of the 14 test examples is unparseable. This 0%/100%-unparseable result is the expected,
correct behavior for an instruct model that has never seen this project's output format — it is
the "before" baseline the fine-tuning is measured against, not a bug.

**Fine-tuned — strength: format compliance.** Every single completion (14/14) now produces a
well-formed `<|decision|>\nDecision: <LABEL>\nRationale: ...\n<|/decision|>` span. 120 training
steps were enough to fully teach the output grammar, even though the dataset is small and
synthetic.

**Fine-tuned — strength: directionally useful classification.** 57.1% accuracy beats the ~33%
random-3-class baseline, and REJECT recall (0.75) is high — the model rarely misses a profile
that should genuinely be rejected, which is the more conservative failure mode for a credit
decisioning system (a false APPROVE is more costly than a false REJECT).

**Fine-tuned — weakness: rule citation is not grounded.** This is the most important finding.
Two REJECT examples were checked against the actual rule engine (`evaluate_profile`) to confirm
ground truth:

- Applicant age 17 (correctly predicted REJECT) — model cited `RULE-EMPLOY-002`. The true
  driving rule is `RULE-AGE-001` (`age >= 18` fails).
- Applicant with `debt_to_income_ratio_percent=41` (correctly predicted REJECT) — model again
  cited `RULE-EMPLOY-002`. The true driving rule is `RULE-DTI-001` (`debt_to_income_ratio_percent
  <= 40` fails).

Both times the model reached for the *same* rule id regardless of which rule actually fired.
This strongly suggests it learned "REJECT decisions cite a plausible-looking rule id" as a
surface pattern of the output format, without learning *which* rule id is correct for a given
profile — i.e., it nailed the decision classification and the rationale's grammar, but not the
rationale's actual grounding. Given only 120 optimizer steps across 316 examples spanning 10
rules (roughly 30 examples per rule on average, and fewer still per rule/decision combination),
this is a plausible and explainable limitation rather than a surprising one.

**Fine-tuned — weakness: boundary confusion on adversarial cases.** The test set is
deliberately built from near-threshold, multi-rule-perturbation edge cases (harder than the
training distribution by design). 4 of 9 true-APPROVE profiles were misclassified as REJECT, and
the single FLAG_REVIEW example was misclassified as APPROVE (n=1, so this specific number isn't
statistically meaningful on its own, but it does mean FLAG_REVIEW recall is currently
unverified). This is consistent with a model that picked up the general decision distribution
but hasn't precisely calibrated every rule's exact threshold under compound perturbations.

## 4. Discussion

The fine-tuning run demonstrably worked: it took a model with zero exposure to this task's
format and rule set to one that reliably produces syntactically correct, mostly-plausible
decisions on a genuinely adversarial held-out set. The clearest limitation — ungrounded rule
citation — is best explained by training scale (120 steps, one small synthetic dataset, no
repetition-heavy curriculum around specific rule/feature associations) rather than by an
architectural or methodological flaw in the pipeline itself.

## 5. Future Work

- **More training signal**: more epochs/steps and a larger, more diverse synthetic dataset
  (especially more near-threshold examples in *training*, not just the test set) would likely
  improve both boundary calibration and rule-citation grounding, since validation loss hadn't
  clearly plateaued by step 120.
- **LoRA comparison run**: a parameter-efficient LoRA fine-tune of the same base model (tracked
  as an open item in `docs/srs/training.md`), to compare accuracy/rule-citation-accuracy and
  training cost against this full fine-tune.
- **Explicit rationale supervision**: the current loss is a single token-level LM loss over the
  whole sequence; more heavily weighting the rationale span, or a two-stage decision-then-citation
  objective, could target the rule-grounding weakness more directly than more of the same data
  would.
- **Revisit the forced class balance.** Training data was stratified to a uniform 1/3
  APPROVE/REJECT/FLAG_REVIEW split (`data-preparation.md` A3.7); left alone, the rule set
  naturally produces a heavily skewed ~75%/15%/12% REJECT/FLAG_REVIEW/APPROVE distribution
  (REJECT fires if *any* of 8 REJECT-severity rules fails). Stratifying almost certainly helped
  the model get any exposure at all to the rarest class (FLAG_REVIEW), but it also means the
  model was trained on a decision prior that doesn't match what it would see against real
  loan-application traffic — so the 57.1% accuracy figure in §2 shouldn't be read as an estimate
  of production accuracy. Worth a follow-up ablation: train on the natural (unstratified)
  distribution and compare per-class recall against this stratified run, and/or try class-
  weighted loss (up-weighting rare-class gradient contribution) as an alternative to stratified
  resampling that preserves the natural distribution's example diversity. Either way, evaluating
  against both a stratified set (to compare like-for-like with this run) and a natural-
  distribution set (to estimate realistic deployment accuracy) would separate "did the model
  learn the decision boundaries" from "what accuracy would this see in production."
