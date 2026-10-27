# Qwen3-8B saved-output reference-free evaluation

Re-evaluated saved September 22 outputs on October 27, 2026. No retraining or new student inference was performed.

Original run: `20260922T214033Z_qwen3-8b-e5-lr1e4_a32b189c/runs/20260922_2140_Qwen3-8B_ep5`.
Student: Qwen3-8B, 4-bit QLoRA, 5 epochs, learning rate 1e-4, LoRA rank/alpha 16, dropout 0.05. Original setup used a T4 and FP16 compute.

## Method

The standalone notebook in `notebooks/03_qwen3_reference_free.ipynb` evaluates teacher, baseline and fine-tuned labels individually against the cluster's top 10 tickets. Candidate model identity and teacher reference are omitted. The rubric matches `JUDGE_PROMPT_REFERENCE_FREE` from the team's qwen2.5-3b branch (`llm_judge.py`, content blob `08ba2b2093cdbece249c5d698b25bb2fd947e61f`).

Judge request model: claude-haiku-4-5; returned model: claude-haiku-4-5-20251001.
Scores cover faithfulness, specificity and coherence (1–5); composite is their arithmetic mean. There are 60 judgments: 4 clusters × 5 prompts × 3 models. New scores are separate from the original teacher-reference evaluation.

## Results

| Model | Faithfulness | Specificity | Coherence | Composite |
|---|---:|---:|---:|---:|
| Teacher | 5.00 | 4.10 | 5.00 | 4.700 |
| Baseline | 5.00 | 4.00 | 5.00 | 4.667 |
| Fine-tuned | 4.90 | 4.15 | 5.00 | 4.683 |

Mean fine-tuning lift: +0.0167 composite points. Per-cluster lift: cluster 0 = 0; cluster 3 = 0; cluster 8 = +0.200; cluster 20 = -0.1333.

Cluster 8 gains reflect more specific PIN recovery/reset labels. Cluster 20 losses are two fine-tuned faithfulness scores of 4 rather than 5 (P2 and P4). Their explanations still describe the labels as accurate; P4 only notes concision, and P2 identifies no concrete defect. These rationales do not establish an actual factual regression.

## Limits and next work

Only four independent clusters were evaluated; five prompts per cluster are related observations. This set was already examined during hyperparameter selection. Coherence is saturated at 5. The small numerical differences do not establish fine-tuning benefit, teacher superiority, equivalence, or production readiness. Claude also generated the teacher labels, so reference-free judging does not eliminate every source of evaluator bias.

The original 4.20/5 teacher-reference score uses a different rubric and is not directly comparable. No inference latency was measured here.

Next: freeze a larger dataset and cluster-level split, tune on validation data, and evaluate the selected setup once on untouched test clusters. Add human review of disputed outputs and measure deployment latency separately.

## Files

- `judge_scores.csv`: 60 raw judgments, labels, explanations and response metadata.
- `summary.csv`: per-model means.
- `paired_by_cluster.csv`: paired composite comparisons.
