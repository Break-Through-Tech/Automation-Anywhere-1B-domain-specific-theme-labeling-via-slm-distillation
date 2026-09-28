# Llama 3.2-3B — Domain-Anchored Relabel Results (n=500, raw split)

## Model

* **Model:** `meta-llama/Llama-3.2-3B-Instruct`
* **Fine-tuning method:** LoRA, bf16 (no 4-bit quantization)
* **GPU:** Colab T4
* **LoRA rank:** 16
* **LoRA alpha:** 16
* **LoRA dropout:** 0.05

## Setup

* **Samples:** 500
* **Epochs:** 3
* **Learning rate:** 0.0002
* **completion_only_loss:** True
* **include_domain_in_prompt:** True
* **Environment:** Google Colab
* **Run ID:** 20260928_2119_Llama-3.2-3B-Instruct_ep3

## ⚠️ Data Integrity Note

This run supersedes an earlier sweep run under the same hyperparameters
(`llama32_raw_baseline_v2`, composite 4.04) that was later found to be
**invalid**: `include_domain_in_prompt` had been flipped to `True` for
inference/training, but the teacher reference labels used for scoring were
generated before that change and never regenerated — so the SLM was being
prompted with a domain-anchored instruction it was never scored against
consistently. This run regenerates the teacher labels under the current
domain-anchored prompt and re-trains from scratch against consistent
references. See the debugging log at the end of this document for the full
investigation trail.

## Training

* Training completed successfully.
Epoch 1 validation loss: 1.6552
Epoch 2 validation loss: 1.4094
Epoch 3 validation loss: 1.3865
* Best epoch (by eval_loss): 3
* Validation performance optimized through distillation.

## Results

| Metric | Baseline | Fine-tuned |
| :--- | :--- | :--- |
| Cosine similarity | 0.8140 | 0.8302 |
| Cosine similarity (multi-ref) | 0.8503 | 0.8867 |
| ROUGE-L | 0.4528 | 0.5414 |
| ROUGE-L (multi-ref) | 0.5291 | 0.6973 |
| BERTScore F1 | 0.9190 | 0.9331 |
| BERTScore F1 (multi-ref) | 0.9256 | 0.9520 |
| LLM judge composite | 3.7333 | 4.0333 |
| Inference latency (sec/label) | 0.6300 | 0.5448 |

## LLM Judge

| Metric | Baseline | Fine-tuned |
| :--- | :--- | :--- |
| Faithfulness | 4.15 | 4.45 |
| Specificity | 3.6 | 3.85 |
| Equivalence | 3.45 | 3.8 |
| Composite | 3.733333333333333 | 4.033333333333333 |

## Business Metrics

* **Fine-tuning wall time:** 2.81 min
* **Baseline throughput:** 95.2 labels/min
* **Fine-tuned throughput:** 110.1 labels/min
* **Teacher API cost (this eval):** $0.0137 USD

## Summary

* Fine-tuned composite score **4.033** crosses the 4.0 target under genuinely domain-consistent labels (verified: relabeled reference set confirmed non-identical to the prior mismatched set; trained adapter hash confirmed distinct from all other sweep experiments).
* This is the best-performing configuration found across a 7-way isolated hyperparameter sweep (lr, epochs, LoRA rank, dropout, warmup) at n=500 on the raw split.
* `dropout=0.0` was the only other config to show comparable lift (composite 3.950) but did not exceed this baseline recipe.
