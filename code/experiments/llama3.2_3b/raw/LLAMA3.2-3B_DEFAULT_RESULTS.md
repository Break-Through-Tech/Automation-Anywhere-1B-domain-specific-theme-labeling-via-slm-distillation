Llama 3.2-3B Phase 1 Smoke Test Results

## Model

* **Model:** `meta-llama/Llama-3.2-3B-Instruct`
* **Fine-tuning method:** 16-bit QLoRA
* **GPU:** colab
* **LoRA rank:** 16
* **LoRA alpha:** 16

## Setup

* **Samples:** 500
* **Epochs:** 3
* **Environment:** Google Colab
* **Fine-tuning:** QLoRA
* **Run ID:** 20260925_0237_Llama-3.2-3B-Instruct_ep3

## Training

* Training completed successfully.
Epoch 1 validation loss: 1.9105
Epoch 2 validation loss: 1.3960
Epoch 3 validation loss: 1.2765
* Best epoch: 3
* Validation performance optimized through distillation.

## Results

| Metric | Baseline | Fine-tuned |
| :--- | :--- | :--- |
| Cosine similarity | 0.8044 | 0.8160 |
| Cosine similarity (multi-ref) | 0.8586 | 0.8813 |
| ROUGE-L | 0.4429 | 0.4442 |
| ROUGE-L (multi-ref) | 0.5400 | 0.5640 |
| BERTScore F1 | 0.9096 | 0.9079 |
| BERTScore F1 (multi-ref) | 0.9232 | 0.9282 |
| LLM judge composite | 3.75 | 3.87 |
| Inference latency (sec/label) | 0.799 | 0.733 |

## LLM Judge

| Metric | Baseline | Fine-tuned |
| :--- | :--- | :--- |
| Equivalence | 3.40 | 3.60 |
| Faithfulness | 4.35 | 4.50 |
| Specificity | 3.50 | 3.50 |
| Composite | 3.75 | 3.87 |

## Summary

* Fine-tuning produced positive improvements across LLM judge alignment dimensions (faithfulness, equivalence, and composite score).
* The fine-tuned model improved inference speed from approximately 0.799 seconds per label down to 0.733 seconds per label.
* Overall, the Llama 3.2-3B evaluation and pipeline completed successfully.
