## Qwen2.5-1.5B-Instruct Experiment Results

### Setup
Student model: Qwen/Qwen2.5-1.5B-Instruct
Fine-tuning method: QLoRA (4-bit, nf4, double quant)
GPU: Tesla T4
LoRA rank: 16
LoRA alpha: 16
LoRA dropout: 0.05
Learning rate: 2e-4 (cosine schedule, 5% warmup)
Max sequence length: 2048
Effective batch size: 16 (4 x 4 gradient accumulation)

### Dataset / Training
Source: Bitext, n_samples=500 (497 tickets after clustering, 21 clusters)
Cluster split: 14 train / 3 val / 4 test
Training examples: 70
Validation examples: 15
Test examples: 20
Training epochs: 3
Training steps: 15
Validation loss by epoch: 1.816 / 1.476 / 1.398
Best checkpoint: epoch 3 (checkpoint-15)
Best validation loss: 1.398

### Training Result
Training completed in 1.2 min
Final training loss: 1.861
Adapter saved to Drive: outputs/20260916_0107_Qwen2.5-1.5B-Instruct_ep3/models/lora_adapter

### Evaluation (test split, 4 clusters x 5 prompts = 20 predictions)

Cosine similarity
| Model | Same prompt | Multi-ref max |
|---|---|---|
| Baseline | 0.7895 | 0.8669 |
| Fine-tuned | 0.7848 | 0.8600 |
Change: -0.0047
Fine-tuned better on [x]/20 examples, baseline better on [y]/20

LLM-as-a-judge (Claude Haiku 4.5, reference mode)
| Model | Faithfulness | Specificity | Equivalence | Composite |
|---|---|---|---|---|
| Baseline | 4.35 | 3.35 | 3.35 | 3.68 |
| Fine-tuned | 4.40 | 3.45 | 3.40 | 3.75 |
Fine-tuned wins: [a], Baseline wins: [b], Ties: [c]

Inference latency: 0.487s/label baseline, 0.410s/label fine-tuned (1.8x faster than teacher)

### Conclusion
The Qwen2.5-1.5B baseline already produces usable theme labels without fine-tuning (3.68/5), unlike SmolLM2-360M, which copied ticket text verbatim. Fine-tuning changed the judge score by +0.07 and cosine similarity by -0.005, so the adapter had no measurable effect at this configuration. Validation loss was still falling at epoch 3 after only 15 steps, so the run is undertrained. The peft "multiple adapters" warning fired during fine-tuned inference; whether the trained adapter is actually active is unresolved [update after checking labels_by_cluster.csv].

### Reproducibility
Run ID: 20260916_0107_Qwen2.5-1.5B-Instruct_ep3
Branch: BTT_WWA_TEST
Config diff from default: model_id only
Adapter and checkpoints stored on Drive, not GitHub
