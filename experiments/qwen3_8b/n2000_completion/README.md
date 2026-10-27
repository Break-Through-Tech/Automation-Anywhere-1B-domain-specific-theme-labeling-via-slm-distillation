# Qwen3-8B: 2,000-ticket completion-only experiment

## Setup

Sampled 2,000 unique Bitext support tickets, excluding exact normalized ticket texts from the earlier experiment. Clustering retained 1,927 tickets in 73 clusters. Clusters were frozen into 43 train, 14 validation, and 16 test clusters. Each cluster has five prompt variants: 215 training, 70 validation, and 80 test examples. Top 10 representative tickets provide context.

Qwen3-8B was fine-tuned with 4-bit NF4 QLoRA on a Tesla T4, FP16 compute, learning rate 1e-4, rank/alpha 16, dropout .05, batch size 1, gradient accumulation 16, and sequence limit 1024. Prompt tokens and padding were masked from loss. Non-thinking prompts were used; there was no augmentation. Training ran for five epochs (70 optimizer steps) and took 34.33 minutes. Lowest validation loss: 0.966745 at step 50, about 3.60 epochs. The code was configured to restore the checkpoint with lowest validation loss; the saved training_complete.json has not yet been inspected independently.

## Reference-free results

Claude evaluated each saved label independently against the cluster tickets on faithfulness, specificity and coherence (1-5). Model identity and teacher reference were omitted. There are 450 judgments: 30 validation/test clusters x five prompts x three models.

| Test metric | Baseline | Fine-tuned | Teacher |
|---|---:|---:|---:|
| Faithfulness | 4.9375 | 4.7125 | 4.8875 |
| Specificity | 4.0125 | 3.8875 | 4.0750 |
| Coherence | 5.0000 | 4.9500 | 4.9875 |
| Composite | 4.6500 | 4.5167 | 4.6500 |

Test fine-tuning lift versus baseline: -0.1333 composite points. Cluster-bootstrap 95% interval: [-0.2708, -0.0250], using 10,000 paired cluster resamples. Two test clusters improved, six tied and eight worsened. Validation lift was -0.0667, with interval [-0.2381, 0.0810].

## Interpretation and examples

This configuration did not improve baseline Qwen3 under the current evaluation. The judge flags labels that broaden account-key recovery to include password reset, omit freemium qualifiers, or add upgrade/downgrade terminology. Some validation labels become repetitive. Original ticket context and teacher targets should be checked before accepting every judge explanation, particularly assumptions about whether freemium represents a downgrade.

Late validation loss increased slightly while training loss generally fell, consistent with some late overfitting. The best-checkpoint setting means that using the last epoch was not the intended selection rule. Validation token loss does not directly optimize reference-free label quality.

## Limits

Only 16 test clusters are independent comparison units; five prompts for a cluster are related. The bootstrap does not include repeated-training or repeated-judge variability. Claude supplied both teacher labels and judge scores, so evaluator bias remains possible. Clustering was performed on the full sample before the supervised cluster split; this is a transductive protocol, not an independently collected intent-generalization benchmark. Dataset size, cleaning, training objective, split and software differed from September, preventing attribution to one change. Earlier teacher-reference scores are not directly comparable.

This test set has now been inspected. Further tuning should use validation data, with a fresh holdout for final confirmation. Do not claim equivalence with the teacher or general inferiority of fine-tuning from this single run.

## Files and verification

- summary.csv: model means by validation/test split.
- paired_by_cluster.csv: paired cluster composite comparisons.
- cluster_bootstrap.csv: paired cluster-bootstrap intervals.
- judge_scores.csv: raw labels, scores, reasoning, and judge response metadata.
- training_history.csv: train/validation history and timing.

All 450 judgment rows were checked for unique model/cluster/prompt keys, valid scores and composite arithmetic. Aggregates, per-cluster means and the supplied bootstrap intervals were reproduced from the submitted files. No model was retrained during analysis.

The executed notebook, saved adapter metadata and complete data manifests are not in this package. Preserve them in the experiment's Drive folder. Next audit: teacher_labels.jsonl and training_complete.json before another training run.
