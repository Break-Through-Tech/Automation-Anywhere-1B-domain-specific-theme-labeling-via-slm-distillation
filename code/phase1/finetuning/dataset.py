"""
phase1/finetuning/dataset.py

Converts the labeled cluster CSV into JSONL files for LoRA/QLoRA fine-tuning.

Split strategy: cluster-level (never ticket-level) to prevent data leakage.
One training example = (cluster, prompt_id) pair → 5 examples per cluster.
"""

import json
import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def build_dataset(cfg: dict, labeled_df: pd.DataFrame, tokenizer) -> dict:
    """
    Build train / val / test JSONL files from the labeled DataFrame.

    Parameters
    ----------
    cfg : dict
        Loaded phase1_config.yaml.
    labeled_df : pd.DataFrame
        Ticket-level labeled DataFrame with cluster_name_* columns.
    tokenizer : transformers.PreTrainedTokenizer
        The student SLM tokenizer (needed for apply_chat_template).

    Returns
    -------
    dict with keys "train", "val", "test" each containing the path (str)
    to the corresponding JSONL file.
    """
    from phase1.data.schema import (
        CLUSTER_ID, TICKET_RANK, TICKET_DETAILS, TICKET_ID,
        PROMPT_IDS, cluster_name_col,
        FILE_TRAIN_JSONL, FILE_VAL_JSONL, FILE_TEST_JSONL,
        top_k_details_col,
    )
    from phase1.prompts.templates import build_messages

    k          = cfg["top_k"]
    model_id   = cfg["teacher_llm"]["model"]
    domain     = cfg["dataset"]["domain"]
    split_cfg  = cfg["split"]
    out_dir    = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cluster-level split ───────────────────────────────────────────────────
    all_clusters = sorted(labeled_df[CLUSTER_ID].unique().tolist())
    random.seed(cfg["seed"])
    random.shuffle(all_clusters)

    n        = len(all_clusters)
    n_train  = int(n * split_cfg["train"])
    n_val    = int(n * split_cfg["val"])

    train_clusters = set(all_clusters[:n_train])
    val_clusters   = set(all_clusters[n_train : n_train + n_val])
    test_clusters  = set(all_clusters[n_train + n_val :])

    logger.info(
        f"[dataset] Cluster split — "
        f"train: {len(train_clusters)}, "
        f"val: {len(val_clusters)}, "
        f"test: {len(test_clusters)}"
    )

    # ── Build top-k ticket lookup per cluster ─────────────────────────────────
    # Each cluster → ordered list of top-k ticket texts
    top_k_df = labeled_df[labeled_df[TICKET_RANK] <= k].copy()
    cluster_tickets: dict[int, list[str]] = {}
    for cid, grp in top_k_df.groupby(CLUSTER_ID):
        grp_sorted = grp.sort_values(TICKET_RANK)
        cluster_tickets[cid] = grp_sorted[TICKET_DETAILS].tolist()

    # ── Build per-cluster labels lookup ──────────────────────────────────────
    cluster_labels: dict[int, dict[str, str]] = {}
    for cid in all_clusters:
        row = labeled_df[labeled_df[CLUSTER_ID] == cid].iloc[0]
        cluster_labels[cid] = {
            pid: str(row[cluster_name_col(model_id, pid)])
            for pid in PROMPT_IDS
            if pd.notna(row[cluster_name_col(model_id, pid)])
        }

    # ── Write JSONL files ─────────────────────────────────────────────────────
    max_seq = cfg["student_slm"]["max_seq_length"]
    paths   = {}
    splits  = {
        "train": (train_clusters, FILE_TRAIN_JSONL),
        "val":   (val_clusters,   FILE_VAL_JSONL),
        "test":  (test_clusters,  FILE_TEST_JSONL),
    }

    for split_name, (cluster_set, filename) in splits.items():
        examples = _build_examples(
            cluster_set, cluster_tickets, cluster_labels,
            cfg, tokenizer, max_seq, domain,
        )
        path = out_dir / filename
        _write_jsonl(examples, path)
        paths[split_name] = str(path)
        logger.info(
            f"[dataset] {split_name}: {len(examples)} examples → {path}"
        )

    return paths


def load_split_clusters(cfg: dict, labeled_df: pd.DataFrame) -> dict:
    """
    Return the cluster IDs for each split (needed by evaluation step).
    Uses the same seed as build_dataset so splits are consistent.
    """
    from phase1.data.schema import CLUSTER_ID

    all_clusters = sorted(labeled_df[CLUSTER_ID].unique().tolist())
    random.seed(cfg["seed"])
    random.shuffle(all_clusters)

    n        = len(all_clusters)
    n_train  = int(n * cfg["split"]["train"])
    n_val    = int(n * cfg["split"]["val"])

    return {
        "train": set(all_clusters[:n_train]),
        "val":   set(all_clusters[n_train : n_train + n_val]),
        "test":  set(all_clusters[n_train + n_val :]),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_examples(
    cluster_set: set,
    cluster_tickets: dict,
    cluster_labels: dict,
    cfg: dict,
    tokenizer,
    max_seq: int,
    domain: str,
) -> list[dict]:
    """Build instruction-following examples for a set of clusters."""
    from phase1.data.schema import PROMPT_IDS, cluster_name_col
    from phase1.prompts.templates import build_messages

    model_id = cfg["teacher_llm"]["model"]
    examples = []
    skipped  = 0

    for cid in sorted(cluster_set):
        if cid not in cluster_tickets or cid not in cluster_labels:
            continue
        ticket_texts = cluster_tickets[cid]
        labels       = cluster_labels[cid]

        for prompt_id in PROMPT_IDS:
            if prompt_id not in labels:
                continue

            label    = labels[prompt_id]
            messages = build_messages(prompt_id, ticket_texts, cfg, domain)

            # Append the assistant turn (the gold label) for training
            full_messages = messages + [{"role": "assistant", "content": label}]

            # Apply model's chat template
            text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Length check (skip if too long)
            token_len = len(tokenizer.encode(text))
            if token_len > max_seq:
                skipped += 1
                continue

            examples.append({
                "text":       text,
                "cluster_id": int(cid),
                "prompt_id":  prompt_id,
            })

    if skipped > 0:
        logger.warning(
            f"[dataset] Skipped {skipped} examples exceeding max_seq_length={max_seq}."
        )

    return examples


def _write_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
