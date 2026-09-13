"""
phase1/labeling/frontier_llm.py

Calls a frontier LLM (Anthropic Claude or OpenAI GPT) to generate cluster labels.
Uses persistent client sessions, explicit temperature controls, and checkpointing.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 10  # save progress every N clusters


# ── Public Entry Point ────────────────────────────────────────────────────────

def run_label_generation(
    cfg: dict,
    clustered_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    business_eval=None,
) -> pd.DataFrame:
    """
    Generate cluster labels from the teacher frontier LLM.
    """
    from phase1.data.schema import (
        CLUSTER_ID, FILE_LABELED_CSV,
        PROMPT_IDS, cluster_name_col,
        top_k_details_col,
    )
    from phase1.prompts.templates import build_messages

    k          = cfg["top_k"]
    llm_cfg    = cfg["teacher_llm"]
    model_id   = llm_cfg["model"]
    provider   = llm_cfg["provider"]
    domain     = cfg["dataset"]["domain"]
    out_dir    = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_csv = out_dir / FILE_LABELED_CSV

    label_cols = [cluster_name_col(model_id, pid) for pid in PROMPT_IDS]
    if labeled_csv.exists():
        labeled_df = pd.read_csv(labeled_csv)
        done_clusters = set(
            labeled_df[CLUSTER_ID][labeled_df[label_cols[0]].notna()].unique()
        )
        logger.info(
            f"[labeling] Resuming — {len(done_clusters)} clusters already labeled."
        )
    else:
        labeled_df = clustered_df.copy()
        for col in label_cols:
            labeled_df[col] = None
        done_clusters = set()

    clusters = grouped_df[CLUSTER_ID].unique()
    todo     = [c for c in clusters if c not in done_clusters]
    logger.info(f"[labeling] Generating labels for {len(todo)} clusters ...")

    # Initialize client ONCE to prevent socket leakage across hundreds of calls
    client = _init_client(provider)

    for i, cluster_id in enumerate(todo):
        cluster_row  = grouped_df[grouped_df[CLUSTER_ID] == cluster_id].iloc[0]
        ticket_texts = json.loads(cluster_row[top_k_details_col(k)])

        row_labels = {}
        for prompt_id in PROMPT_IDS:
            messages = build_messages(prompt_id, ticket_texts, cfg, domain)

            t0 = time.time()
            if provider == "anthropic":
                label = _call_anthropic(client, messages, llm_cfg)
            elif provider == "openai":
                label = _call_openai(client, messages, llm_cfg)
            else:
                raise ValueError(f"Unknown provider '{provider}'.")
            elapsed = time.time() - t0

            if business_eval is not None:
                business_eval.record_label_latency(cluster_id, prompt_id, elapsed)

            row_labels[cluster_name_col(model_id, prompt_id)] = label.strip()
            logger.debug(f"  cluster {cluster_id} | {prompt_id}: {label[:80]}")

            time.sleep(llm_cfg.get("sleep_between_calls", 0.1))

        # Update all tickets assigned to this cluster
        mask = labeled_df[CLUSTER_ID] == cluster_id
        for col, val in row_labels.items():
            labeled_df.loc[mask, col] = val

        # Progress checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
            labeled_df.to_csv(labeled_csv, index=False)
            logger.info(
                f"[labeling] {i + 1}/{len(todo)} clusters done. "
                f"Checkpoint saved to {labeled_csv}"
            )

    labeled_df.to_csv(labeled_csv, index=False)
    logger.info(f"[labeling] All labels saved to {labeled_csv}")
    return labeled_df


# ── Client Initializer ────────────────────────────────────────────────────────

def _init_client(provider: str):
    """Instantiate SDK client once with environment API keys."""
    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")
        return anthropic.Anthropic(api_key=api_key)

    if provider == "openai":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment.")
        return openai.OpenAI(api_key=api_key)

    raise ValueError(f"Unknown provider '{provider}'. Must be 'anthropic' or 'openai'.")


# ── Robust Callers with Top-Level Retry ────────────────────────────────────────

def _is_anthropic_retryable(exc: BaseException) -> bool:
    import anthropic
    return isinstance(exc, (anthropic.RateLimitError, anthropic.APIStatusError))


def _is_openai_retryable(exc: BaseException) -> bool:
    import openai
    return isinstance(exc, (openai.RateLimitError, openai.APIStatusError))


@retry(
    retry=retry_if_exception_type((Exception,)),  # Narrowed below via predicate if needed
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_anthropic(client, messages: list[dict], llm_cfg: dict) -> str:
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs  = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=llm_cfg["model"],
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg.get("temperature", 0.0),  # Explicitly passed
        system=system_msg,
        messages=user_msgs,
    )
    return response.content[0].text


@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_openai(client, messages: list[dict], llm_cfg: dict) -> str:
    response = client.chat.completions.create(
        model=llm_cfg["model"],
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg.get("temperature", 0.0),
        messages=messages,
    )
    return response.choices[0].message.content
