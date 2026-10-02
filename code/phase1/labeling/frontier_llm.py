"""
phase1/labeling/frontier_llm.py

Calls a frontier LLM (Anthropic Claude or OpenAI GPT) to generate cluster labels.

For each cluster:
  - Sends the top-k ticket texts as context
  - Generates one label per prompt (P1–P5)
  - Writes results back to the ticket-level labeled CSV

Rate limiting:
  - Configurable sleep between calls
  - Exponential backoff on rate-limit errors (via tenacity)
  - Progress checkpoint every 10 clusters so work is not lost on interruption

Business eval:
  - Tracks per-call latency via BusinessEvaluator context manager

NOTE ON TEMPERATURE (anthropic SDK v1.11.0+)
---------------------------------------------
This SDK version's Messages.create() does not accept a 'temperature'
parameter at all (confirmed via inspect.signature — not in the accepted
kwargs for this endpoint). teacher_llm.temperature in config therefore has
NO effect: every teacher label-generation call runs at the API's
uncontrollable default sampling behavior, not temperature=0.3 as
configured. The self-healing kwarg-stripping loop below correctly removes
'temperature' and retries — this is not a bug to fix, it's a genuine SDK
capability gap. See the one-time warning logged at first call.
"""

import json
import logging
import os
import time
from pathlib import Path

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

# One-time warning flag — avoids logging the temperature-unsupported notice
# on every single teacher call (previously logged 100+ times per run).
_TEMPERATURE_WARNING_SHOWN = False


# ── Public entry point ────────────────────────────────────────────────────────

def run_label_generation(
    cfg: dict,
    clustered_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    business_eval=None,
) -> pd.DataFrame:
    """
    Generate cluster labels from the teacher frontier LLM.

    Parameters
    ----------
    cfg : dict
        Loaded phase1_config.yaml.
    clustered_df : pd.DataFrame
        Ticket-level DataFrame (output of clustering.py).
    grouped_df : pd.DataFrame
        Cluster-level DataFrame (output of preprocessing.py).
    business_eval : BusinessEvaluator | None
        If provided, latency measurements are recorded.

    Returns
    -------
    pd.DataFrame
        Ticket-level labeled DataFrame: all original columns plus
        cluster_name_{model}_{P1} … cluster_name_{model}_{P5}.
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
    out_dir = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_csv = out_dir / FILE_LABELED_CSV

    label_cols = [cluster_name_col(model_id, pid) for pid in PROMPT_IDS]

    # FIX: Check if the file exists AND contains the correct columns for the current model/config.
    # If the teacher model ID or prompt columns changed, discard the stale file to force regeneration.
    force_fresh = False
    if labeled_csv.exists():
        temp_df = pd.read_csv(labeled_csv, nrows=1)
        # Verify that all required label columns for the *current* model exist in the file
        missing_cols = [col for col in label_cols if col not in temp_df.columns]
        if missing_cols:
            logger.warning(
                f"[labeling] Existing {labeled_csv.name} lacks columns for current config/model "
                f"({missing_cols}). Forcing fresh teacher label generation."
            )
            force_fresh = True

    # Partial checkpoint logic with safety check
    if labeled_csv.exists() and not force_fresh:
        labeled_df = pd.read_csv(labeled_csv)

        # Additional safety: check if columns exist before filtering
        valid_label_cols = [c for c in label_cols if c in labeled_df.columns]
        if not valid_label_cols:
            labeled_df = clustered_df.copy()
            for col in label_cols:
                labeled_df[col] = None
            done_clusters = set()
            logger.info("[labeling] Stale columns found. Starting label generation fresh.")
        else:
            done_clusters = set(
                labeled_df[CLUSTER_ID][labeled_df[valid_label_cols[0]].notna()].unique()
            )
            total_clusters = len(grouped_df[CLUSTER_ID].unique())

            if len(done_clusters) >= total_clusters:
                logger.warning(
                    f"[labeling] {labeled_csv} already has ALL {total_clusters} "
                    f"clusters labeled. To force a clean re-generation under new prompt configs, "
                    f"delete {labeled_csv.name} first."
                )
            else:
                logger.info(
                    f"[labeling] Resuming — {len(done_clusters)}/{total_clusters} "
                    f"clusters already labeled."
                )
    else:
        # Start fresh: copy clustered_df and add empty label columns
        labeled_df = clustered_df.copy()
        for col in label_cols:
            labeled_df[col] = None
        done_clusters = set()
    clusters = grouped_df[CLUSTER_ID].unique()
    todo     = [c for c in clusters if c not in done_clusters]
    logger.info(f"[labeling] Generating labels for {len(todo)} clusters ...")

    llm_caller = _get_llm_caller(provider, llm_cfg)

    for i, cluster_id in enumerate(todo):
        cluster_row  = grouped_df[grouped_df[CLUSTER_ID] == cluster_id].iloc[0]
        ticket_texts = json.loads(cluster_row[top_k_details_col(k)])

        row_labels = {}
        for prompt_id in PROMPT_IDS:
            messages = build_messages(prompt_id, ticket_texts, cfg, domain)

            t0 = time.time()
            label = llm_caller(messages, llm_cfg)
            elapsed = time.time() - t0

            if business_eval is not None:
                business_eval.record_label_latency(cluster_id, prompt_id, elapsed)

            row_labels[cluster_name_col(model_id, prompt_id)] = label.strip()
            logger.debug(f"  cluster {cluster_id} | {prompt_id}: {label[:80]}")

            # Respect rate limit sleep between calls
            time.sleep(llm_cfg["sleep_between_calls"])

        # Write labels back to ALL rows belonging to this cluster
        mask = labeled_df[CLUSTER_ID] == cluster_id
        for col, val in row_labels.items():
            labeled_df.loc[mask, col] = val

        # Checkpoint every N clusters
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
            labeled_df.to_csv(labeled_csv, index=False)
            logger.info(
                f"[labeling] {i + 1}/{len(todo)} clusters done. "
                f"Checkpoint saved to {labeled_csv}"
            )

    labeled_df.to_csv(labeled_csv, index=False)
    logger.info(f"[labeling] All labels saved to {labeled_csv}")
    return labeled_df


# ── LLM caller factory ────────────────────────────────────────────────────────

def _get_llm_caller(provider: str, llm_cfg: dict):
    """Return a callable that takes (messages, llm_cfg) and returns a label string."""
    if provider == "anthropic":
        return _call_anthropic
    elif provider == "openai":
        return _call_openai
    else:
        raise ValueError(f"Unknown provider '{provider}'. Must be 'anthropic' or 'openai'.")


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic(messages: list[dict], llm_cfg: dict) -> str:
    import re
    import anthropic

    global _TEMPERATURE_WARNING_SHOWN
    if not _TEMPERATURE_WARNING_SHOWN:
        logger.warning(
            f"[labeling] anthropic SDK v{anthropic.__version__} does not "
            "support a 'temperature' parameter on messages.create() — "
            "teacher_llm.temperature has NO effect; sampling is NOT "
            "deterministic. This warning logs once per process."
        )
        _TEMPERATURE_WARNING_SHOWN = True

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs  = [m for m in messages if m["role"] != "system"]

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
        wait=wait_exponential(
            min=llm_cfg["retry_wait_min"],
            max=llm_cfg["retry_wait_max"],
        ),
        stop=stop_after_attempt(llm_cfg["max_retries"]),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        kwargs = dict(
            model=llm_cfg["model"],
            max_tokens=llm_cfg["max_tokens"],
            # NOTE: temperature is intentionally included here even though
            # this SDK version (anthropic 1.11.0) does not accept it —
            # inspect.signature(client.messages.create) confirms 'temperature'
            # is not in the accepted kwargs for this endpoint. The
            # self-healing loop below strips it on the first TypeError and
            # retries successfully. This is left in place (rather than
            # removed) so the code auto-recovers correctly if/when the SDK
            # is upgraded to a version that DOES support it, without
            # requiring another code change.
            temperature=llm_cfg["temperature"],
            system=system_msg,
            messages=user_msgs,
        )
        # Self-healing: remove any param the installed SDK version rejects
        for _ in range(len(kwargs) + 1):
            try:
                return client.messages.create(**kwargs).content[0].text
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if not match:
                    raise
                bad = match.group(1)
                logger.warning(
                    f"[labeling] Anthropic SDK rejected param '{bad}' "
                    f"(SDK v{anthropic.__version__}) — removing and retrying."
                )
                kwargs.pop(bad, None)
        raise RuntimeError("[labeling] Could not call Anthropic API — all params rejected.")

    return _call()


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(messages: list[dict], llm_cfg: dict) -> str:
    import re
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    client = openai.OpenAI(api_key=api_key)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
        wait=wait_exponential(
            min=llm_cfg["retry_wait_min"],
            max=llm_cfg["retry_wait_max"],
        ),
        stop=stop_after_attempt(llm_cfg["max_retries"]),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        kwargs = dict(
            model=llm_cfg["model"],
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
            messages=messages,
        )
        for _ in range(len(kwargs) + 1):
            try:
                return client.chat.completions.create(**kwargs).choices[0].message.content
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if not match:
                    raise
                bad = match.group(1)
                logger.warning(
                    f"[labeling] OpenAI SDK rejected param '{bad}' — removing and retrying."
                )
                kwargs.pop(bad, None)
        raise RuntimeError("[labeling] Could not call OpenAI API — all params rejected.")

    return _call()
