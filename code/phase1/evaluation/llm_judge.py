"""
phase1/evaluation/llm_judge.py

LLM-as-judge evaluation with two modes:

  "reference" (default)
      Judge receives the cluster tickets, the teacher reference label, and the
      candidate label.  Rates: Faithfulness, Specificity, Equivalence.
      For the teacher tag the API is never called — scores are hardcoded to
      5/5/5 (the teacher is trivially equivalent to itself) and the saving is
      done to keep all pipeline outputs consistent.

  "reference_free"
      Judge receives only the cluster tickets and the candidate label — no
      reference label.  Rates: Faithfulness, Specificity, Coherence.
      Useful for comparing all three models on the same intrinsic scale without
      the reference creating a circular-evaluation bias for the teacher.
      Set in configs/phase1_config.yaml → evaluation.judge_mode: "reference_free"

Both modes apply to all three tags (teacher / baseline / finetuned).
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

JUDGE_PROMPT_WITH_REFERENCE = """\
You are evaluating the quality of a label generated for a cluster of IT/HR/CX support tickets.

Below are the top support tickets from the cluster, the reference label (from a frontier LLM), \
and the candidate label (from a smaller model being evaluated).

Rate the CANDIDATE LABEL on these three dimensions, each on a scale of 1–5:
1. Faithfulness (1–5): Does the candidate label accurately reflect what these tickets are about?
2. Specificity (1–5): Is the candidate specific enough to distinguish this cluster? (not too vague, not too narrow)
3. Equivalence (1–5): How similar in meaning is the candidate to the reference label?

Scoring guide:  5 = excellent  |  4 = good  |  3 = acceptable  |  2 = poor  |  1 = wrong

--- TICKETS ---
{tickets}

--- REFERENCE LABEL ---
{reference}

--- CANDIDATE LABEL ---
{candidate}

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{"faithfulness": <int>, "specificity": <int>, "equivalence": <int>, "reasoning": "<one sentence>"}}
"""

JUDGE_PROMPT_REFERENCE_FREE = """\
You are evaluating the quality of a label generated for a cluster of IT/HR/CX support tickets.

Below are the top support tickets from the cluster and the candidate label to evaluate.

Rate the CANDIDATE LABEL on these three dimensions, each on a scale of 1–5:
1. Faithfulness (1–5): Does the label accurately reflect the dominant theme of these tickets?
2. Specificity (1–5): Is the label specific enough to distinguish this cluster from others?
3. Coherence (1–5): Is the label clearly written, well-formed, and professional?

Scoring guide:  5 = excellent  |  4 = good  |  3 = acceptable  |  2 = poor  |  1 = wrong

--- TICKETS ---
{tickets}

--- CANDIDATE LABEL ---
{candidate}

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{"faithfulness": <int>, "specificity": <int>, "coherence": <int>, "reasoning": "<one sentence>"}}
"""

# Dimension names per mode (used for saving and summarising)
DIMS_WITH_REF  = ("faithfulness", "specificity", "equivalence")
DIMS_REF_FREE  = ("faithfulness", "specificity", "coherence")


# ── Public entry point ────────────────────────────────────────────────────────

def run_llm_judge(
    cfg: dict,
    labeled_df: pd.DataFrame,
    predictions_path: str,
    fine_tuned: bool,
    output_path: str,
    eval_label: str = None,
    tag: str = None,        # "teacher" | "baseline" | "finetuned"
) -> pd.DataFrame:
    """
    Run LLM-as-judge evaluation on a sample of test predictions.

    Teacher tag shortcut
    --------------------
    When tag == "teacher" and judge_mode == "reference", API calls are skipped.
    The teacher is comparing a label to itself — equivalence is trivially 5/5.
    Hardcoded 5/5/5 scores are saved so all downstream combine steps work
    identically, but zero API tokens are consumed.

    Reference-free mode
    -------------------
    Set evaluation.judge_mode: "reference_free" in the config to evaluate all
    models (including teacher) purely against the cluster tickets — no reference
    label sent.  In this mode the third dimension is "coherence" not "equivalence".
    """
    import random
    from phase1.data.schema import (
        CLUSTER_ID, TICKET_DETAILS, TICKET_RANK,
        PROMPT_IDS, cluster_name_col,
        PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_GENERATED_LABEL,
        llm_file,
    )

    judge_cfg  = cfg["evaluation"]["judge_llm"]
    judge_mode = cfg["evaluation"].get("judge_mode", "reference")
    k          = cfg["top_k"]
    model_id   = cfg["teacher_llm"]["model"]
    n_samples  = judge_cfg["n_samples"]
    dims       = DIMS_WITH_REF if judge_mode == "reference" else DIMS_REF_FREE

    preds = []
    with open(predictions_path) as f:
        for line in f:
            preds.append(json.loads(line))

    random.seed(cfg["seed"])
    sampled = random.sample(preds, min(n_samples, len(preds)))

    # ── Teacher shortcut (reference mode only) ────────────────────────────────
    if tag == "teacher" and judge_mode == "reference":
        logger.info(
            f"[llm_judge] Tag=teacher, mode=reference → "
            f"skipping {len(sampled)} API calls (teacher vs itself = trivially 5/5/5). "
            "Hardcoding perfect scores."
        )
        results = _make_perfect_teacher_rows(sampled, dims)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        _save_llm_tag_file(results_df, output_path, tag, dims)
        _log_judge_summary(results_df, fine_tuned, eval_label, dims)
        return results_df

    logger.info(
        f"[llm_judge] Running judge on {len(sampled)} predictions "
        f"(tag={tag}, mode={judge_mode}) ..."
    )

    # ── Build lookups ─────────────────────────────────────────────────────────
    top_k_df = labeled_df[labeled_df[TICKET_RANK] <= k].copy()
    cluster_tickets: dict = {}
    for cid, grp in top_k_df.groupby(CLUSTER_ID):
        cluster_tickets[int(cid)] = grp.sort_values(TICKET_RANK)[TICKET_DETAILS].tolist()

    teacher_labels: dict = {}
    for cid, grp in labeled_df.groupby(CLUSTER_ID):
        row = grp.iloc[0]
        teacher_labels[int(cid)] = {
            pid: str(row[cluster_name_col(model_id, pid)])
            for pid in PROMPT_IDS
            if cluster_name_col(model_id, pid) in row.index
            and pd.notna(row[cluster_name_col(model_id, pid)])
        }

    judge_call = _get_judge_call(judge_cfg, judge_mode)
    results = []

    for i, pred in enumerate(sampled):
        cid       = int(pred[PRED_CLUSTER_ID])
        pid       = pred[PRED_PROMPT_ID]
        candidate = pred[PRED_GENERATED_LABEL]
        tickets   = cluster_tickets.get(cid, [])
        reference = teacher_labels.get(cid, {}).get(pid, "")

        if not tickets:
            logger.warning(f"[llm_judge] Cluster {cid}: no tickets found. Skipping.")
            continue
        if judge_mode == "reference" and not reference:
            logger.warning(f"[llm_judge] Cluster {cid}: no reference label. Skipping.")
            continue

        scores = judge_call(tickets, reference, candidate, judge_cfg)

        composite = sum(scores.get(d, 0) for d in dims) / len(dims)
        row = {
            PRED_CLUSTER_ID:      cid,
            PRED_PROMPT_ID:       pid,
            "fine_tuned":         fine_tuned,
            PRED_GENERATED_LABEL: candidate,
            "reference_label":    reference,
            **{d: scores.get(d, 0) for d in dims},
            "composite_score":    composite,
            "reasoning":          scores.get("reasoning", ""),
        }
        results.append(row)
        time.sleep(cfg["teacher_llm"]["sleep_between_calls"])

        if (i + 1) % 10 == 0:
            logger.info(f"[llm_judge] {i + 1}/{len(sampled)} judged.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"[llm_judge] Judge scores → {output_path}")

    _save_llm_tag_file(results_df, output_path, tag, dims)
    _log_judge_summary(results_df, fine_tuned, eval_label, dims)
    return results_df


# ── Judge score functions (one per mode) ─────────────────────────────────────

def score_with_reference(
    ticket_texts: list[str], reference: str, candidate: str, judge_cfg: dict
) -> dict:
    """Call judge with reference label (standard mode)."""
    from phase1.prompts.templates import format_tickets_block
    tickets_block = format_tickets_block(ticket_texts)
    prompt = JUDGE_PROMPT_WITH_REFERENCE.format(
        tickets=tickets_block, reference=reference, candidate=candidate
    )
    messages = [{"role": "user", "content": prompt}]
    return _call_judge(messages, judge_cfg, DIMS_WITH_REF)


def score_reference_free(
    ticket_texts: list[str], _reference: str, candidate: str, judge_cfg: dict
) -> dict:
    """Call judge WITHOUT reference label (reference_free mode)."""
    from phase1.prompts.templates import format_tickets_block
    tickets_block = format_tickets_block(ticket_texts)
    prompt = JUDGE_PROMPT_REFERENCE_FREE.format(
        tickets=tickets_block, candidate=candidate
    )
    messages = [{"role": "user", "content": prompt}]
    return _call_judge(messages, judge_cfg, DIMS_REF_FREE)


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_judge_call(judge_cfg: dict, judge_mode: str):
    """Return the right scoring function based on mode."""
    return score_reference_free if judge_mode == "reference_free" else score_with_reference


def _make_perfect_teacher_rows(sampled: list[dict], dims: tuple) -> list[dict]:
    """Build hardcoded 5/5/5 rows for the teacher self-evaluation shortcut."""
    from phase1.data.schema import PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_GENERATED_LABEL
    rows = []
    for pred in sampled:
        rows.append({
            PRED_CLUSTER_ID:      int(pred[PRED_CLUSTER_ID]),
            PRED_PROMPT_ID:       pred[PRED_PROMPT_ID],
            "fine_tuned":         False,
            PRED_GENERATED_LABEL: pred[PRED_GENERATED_LABEL],
            "reference_label":    pred[PRED_GENERATED_LABEL],  # same as candidate
            **{d: 5 for d in dims},
            "composite_score":    5.0,
            "reasoning":          "Skipped: teacher label compared to itself (trivially perfect). No API call made.",
        })
    return rows


def _call_judge(messages: list[dict], judge_cfg: dict, dims: tuple) -> dict:
    """Call the judge LLM and parse the JSON response."""
    if judge_cfg["provider"] == "anthropic":
        raw = _anthropic_call(messages, judge_cfg)
    else:
        raw = _openai_call(messages, judge_cfg)

    raw = _strip_markdown_fences(raw)
    try:
        scores = json.loads(raw)
        for d in dims:
            scores[d] = int(scores.get(d, 0))
        scores.setdefault("reasoning", "")
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[llm_judge] Parse error: {raw!r}. Error: {e}")
        return {d: 0 for d in dims} | {"reasoning": "parse_error"}


def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1:] if first_newline != -1 else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


def _get_judge_fn(judge_cfg: dict):
    """Legacy compat shim used by inference_pipeline.py."""
    return score_with_reference


def _judge_anthropic(ticket_texts, reference, candidate, judge_cfg):
    return score_with_reference(ticket_texts, reference, candidate, judge_cfg)


def _judge_openai(ticket_texts, reference, candidate, judge_cfg):
    return score_with_reference(ticket_texts, reference, candidate, judge_cfg)


def _anthropic_call(messages: list[dict], judge_cfg: dict) -> str:
    import anthropic
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        resp = client.messages.create(
            model=judge_cfg["model"],
            max_tokens=200,
            temperature=judge_cfg["temperature"],
            messages=messages,
        )
        return resp.content[0].text

    return _call()


def _openai_call(messages: list[dict], judge_cfg: dict) -> str:
    import openai
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        resp = client.chat.completions.create(
            model=judge_cfg["model"],
            max_tokens=200,
            temperature=judge_cfg["temperature"],
            messages=messages,
        )
        return resp.choices[0].message.content

    return _call()


def _save_llm_tag_file(results_df: pd.DataFrame, output_path: str,
                        tag: str | None, dims: tuple) -> None:
    """Save llm_{tag}.csv with JSON metrics blob."""
    if not tag:
        return
    from phase1.data.schema import llm_file, PRED_GENERATED_LABEL, PRED_CLUSTER_ID, PRED_PROMPT_ID

    llm_path = Path(output_path).parent / llm_file(tag)
    rows = []
    for _, r in results_df.iterrows():
        blob = {d: int(r.get(d, 0)) for d in dims}
        blob["composite"] = float(r.get("composite_score", 0))
        blob["reasoning"] = str(r.get("reasoning", ""))
        rows.append({
            "cluster_id":      r[PRED_CLUSTER_ID],
            "prompt_id":       r[PRED_PROMPT_ID],
            "model":           tag,
            "generated_label": r.get(PRED_GENERATED_LABEL, ""),
            "reference_label": r.get("reference_label", ""),
            "llm_metrics":     json.dumps(blob),
        })
    pd.DataFrame(rows).to_csv(llm_path, index=False)
    logger.info(f"[llm_judge] LLM metrics (JSON) → {llm_path}")


def _log_judge_summary(df: pd.DataFrame, fine_tuned: bool,
                        eval_label: str | None, dims: tuple) -> None:
    label = (
        eval_label.upper() if eval_label
        else ("FINE-TUNED (post-distillation)" if fine_tuned else "BASELINE (pre-distillation)")
    )
    if df.empty:
        return
    lines = "\n".join(
        f"  {d.capitalize():<16}: {df[d].mean():.2f} / 5"
        for d in dims if d in df.columns
    )
    logger.info(
        f"\n{'=' * 55}\n"
        f"  LLM JUDGE SUMMARY — {label}\n"
        f"{'=' * 55}\n"
        f"{lines}\n"
        f"  Composite score:  {df['composite_score'].mean():.2f} / 5\n"
        f"{'=' * 55}"
    )
