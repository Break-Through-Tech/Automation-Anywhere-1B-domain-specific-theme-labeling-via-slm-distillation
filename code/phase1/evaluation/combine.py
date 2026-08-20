"""
phase1/evaluation/combine.py

Reads the per-model evaluation files written independently by metrics.py,
llm_judge.py, and business_eval.py, then produces three combined outputs:

  latency_by_cluster.csv  — wide pivot, one row per cluster, columns grouped by
                             prompt so Teacher / Baseline / Fine-tuned are adjacent
  metrics_summary.csv     — mean non-LLM scores per model (one row per model)
  judge_summary.csv        — mean LLM judge scores per model

Can be called at any time after any subset of per-model files exist.
Missing files are skipped gracefully rather than raising errors.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TAGS        = ["teacher", "baseline", "finetuned"]
PROMPT_IDS  = ["P1", "P2", "P3", "P4", "P5"]


# ── Public entry point ────────────────────────────────────────────────────────

def run_combine(eval_dir: str | Path) -> None:
    """Run all combine operations. Missing per-model files are skipped."""
    eval_dir = Path(eval_dir)
    combine_latencies(eval_dir)
    combine_nonllm_summary(eval_dir)
    combine_llm_summary(eval_dir)
    combine_nonllm_by_cluster(eval_dir)   # per-cluster pivot for non-LLM metrics
    combine_llm_by_cluster(eval_dir)      # per-cluster pivot for judge scores


# ── Per-cluster non-LLM pivot ─────────────────────────────────────────────────

def combine_nonllm_by_cluster(eval_dir: Path) -> None:
    """
    Pivot per-model non-LLM metric files into one wide cluster-level file.

    Column order (left-to-right comparison per metric):
      cluster_id | prompt_id |
      cosine_same_teacher | cosine_same_baseline | cosine_same_finetuned |
      cosine_multi_teacher | cosine_multi_baseline | cosine_multi_finetuned |
      rouge_l_same_teacher | ... | rouge_l_multi_finetuned

    Missing models produce NaN columns.
    """
    from phase1.data.schema import nonllm_file

    METRICS = ["cosine_sim_same", "cosine_sim_multi", "rouge_l_same", "rouge_l_multi",
               "bertscore_f1_same", "bertscore_f1_multi"]

    frames = {}
    for tag in TAGS:
        path = eval_dir / nonllm_file(tag)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        expanded = pd.json_normalize(df["nonllm_metrics"].apply(json.loads))
        for m in METRICS:
            if m in expanded.columns:
                df[f"{m}_{tag}"] = expanded[m].values
        frames[tag] = df.set_index(["cluster_id", "prompt_id"])

    if not frames:
        logger.info("[combine] No nonllm files found — skipping nonllm_by_cluster.")
        return

    # Merge all tags
    merged = None
    metric_cols = [f"{m}_{tag}" for m in METRICS for tag in TAGS]
    for tag, df in frames.items():
        tag_cols = [c for c in metric_cols if c in df.columns]
        piece = df[tag_cols]
        merged = piece if merged is None else merged.join(piece, how="outer")

    merged = merged.reset_index().sort_values(["cluster_id", "prompt_id"])

    # Order columns: cluster_id, prompt_id, then grouped by metric
    ordered = ["cluster_id", "prompt_id"] + [
        c for c in metric_cols if c in merged.columns
    ]
    merged = merged[[c for c in ordered if c in merged.columns]]

    out = eval_dir / "nonllm_by_cluster.csv"
    merged.to_csv(out, index=False)
    logger.info(f"[combine] Non-LLM by-cluster pivot → {out}  ({len(merged)} rows)")


# ── Per-cluster LLM judge pivot ───────────────────────────────────────────────

def combine_llm_by_cluster(eval_dir: Path) -> None:
    """
    Pivot per-model LLM judge files into one wide cluster-level file.

    Column order (left-to-right comparison per dimension):
      cluster_id | prompt_id |
      faithfulness_teacher | faithfulness_baseline | faithfulness_finetuned |
      specificity_teacher  | specificity_baseline  | specificity_finetuned  |
      [third dim varies: equivalence in reference mode, coherence in reference_free mode]
      composite_teacher    | composite_baseline    | composite_finetuned
    """
    from phase1.data.schema import llm_file

    frames = {}
    all_dims: set = set()
    for tag in TAGS:
        path = eval_dir / llm_file(tag)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        expanded = pd.json_normalize(df["llm_metrics"].apply(json.loads))
        num_cols = [c for c in expanded.columns if c not in ("reasoning",)]
        all_dims.update(num_cols)
        for col in num_cols:
            df[f"{col}_{tag}"] = expanded[col].values
        frames[tag] = df.set_index(["cluster_id", "prompt_id"])

    if not frames:
        logger.info("[combine] No llm files found — skipping llm_by_cluster.")
        return

    # Merge all tags
    dim_list = sorted(all_dims - {"composite"}) + ["composite"]
    metric_cols = [f"{d}_{tag}" for d in dim_list for tag in TAGS]

    merged = None
    for tag, df in frames.items():
        tag_cols = [c for c in metric_cols if c in df.columns]
        piece = df[tag_cols]
        merged = piece if merged is None else merged.join(piece, how="outer")

    merged = merged.reset_index().sort_values(["cluster_id", "prompt_id"])
    ordered = ["cluster_id", "prompt_id"] + [c for c in metric_cols if c in merged.columns]
    merged = merged[[c for c in ordered if c in merged.columns]]

    out = eval_dir / "llm_by_cluster.csv"
    merged.to_csv(out, index=False)
    logger.info(f"[combine] LLM judge by-cluster pivot → {out}  ({len(merged)} rows)")


# ── Latency pivot ─────────────────────────────────────────────────────────────

def combine_latencies(eval_dir: Path) -> None:
    """
    Pivot per-model latency CSVs into one wide cluster-level file.

    Column order (easy left-to-right comparison per prompt):
      cluster_id | P1_teacher_s | P1_baseline_s | P1_finetuned_s | P2_teacher_s | …

    Missing models produce NaN columns.
    """
    from phase1.data.schema import latency_file, FILE_LATENCY_PIVOT

    dfs = {}
    for tag in TAGS:
        path = eval_dir / latency_file(tag)
        if path.exists():
            df = pd.read_csv(path)
            # Rename latency_s to tag-specific column
            df = df.rename(columns={"latency_s": tag})
            dfs[tag] = df.set_index(["cluster_id", "prompt_id"])
        else:
            logger.debug(f"[combine] {path.name} not found — skipping.")

    if not dfs:
        logger.info("[combine] No latency files found — skipping latency pivot.")
        return

    # Merge all available tags on (cluster_id, prompt_id)
    combined = None
    for tag, df in dfs.items():
        combined = df[[tag]] if combined is None else combined.join(df[[tag]], how="outer")

    combined = combined.reset_index()

    # Pivot: one row per cluster, columns ordered by prompt then model
    rows = {}
    for _, row in combined.iterrows():
        cid = int(row["cluster_id"])
        pid = str(row["prompt_id"])
        if cid not in rows:
            rows[cid] = {"cluster_id": cid}
        for tag in TAGS:
            col = f"{pid}_{tag}_s"
            rows[cid][col] = row.get(tag, float("nan"))

    # Build ordered column list: cluster_id | P1_teacher_s | P1_baseline_s | … | P5_finetuned_s
    ordered_cols = ["cluster_id"] + [
        f"{pid}_{tag}_s"
        for pid in PROMPT_IDS
        for tag in TAGS
    ]

    pivot_df = pd.DataFrame(list(rows.values()))
    # Keep only columns that actually exist
    existing_cols = [c for c in ordered_cols if c in pivot_df.columns]
    pivot_df = pivot_df[existing_cols].sort_values("cluster_id")

    out = eval_dir / FILE_LATENCY_PIVOT
    pivot_df.to_csv(out, index=False)
    logger.info(f"[combine] Latency pivot → {out}  ({len(pivot_df)} clusters)")


# ── Non-LLM metrics summary ───────────────────────────────────────────────────

def combine_nonllm_summary(eval_dir: Path) -> None:
    """
    Read nonllm_{tag}.csv files, unpack JSON metrics column, compute means
    per model, and write metrics_summary.csv.

    Row order: teacher → baseline → finetuned (ceiling → baseline → fine-tuned).
    """
    from phase1.data.schema import nonllm_file, FILE_METRICS_SUMMARY

    rows = []
    for tag in TAGS:
        path = eval_dir / nonllm_file(tag)
        if not path.exists():
            logger.debug(f"[combine] {path.name} not found — skipping.")
            continue

        df = pd.read_csv(path)
        if "nonllm_metrics" not in df.columns:
            logger.warning(f"[combine] {path.name} missing 'nonllm_metrics' column.")
            continue

        # Unpack JSON blobs
        metrics_df = pd.json_normalize(df["nonllm_metrics"].apply(json.loads))
        means      = metrics_df.mean().to_dict()
        rows.append({"model": tag, **means})

    if not rows:
        logger.info("[combine] No non-LLM metric files found — skipping summary.")
        return

    summary = pd.DataFrame(rows)
    # model column first, then metrics
    metric_cols = [c for c in summary.columns if c != "model"]
    summary     = summary[["model"] + metric_cols]

    out = eval_dir / FILE_METRICS_SUMMARY
    summary.to_csv(out, index=False)
    logger.info(f"[combine] Non-LLM metrics summary → {out}")
    _log_summary_table(summary, "NON-LLM METRICS SUMMARY")


# ── LLM judge summary ─────────────────────────────────────────────────────────

def combine_llm_summary(eval_dir: Path) -> None:
    """
    Read llm_{tag}.csv files, unpack JSON metrics column, compute means,
    write judge_summary.csv.
    """
    from phase1.data.schema import llm_file, FILE_JUDGE_SUMMARY

    rows = []
    for tag in TAGS:
        path = eval_dir / llm_file(tag)
        if not path.exists():
            logger.debug(f"[combine] {path.name} not found — skipping.")
            continue

        df = pd.read_csv(path)
        if "llm_metrics" not in df.columns:
            logger.warning(f"[combine] {path.name} missing 'llm_metrics' column.")
            continue

        metrics_df = pd.json_normalize(df["llm_metrics"].apply(json.loads))
        # Exclude reasoning (text) from numeric means
        num_cols   = metrics_df.select_dtypes("number").columns.tolist()
        means      = metrics_df[num_cols].mean().to_dict()
        rows.append({"model": tag, **means})

    if not rows:
        logger.info("[combine] No LLM judge files found — skipping summary.")
        return

    summary = pd.DataFrame(rows)
    metric_cols = [c for c in summary.columns if c != "model"]
    summary     = summary[["model"] + metric_cols]

    out = eval_dir / FILE_JUDGE_SUMMARY
    summary.to_csv(out, index=False)
    logger.info(f"[combine] LLM judge summary → {out}")
    _log_summary_table(summary, "LLM JUDGE SUMMARY")


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_summary_table(df: pd.DataFrame, title: str) -> None:
    try:
        from tabulate import tabulate
        table = tabulate(df, headers="keys", tablefmt="rounded_outline",
                         floatfmt=".4f", showindex=False)
    except ImportError:
        table = df.to_string(index=False)

    logger.info(f"\n{'=' * 65}\n  {title}\n{'=' * 65}\n{table}\n{'=' * 65}")
