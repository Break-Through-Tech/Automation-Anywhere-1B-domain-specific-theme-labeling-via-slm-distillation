"""
phase1/evaluation/combine.py

Reads per-model evaluation files from _cache/ and prediction JSONL files, then
produces four clean user-facing pivot CSVs (all with split column):

  labels_by_cluster.csv    — actual label text per (cluster, split, prompt)
  latency_by_cluster.csv   — inference timing per (cluster, split, prompt)
  nonllm_by_cluster.csv    — cosine sim, ROUGE-L per (cluster, split, prompt)
  llm_by_cluster.csv       — judge scores per (cluster, split, prompt)

Plus two summary files (one row per split × model):
  metrics_summary.csv
  judge_summary.csv

All functions tolerate missing files — a model with no predictions simply
produces NaN in that model's columns without breaking the other models.

Optional master combine
-----------------------
Call create_master_csv(eval_dir) to join all four pivot files into a single
wide CSV for ad-hoc analysis.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TAGS       = ["teacher", "baseline", "finetuned"]
PROMPT_IDS = ["P1", "P2", "P3", "P4", "P5"]


# ── Public entry point ────────────────────────────────────────────────────────

def run_combine(
    eval_dir: str | Path,
    split_map: dict | None = None,       # {cluster_id (int): "train"/"val"/"test"}
    labeled_df=None,                     # pd.DataFrame with teacher label columns
    cfg: dict | None = None,
) -> None:
    """
    Run all combine operations.

    split_map and labeled_df are needed for the split column and labels pivot.
    If not provided, they are loaded from the standard paths in data/processed/.
    """
    eval_dir  = Path(eval_dir)
    cache_dir = eval_dir / "_cache"

    # Load split_map from disk if not passed in
    if split_map is None:
        split_map = _load_split_map(eval_dir)

    # Load labeled_df from disk if not passed in
    if labeled_df is None and cfg is not None:
        from pathlib import Path as P
        labeled_csv = P(cfg["paths"]["data_processed"]) / "bitext_labeled.csv"
        if labeled_csv.exists():
            labeled_df = pd.read_csv(labeled_csv)

    combine_labels_by_cluster(eval_dir, split_map, labeled_df, cfg)
    combine_latencies(eval_dir, cache_dir, split_map)
    combine_nonllm_by_cluster(eval_dir, cache_dir, split_map)
    combine_llm_by_cluster(eval_dir, cache_dir, split_map)
    combine_nonllm_summary(eval_dir, cache_dir, split_map)
    combine_llm_summary(eval_dir, cache_dir, split_map)


# ── Labels pivot ──────────────────────────────────────────────────────────────

def combine_labels_by_cluster(
    eval_dir: Path,
    split_map: dict | None,
    labeled_df,
    cfg: dict | None,
) -> None:
    """
    Build labels_by_cluster.csv: actual label text for all models.

    Teacher labels are available for ALL clusters (from labeled_df).
    SLM labels only exist for clusters that had inference run on them
    (baseline/finetuned_predictions.jsonl).
    """
    from phase1.data.schema import (
        CLUSTER_ID, FILE_LABELS_PIVOT,
        PROMPT_IDS as PIDS, cluster_name_col,
        PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_GENERATED_LABEL,
    )

    if labeled_df is None or cfg is None:
        logger.info("[combine] No labeled_df/cfg — skipping labels_by_cluster.")
        return

    teacher_model = cfg["teacher_llm"]["model"]
    out           = eval_dir / FILE_LABELS_PIVOT
    rows          = []

    # Teacher labels — all clusters, all prompts
    teacher_labels: dict[tuple, str] = {}
    for _, row in labeled_df.iterrows():
        cid = int(row[CLUSTER_ID])
        for pid in PIDS:
            col = cluster_name_col(teacher_model, pid)
            if col in row.index and pd.notna(row[col]):
                teacher_labels[(cid, pid)] = str(row[col])

    # SLM labels — from prediction JSONL files
    slm_labels: dict[str, dict[tuple, str]] = {"baseline": {}, "finetuned": {}}
    for tag in ("baseline", "finetuned"):
        jsonl_path = eval_dir / f"{tag}_predictions.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    r = json.loads(line)
                    slm_labels[tag][(int(r[PRED_CLUSTER_ID]), r[PRED_PROMPT_ID])] = \
                        r.get(PRED_GENERATED_LABEL, "")

    all_clusters = sorted(labeled_df[CLUSTER_ID].unique().astype(int).tolist())
    for cid in all_clusters:
        split = (split_map or {}).get(cid, "unknown")
        for pid in PIDS:
            rows.append({
                "cluster_id":       cid,
                "split":            split,
                "prompt_id":        pid,
                "teacher_label":    teacher_labels.get((cid, pid), ""),
                "baseline_label":   slm_labels["baseline"].get((cid, pid), float("nan")),
                "finetuned_label":  slm_labels["finetuned"].get((cid, pid), float("nan")),
            })

    df = pd.DataFrame(rows).sort_values(["cluster_id", "prompt_id"])
    df.to_csv(out, index=False)
    logger.info(f"[combine] Labels pivot → {out}  ({len(df)} rows)")


# ── Latency pivot ─────────────────────────────────────────────────────────────

def combine_latencies(eval_dir: Path, cache_dir: Path, split_map: dict | None) -> None:
    """
    Pivot per-model latency CSVs into one wide cluster-level file.

    Row: (cluster_id, split, prompt_id)
    Columns: teacher_latency_s | baseline_latency_s | finetuned_latency_s
    """
    from phase1.data.schema import latency_file, FILE_LATENCY_PIVOT

    dfs = {}
    for tag in TAGS:
        path = cache_dir / latency_file(tag)
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={"latency_s": f"{tag}_latency_s"})
            dfs[tag] = df.set_index(["cluster_id", "prompt_id"])[[f"{tag}_latency_s"]]

    if not dfs:
        logger.info("[combine] No latency files found — skipping latency pivot.")
        return

    merged = None
    for tag, df in dfs.items():
        merged = df if merged is None else merged.join(df, how="outer")

    merged = merged.reset_index()
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    merged["split"] = merged["cluster_id"].map(split_map or {}).fillna("unknown")

    cols = ["cluster_id", "split", "prompt_id",
            "teacher_latency_s", "baseline_latency_s", "finetuned_latency_s"]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged = merged.sort_values(["cluster_id", "prompt_id"])

    out = eval_dir / FILE_LATENCY_PIVOT
    merged.to_csv(out, index=False)
    logger.info(f"[combine] Latency pivot → {out}  ({len(merged)} rows)")


# ── Non-LLM metrics pivot ─────────────────────────────────────────────────────

def combine_nonllm_by_cluster(eval_dir: Path, cache_dir: Path, split_map: dict | None) -> None:
    """
    Pivot non-LLM metric files: one row per (cluster_id, split, prompt_id).
    Columns: cosine_same_{model} | cosine_multi_{model} | rouge_l_same_{model} | ...
    """
    from phase1.data.schema import nonllm_file, FILE_NONLLM_PIVOT

    METRICS = ["cosine_sim_same", "cosine_sim_multi", "rouge_l_same", "rouge_l_multi",
               "bertscore_f1_same", "bertscore_f1_multi"]

    frames = {}
    for tag in TAGS:
        path = cache_dir / nonllm_file(tag)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        expanded = pd.json_normalize(df["nonllm_metrics"].apply(json.loads))
        for m in METRICS:
            if m in expanded.columns:
                df[f"{m}_{tag}"] = expanded[m].values
        metric_tag_cols = [f"{m}_{tag}" for m in METRICS if f"{m}_{tag}" in df.columns]
        frames[tag] = df.set_index(["cluster_id", "prompt_id"])[metric_tag_cols]

    if not frames:
        logger.info("[combine] No nonllm files — skipping nonllm_by_cluster.")
        return

    merged = None
    for df in frames.values():
        merged = df if merged is None else merged.join(df, how="outer")

    merged = merged.reset_index()
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    merged["split"] = merged["cluster_id"].map(split_map or {}).fillna("unknown")

    # Order: cluster_id, split, prompt_id, then grouped by metric across models
    metric_cols = [f"{m}_{tag}" for m in METRICS for tag in TAGS]
    ordered = ["cluster_id", "split", "prompt_id"] + [c for c in metric_cols if c in merged.columns]
    merged = merged[[c for c in ordered if c in merged.columns]]
    merged = merged.sort_values(["cluster_id", "prompt_id"])

    out = eval_dir / FILE_NONLLM_PIVOT
    merged.to_csv(out, index=False)
    logger.info(f"[combine] Non-LLM by-cluster pivot → {out}  ({len(merged)} rows)")


# ── LLM judge pivot ───────────────────────────────────────────────────────────

def combine_llm_by_cluster(eval_dir: Path, cache_dir: Path, split_map: dict | None) -> None:
    """
    Pivot LLM judge files: one row per (cluster_id, split, prompt_id).
    Columns: faithfulness_{model} | specificity_{model} | composite_{model} | ...
    """
    from phase1.data.schema import llm_file, FILE_LLM_PIVOT

    frames = {}
    all_dims: set = set()
    for tag in TAGS:
        path = cache_dir / llm_file(tag)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        expanded = pd.json_normalize(df["llm_metrics"].apply(json.loads))
        num_cols = [c for c in expanded.columns if c not in ("reasoning",)]
        all_dims.update(num_cols)
        for col in num_cols:
            df[f"{col}_{tag}"] = expanded[col].values
        tag_cols = [f"{c}_{tag}" for c in num_cols if f"{c}_{tag}" in df.columns]
        frames[tag] = df.set_index(["cluster_id", "prompt_id"])[tag_cols]

    if not frames:
        logger.info("[combine] No llm files — skipping llm_by_cluster.")
        return

    merged = None
    for df in frames.values():
        merged = df if merged is None else merged.join(df, how="outer")

    merged = merged.reset_index()
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    merged["split"] = merged["cluster_id"].map(split_map or {}).fillna("unknown")

    dim_list   = sorted(all_dims - {"composite"}) + ["composite"]
    metric_cols = [f"{d}_{tag}" for d in dim_list for tag in TAGS]
    ordered    = ["cluster_id", "split", "prompt_id"] + [c for c in metric_cols if c in merged.columns]
    merged     = merged[[c for c in ordered if c in merged.columns]]
    merged     = merged.sort_values(["cluster_id", "prompt_id"])

    out = eval_dir / FILE_LLM_PIVOT
    merged.to_csv(out, index=False)
    logger.info(f"[combine] LLM judge by-cluster pivot → {out}  ({len(merged)} rows)")


# ── Summary tables ────────────────────────────────────────────────────────────

def combine_nonllm_summary(eval_dir: Path, cache_dir: Path, split_map: dict | None) -> None:
    """
    Mean non-LLM scores per (split, model). Reads nonllm_by_cluster.csv.
    """
    from phase1.data.schema import FILE_METRICS_SUMMARY, FILE_NONLLM_PIVOT

    pivot_path = eval_dir / FILE_NONLLM_PIVOT
    if not pivot_path.exists():
        return

    pivot = pd.read_csv(pivot_path)
    rows  = []
    for split in ["train", "val", "test"]:
        split_df = pivot[pivot["split"] == split] if "split" in pivot.columns else pivot
        for tag in TAGS:
            tag_cols = [c for c in pivot.columns if c.endswith(f"_{tag}")]
            if not tag_cols:
                continue
            means = split_df[tag_cols].mean()
            clean_means = {c.replace(f"_{tag}", ""): v for c, v in means.items()}
            rows.append({"split": split, "model": tag, **clean_means})

    if not rows:
        return

    summary = pd.DataFrame(rows)
    out     = eval_dir / FILE_METRICS_SUMMARY
    summary.to_csv(out, index=False)
    logger.info(f"[combine] Non-LLM metrics summary → {out}")
    _log_summary_table(summary, "NON-LLM METRICS SUMMARY (by split)")


def combine_llm_summary(eval_dir: Path, cache_dir: Path, split_map: dict | None) -> None:
    """
    Mean LLM judge scores per (split, model). Reads llm_by_cluster.csv.
    """
    from phase1.data.schema import FILE_JUDGE_SUMMARY, FILE_LLM_PIVOT

    pivot_path = eval_dir / FILE_LLM_PIVOT
    if not pivot_path.exists():
        return

    pivot = pd.read_csv(pivot_path)
    rows  = []
    for split in ["train", "val", "test"]:
        split_df = pivot[pivot["split"] == split] if "split" in pivot.columns else pivot
        for tag in TAGS:
            tag_cols = [c for c in pivot.columns if c.endswith(f"_{tag}")]
            if not tag_cols:
                continue
            means = split_df[tag_cols].mean()
            clean_means = {c.replace(f"_{tag}", ""): v for c, v in means.items()}
            rows.append({"split": split, "model": tag, **clean_means})

    if not rows:
        return

    summary = pd.DataFrame(rows)
    out     = eval_dir / FILE_JUDGE_SUMMARY
    summary.to_csv(out, index=False)
    logger.info(f"[combine] LLM judge summary → {out}")
    _log_summary_table(summary, "LLM JUDGE SUMMARY (by split)")


# ── Optional master CSV ───────────────────────────────────────────────────────

def create_master_csv(eval_dir: str | Path) -> Path:
    """
    Join all four pivot files into one wide master CSV for ad-hoc analysis.

    Usage:
        python main.py --phase 1 --config configs/phase1_config.yaml \\
            --create_master_csv --run_dir outputs/20260820_0014_SmolLM2_ep2
    """
    from phase1.data.schema import (FILE_LABELS_PIVOT, FILE_LATENCY_PIVOT,
                                     FILE_NONLLM_PIVOT, FILE_LLM_PIVOT)

    eval_dir = Path(eval_dir)
    KEY      = ["cluster_id", "split", "prompt_id"]

    master = None
    for fname in [FILE_LABELS_PIVOT, FILE_LATENCY_PIVOT, FILE_NONLLM_PIVOT, FILE_LLM_PIVOT]:
        p = eval_dir / fname
        if not p.exists():
            logger.warning(f"[combine] {fname} not found — skipping in master.")
            continue
        df = pd.read_csv(p)
        if master is None:
            master = df
        else:
            extra = [c for c in df.columns if c not in KEY]
            master = master.merge(df[KEY + extra], on=KEY, how="outer")

    if master is None:
        logger.warning("[combine] No pivot files found — master CSV not created.")
        return eval_dir

    out = eval_dir / "master_by_cluster.csv"
    master.sort_values(KEY).to_csv(out, index=False)
    logger.info(f"[combine] Master CSV → {out}  ({len(master)} rows, {len(master.columns)} cols)")
    return out


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_split_map(eval_dir: Path) -> dict:
    """
    Try to load cluster split mapping from data/processed/cluster_splits.json.

    Directory depth reference (eval_dir is always .../.../evaluation/):
      local:  ./outputs/{run_id}/evaluation          → go up 2 levels for drive_root
      Colab:  /content/drive/.../slm-distillation/outputs/{run_id}/evaluation
                                                     → go up 3 levels for drive_root
    """
    from phase1.data.schema import FILE_CLUSTER_SPLITS

    candidates = [
        # Go up 3 levels: eval → run_id → outputs → drive_root
        eval_dir.parent.parent.parent / "data" / "processed" / FILE_CLUSTER_SPLITS,
        # Go up 2 levels: eval → run_id → outputs (local fallback)
        eval_dir.parent.parent / "data" / "processed" / FILE_CLUSTER_SPLITS,
        # Go up 1 level  
        eval_dir.parent / "data" / "processed" / FILE_CLUSTER_SPLITS,
        # CWD relative (running from project root locally)
        Path("data") / "processed" / FILE_CLUSTER_SPLITS,
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                raw = json.load(f)
            logger.debug(f"[combine] Loaded split map from {candidate}")
            return {int(k): v for k, v in raw.items()}

    logger.warning(
        "[combine] cluster_splits.json not found in any expected location — "
        "split column will be 'unknown'. Run with eval_all_splits to regenerate."
    )
    return {}


def _log_summary_table(df: pd.DataFrame, title: str) -> None:
    try:
        from tabulate import tabulate
        table = tabulate(df, headers="keys", tablefmt="rounded_outline",
                         floatfmt=".4f", showindex=False)
    except ImportError:
        table = df.to_string(index=False)
    logger.info(f"\n{'=' * 65}\n  {title}\n{'=' * 65}\n{table}\n{'=' * 65}")
