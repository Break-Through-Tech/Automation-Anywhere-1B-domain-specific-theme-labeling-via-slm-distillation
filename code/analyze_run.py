"""
analyze_run.py - post-hoc analysis of ONE finished pipeline run.

Covers: average latency per model, cost (teacher token pricing vs SLM on AWS/Azure/GCP),
extra non-LLM evaluation metrics, and statistical tests comparing the groups.
No GPU or API calls needed - it only reads the files the pipeline already wrote.

Usage
-----
python analyze_run.py --run_dir outputs/20261018_2010_Qwen2.5-3B-Instruct_ep5
python analyze_run.py --run_dir <run_dir> --monthly_labels 1000000      # volume scenario

Writes to <run_dir>/evaluation/analysis/ :
  latency_stats.csv, latency_stats_tests.csv, quality_tests.csv,
  extra_metrics.csv, cost_per_label.csv, cost_breakeven.csv,
  analysis_report.txt, latency_boxplot.png, cost_vs_volume.png
"""

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from phase1.evaluation.run_summary import per_item_scores
from phase1.evaluation.stats import (
    format_pvalue, multi_group_compare, paired_compare,
)

logger = logging.getLogger("analyze_run")
TAGS = ["teacher", "baseline", "finetuned"]
NICE = {"teacher": "Teacher LLM", "baseline": "Baseline SLM", "finetuned": "Fine-tuned SLM"}

# ── Default pricing (USD). APPROXIMATE list prices - VERIFY before quoting them. ──
# Override any of this under a top-level `cost:` section in phase1_config.yaml.
DEFAULT_COST = {
    "chars_per_token": 4,                     # rough English heuristic
    "teacher_pricing_per_1m_tokens": {        # {input, output}
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "gpt-4o-mini":      {"input": 0.15, "output": 0.60},
    },
    # Latency was MEASURED on a Colab T4. `speed_vs_t4` rescales it for other GPUs and is an
    # ASSUMPTION (benchmark on the real instance to replace it).
    "cloud_instances": [
        {"name": "AWS g4dn.xlarge (T4 16GB)",         "hourly_usd": 0.526, "speed_vs_t4": 1.0},
        {"name": "AWS g6.xlarge (L4 24GB)",           "hourly_usd": 0.805, "speed_vs_t4": 1.5},
        {"name": "AWS g5.xlarge (A10G 24GB)",         "hourly_usd": 1.006, "speed_vs_t4": 2.0},
        {"name": "Azure NC4as T4 v3 (T4 16GB)",       "hourly_usd": 0.526, "speed_vs_t4": 1.0},
        {"name": "GCP n1-standard-4 + T4 (16GB)",     "hourly_usd": 0.54,  "speed_vs_t4": 1.0},
        {"name": "GCP g2-standard-4 (L4 24GB)",       "hourly_usd": 0.71,  "speed_vs_t4": 1.5},
    ],
    "hours_per_month": 730,
    "target_utilization": 0.6,    # fraction of an always-on GPU's time spent generating
}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_cfg(path: str, run_dir: Path) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cost = copy.deepcopy(DEFAULT_COST)
    user = cfg.get("cost") or {}
    for k, v in user.items():
        cost[k] = v
    cfg["cost"] = cost
    return cfg


def load_latency(eval_dir: Path, split: str | None) -> pd.DataFrame | None:
    p = eval_dir / "latency_by_cluster.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if split and "split" in df.columns and (df["split"] == split).any():
        df = df[df["split"] == split]
    return df


# ── 1. Latency ────────────────────────────────────────────────────────────────

def latency_section(lat: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for tag in TAGS:
        col = f"{tag}_latency_s"
        if col not in lat.columns or lat[col].dropna().empty:
            continue
        s = lat[col].dropna()
        rows.append({"model": NICE[tag], "n_calls": len(s), "mean_s": s.mean(),
                     "median_s": s.median(), "std_s": s.std(ddof=1) if len(s) > 1 else np.nan,
                     "p95_s": s.quantile(0.95), "min_s": s.min(), "max_s": s.max(),
                     "labels_per_min": 60.0 / s.mean()})
    stats_df = pd.DataFrame(rows)
    lines = []
    if stats_df.empty:
        return stats_df, ["No latency data found."]
    if "Teacher LLM" in set(stats_df["model"]):
        t_mean = stats_df.loc[stats_df["model"] == "Teacher LLM", "mean_s"].iloc[0]
        stats_df["speedup_vs_teacher"] = t_mean / stats_df["mean_s"]
    stats_df.to_csv(out / "latency_stats.csv", index=False)
    lines.append("AVERAGE LATENCY PER MODEL (seconds per label; excludes one-time model load)")
    lines.append(stats_df.round(3).to_string(index=False))

    # Statistical tests: items are paired on (cluster, prompt)
    present = [t for t in TAGS if f"{t}_latency_s" in lat.columns]
    wide = lat.set_index(["cluster_id", "prompt_id"])[[f"{t}_latency_s" for t in present]]
    wide.columns = [NICE[t] for t in present]
    wide = wide.dropna()
    if len(present) >= 3 and len(wide) >= 3:
        res = multi_group_compare(wide, higher_is_better=False)
        lines.append(f"\nLatency - omnibus tests across {len(present)} models "
                     f"({res['n_items']} paired items):")
        for k, v in res["omnibus"].items():
            lines.append(f"  {k}: {format_pvalue(v) if 'pvalue' in k else round(v, 4)}")
        pw = res["pairwise"]
        pw.to_csv(out / "latency_stats_tests.csv", index=False)
        lines.append("Pairwise (Wilcoxon signed-rank, Holm-corrected):")
        keep = ["group_a", "group_b", "mean_diff", "cohens_dz", "wilcoxon_p_holm"]
        lines.append(pw[[c for c in keep if c in pw.columns]].round(4).to_string(index=False))
    elif len(present) == 2 and len(wide) >= 2:
        r = paired_compare(wide.iloc[:, 0], wide.iloc[:, 1], wide.columns[0], wide.columns[1])
        pd.DataFrame([r]).to_csv(out / "latency_stats_tests.csv", index=False)
        lines.append(f"\nLatency paired test: {r}")
    return stats_df, lines


# ── 2. Quality tests (baseline vs fine-tuned, + teacher where available) ─────

def quality_tests(run_dir: Path, out: Path) -> list[str]:
    lines = ["QUALITY: Fine-tuned vs Baseline SLM (paired on cluster x prompt; diff = FT - baseline)"]
    rows = []
    for metric, nice in [("composite", "LLM-judge composite (1-5)"),
                         ("cosine_sim_same", "Cosine similarity"),
                         ("rouge_l_same", "ROUGE-L")]:
        a = per_item_scores(run_dir, "baseline", metric)
        b = per_item_scores(run_dir, "finetuned", metric)
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        r = paired_compare(a[common], b[common], "baseline", "finetuned")
        r["metric"] = nice
        rows.append(r)
    if not rows:
        return lines + ["  (no per-item score files found - run the judge / non-LLM eval first)"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "quality_tests.csv", index=False)
    keep = ["metric", "n_pairs", "mean_a", "mean_b", "mean_diff", "ci95_low", "ci95_high",
            "cohens_dz", "t_pvalue", "wilcoxon_pvalue"]
    lines.append(df[[c for c in keep if c in df.columns]].round(4).to_string(index=False))
    lines.append("  mean_a = baseline, mean_b = fine-tuned. CI = bootstrap 95% of mean diff.\n"
                 "  Cohen's d_z: ~0.2 small, 0.5 medium, 0.8 large. p < 0.05 => unlikely to be chance.")
    return lines


# ── 3. Extra non-LLM evaluation metrics (label-quality diagnostics) ──────────

def _norm(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _token_f1(a: str, b: str) -> float:
    ta, tb = _norm(a), _norm(b)
    if not ta or not tb:
        return 0.0
    common = {}
    for t in ta:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in tb:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def extra_metrics(eval_dir: Path, out: Path, split: str | None) -> list[str]:
    p = eval_dir / "labels_by_cluster.csv"
    if not p.exists():
        return ["No labels_by_cluster.csv - skipping extra metrics."]
    df = pd.read_csv(p)
    if split and "split" in df.columns and (df["split"] == split).any():
        df = df[df["split"] == split]
    rows = []
    for tag in TAGS:
        col = f"{tag}_label"
        if col not in df.columns or df[col].dropna().empty:
            continue
        sub = df.dropna(subset=[col])
        labels = sub[col].astype(str)
        words = labels.apply(lambda s: len(_norm(s)))
        row = {"model": NICE[tag], "n": len(sub),
               "mean_words": words.mean(),
               "in_5_15_words_%": 100 * ((words >= 5) & (words <= 15)).mean(),
               "empty_%": 100 * (words == 0).mean(),
               # diversity: within each prompt, distinct clusters should get distinct labels
               "distinct_label_ratio": labels.str.lower().nunique() / len(labels)}
        if tag != "teacher" and "teacher_label" in sub.columns:
            tl = sub["teacher_label"].astype(str)
            row["exact_match_teacher_%"] = 100 * np.mean(
                [" ".join(_norm(a)) == " ".join(_norm(b)) for a, b in zip(labels, tl)])
            row["token_f1_vs_teacher"] = np.mean([_token_f1(a, b) for a, b in zip(labels, tl)])
            row["length_ratio_vs_teacher"] = float(
                words.mean() / max(1e-9, tl.apply(lambda s: len(_norm(s))).mean()))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "extra_metrics.csv", index=False)
    return ["EXTRA LABEL-QUALITY METRICS (non-LLM)", res.round(3).to_string(index=False),
            "  in_5_15_words_% = format compliance with the prompt's length rule; "
            "distinct_label_ratio = label diversity (low => collapsed/generic labels)."]


# ── 4. Cost ───────────────────────────────────────────────────────────────────

def _teacher_tokens(run_dir: Path, cfg: dict, labels_df: pd.DataFrame | None) -> tuple[float, float, str]:
    """Estimate mean (input_tokens, output_tokens) per teacher call from the real prompts."""
    cpt = float(cfg["cost"]["chars_per_token"])
    labeled = run_dir / "labeled_data.csv"
    try:
        from phase1.data.schema import CLUSTER_ID, TICKET_DETAILS, TICKET_RANK, PROMPT_IDS
        from phase1.prompts.templates import build_messages
        df = pd.read_csv(labeled)
        k = cfg["top_k"]
        in_chars = []
        for _, grp in df[df[TICKET_RANK] <= k].groupby(CLUSTER_ID):
            tickets = grp.sort_values(TICKET_RANK)[TICKET_DETAILS].tolist()
            for pid in PROMPT_IDS:
                msgs = build_messages(pid, tickets, cfg, cfg["dataset"]["domain"])
                in_chars.append(sum(len(m["content"]) for m in msgs))
        t_in = float(np.mean(in_chars)) / cpt
        src = f"estimated from {len(in_chars)} real prompts (chars/{cpt:g})"
    except Exception as e:  # noqa: BLE001
        t_in, src = 500.0, f"default 500 (could not rebuild prompts: {e})"
    t_out = 20.0
    if labels_df is not None and "teacher_label" in labels_df.columns:
        t_out = float(labels_df["teacher_label"].astype(str).str.len().mean()) / cpt
    return t_in, t_out, src


def cost_section(run_dir: Path, eval_dir: Path, cfg: dict, out: Path,
                 lat_stats: pd.DataFrame, monthly_labels: int) -> list[str]:
    cost = cfg["cost"]
    labels_p = eval_dir / "labels_by_cluster.csv"
    labels_df = pd.read_csv(labels_p) if labels_p.exists() else None
    lines = ["COST"]

    # Teacher: per-token API pricing
    model = cfg["teacher_llm"]["model"]
    price = cost["teacher_pricing_per_1m_tokens"].get(model)
    if price is None:
        lines.append(f"  ! No pricing for teacher '{model}' - add it under cost.teacher_pricing_per_1m_tokens")
        return lines
    t_in, t_out, src = _teacher_tokens(run_dir, cfg, labels_df)
    teacher_per_label = (t_in * price["input"] + t_out * price["output"]) / 1e6
    lines.append(f"  Teacher {model}: ${price['input']}/1M input, ${price['output']}/1M output tokens")
    lines.append(f"    avg tokens/call: {t_in:.0f} in + {t_out:.0f} out  [{src}]")
    lines.append(f"    cost per label: ${teacher_per_label:.6f}   (per 1,000 labels: ${teacher_per_label * 1000:.3f})")

    # SLM: GPU-hour pricing x measured latency
    ft = lat_stats[lat_stats["model"] == NICE["finetuned"]] if not lat_stats.empty else lat_stats
    if ft.empty:
        lines.append("  ! No fine-tuned latency measured - cannot price the SLM.")
        return lines
    ft_lat = float(ft["mean_s"].iloc[0])
    hours_pm, util = cost["hours_per_month"], cost["target_utilization"]
    rows = [{"deployment": f"Teacher API ({model})", "latency_s": float(
        lat_stats.loc[lat_stats["model"] == NICE["teacher"], "mean_s"].iloc[0])
        if (lat_stats["model"] == NICE["teacher"]).any() else np.nan,
        "hourly_usd": np.nan, "cost_per_label_usd": teacher_per_label,
        "cost_per_1k_labels_usd": teacher_per_label * 1000, "monthly_fixed_usd": 0.0,
        "breakeven_labels_per_month": np.nan}]
    for inst in cost["cloud_instances"]:
        lat = ft_lat / float(inst.get("speed_vs_t4", 1.0))
        # Always-on endpoint: you pay 24/7; effective capacity = hours * util / latency
        monthly = inst["hourly_usd"] * hours_pm
        capacity = hours_pm * 3600 * util / lat
        per_label_busy = inst["hourly_usd"] * lat / 3600          # 100% utilisation (batch job)
        per_label_always_on = monthly / capacity                  # at target utilisation
        breakeven = monthly / teacher_per_label                   # labels/month where SLM == API
        rows.append({"deployment": inst["name"], "latency_s": lat, "hourly_usd": inst["hourly_usd"],
                     "cost_per_label_usd": per_label_always_on,
                     "cost_per_1k_labels_usd": per_label_always_on * 1000,
                     "monthly_fixed_usd": monthly, "breakeven_labels_per_month": breakeven,
                     "cost_per_label_batch_100pct_util_usd": per_label_busy,
                     "monthly_capacity_labels": capacity})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "cost_per_label.csv", index=False)
    show = cdf[["deployment", "latency_s", "hourly_usd", "cost_per_1k_labels_usd",
                "monthly_fixed_usd", "breakeven_labels_per_month"]]
    lines.append(f"\n  SLM measured latency {ft_lat:.3f}s/label on this run's GPU. Always-on endpoint at "
                 f"{util:.0%} utilisation, {hours_pm} h/month. speed_vs_t4 is an assumption.")
    lines.append(show.round(4).to_string(index=False))

    # One-time training cost
    biz = eval_dir / "business_eval.csv"
    if biz.exists():
        d = dict(zip(*pd.read_csv(biz)[["metric", "value"]].T.values))
        if "finetuning_wall_time_min" in d:
            t4 = cost["cloud_instances"][0]["hourly_usd"]
            lines.append(f"\n  One-time fine-tuning: {d['finetuning_wall_time_min']:.1f} min "
                         f"-> ~${d['finetuning_wall_time_min'] / 60 * t4:.2f} at ${t4}/h (T4)")

    be_rows = []
    for _, r in cdf.iloc[1:].iterrows():
        slm_total = r["monthly_fixed_usd"]
        api_total = monthly_labels * teacher_per_label
        be_rows.append({"deployment": r["deployment"], "monthly_labels": monthly_labels,
                        "teacher_api_usd": api_total, "slm_usd": slm_total,
                        "slm_cheaper": slm_total < api_total,
                        "capacity_ok": monthly_labels <= r["monthly_capacity_labels"]})
    be = pd.DataFrame(be_rows)
    be.to_csv(out / "cost_breakeven.csv", index=False)
    lines.append(f"\n  Scenario: {monthly_labels:,} labels/month")
    lines.append(be.round(2).to_string(index=False))
    lines.append("  NOTE: a labelling job this small (a few hundred calls) costs cents via the API; the "
                 "SLM only wins on cost at volume - and a pay-per-second batch job (see "
                 "cost_per_label_batch_100pct_util_usd) beats an always-on endpoint at low volume.")

    _plot_cost(cdf, teacher_per_label, out)
    return lines


def _plot_cost(cdf: pd.DataFrame, teacher_per_label: float, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    vol = np.logspace(3, 8, 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vol, vol * teacher_per_label, "k-", lw=2.5, label="Teacher API (per-token)")
    for _, r in cdf.iloc[1:].iterrows():
        fixed = np.where(vol <= r["monthly_capacity_labels"], r["monthly_fixed_usd"], np.nan)
        ax.plot(vol, fixed, label=r["deployment"], alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("labels per month")
    ax.set_ylabel("monthly cost (USD)")
    ax.set_title("Monthly cost: Teacher API vs always-on SLM endpoint\n(SLM lines stop at the GPU's capacity)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_volume.png", dpi=150)
    plt.close(fig)


def _plot_latency(lat: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    data, names = [], []
    for t in TAGS:
        c = f"{t}_latency_s"
        if c in lat.columns and lat[c].notna().any():
            data.append(lat[c].dropna().values)
            names.append(NICE[t])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=names, showmeans=True)
    ax.set_ylabel("seconds per label")
    ax.set_title("Per-label latency by model (triangle = mean)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "latency_boxplot.png", dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse one finished pipeline run")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="configs/phase1_config.yaml")
    ap.add_argument("--split", default="test", help="Split to analyse (falls back to all rows)")
    ap.add_argument("--monthly_labels", type=int, default=1_000_000,
                    help="Volume scenario for the cost break-even table")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    run_dir  = Path(args.run_dir)
    eval_dir = run_dir / "evaluation"
    if not eval_dir.exists():
        sys.exit(f"Evaluation dir not found: {eval_dir}")
    out = eval_dir / "analysis"
    out.mkdir(exist_ok=True)
    cfg = load_cfg(args.config, run_dir)

    report: list[str] = []
    lat = load_latency(eval_dir, args.split)
    lat_stats = pd.DataFrame()
    if lat is not None:
        lat_stats, lines = latency_section(lat, out)
        report += lines + [""]
        _plot_latency(lat, out)
    else:
        report += ["No latency_by_cluster.csv found.", ""]
    report += quality_tests(run_dir, out) + [""]
    report += extra_metrics(eval_dir, out, args.split) + [""]
    report += cost_section(run_dir, eval_dir, cfg, out, lat_stats, args.monthly_labels)

    text = "\n".join(report)
    (out / "analysis_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
