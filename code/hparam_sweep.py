"""
hparam_sweep.py - run the Phase 1 pipeline once per hyper-parameter configuration and
compare composite scores (and latency) across all configurations.

Usage
-----
# Preview the runs without executing anything:
python hparam_sweep.py --sweep configs/sweep_config.yaml --dry_run

# Run the sweep (Colab):
python hparam_sweep.py --sweep configs/sweep_config.yaml --device_mode colab

# Resume after a disconnect (finished runs are skipped):
python hparam_sweep.py --sweep configs/sweep_config.yaml --device_mode colab --resume

# Only re-build tables/plots/stats from runs that already finished:
python hparam_sweep.py --sweep configs/sweep_config.yaml --device_mode colab --report_only

How it works
------------
For every config the script writes a patched copy of phase1_config.yaml and launches
`python main.py --phase 1 --config <patched>` in a SUBPROCESS. A subprocess gives each
run a clean GPU (no leaked VRAM between runs) and means one OOM only fails that run.
After each run it reads the run's evaluation files and appends a row to
sweep_results.csv, then writes plots + statistical tests.

Outputs  ({outputs}/sweeps/{sweep_name}/)
-------
sweep_results.csv      one row per config: overrides, composite, cosine, ROUGE, latency, ...
sweep_summary.png      ranked bar chart + composite-vs-latency scatter (+ heatmap / line plot)
sweep_stats.txt        Friedman / RM-ANOVA across configs + Holm-corrected pairwise tests
best_config.yaml       overrides of the top-ranked configuration
sweep_state.json       resume bookkeeping
logs/                  one log file per run
"""

import argparse
import copy
import itertools
import json
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from phase1.evaluation.run_summary import (
    collect_run_metrics, lower_is_better, per_item_scores, primary_value,
)

logger = logging.getLogger("hparam_sweep")


# ── Config helpers ────────────────────────────────────────────────────────────

def _coerce(v):
    """PyYAML reads '1e-4' (no dot) as a string - turn numeric-looking strings into floats."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def set_dotted(cfg: dict, dotted: str, value) -> None:
    """cfg['a']['b'] = value for dotted='a.b' (creates intermediate dicts)."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def expand_runs(sweep: dict) -> list[dict]:
    """Grid combinations + explicit runs -> list of override dicts."""
    runs: list[dict] = []
    grid = sweep.get("grid") or {}
    if grid:
        keys = list(grid.keys())
        for combo in itertools.product(*[grid[k] for k in keys]):
            runs.append(dict(zip(keys, combo)))
    for extra in sweep.get("runs") or []:
        runs.append(dict(extra))
    if not runs:
        raise ValueError("Sweep defines no runs (empty `grid` and `runs`).")
    return runs


def label_for(overrides: dict) -> str:
    """Short readable name, e.g. 'lr=0.0002 | r=16'."""
    short = {"learning_rate": "lr", "num_train_epochs": "ep", "lora_dropout": "drop",
             "per_device_train_batch_size": "bs", "gradient_accumulation_steps": "ga",
             "warmup_ratio": "warm", "lora_alpha": "alpha", "model_id": "model"}
    parts = []
    for k, v in overrides.items():
        name = k.split(".")[-1]
        if name == "model_id":
            v = str(v).split("/")[-1]
        parts.append(f"{short.get(name, name)}={_coerce(v)}")
    return " | ".join(parts)


def build_run_config(base_cfg: dict, sweep: dict, overrides: dict, run_idx: int,
                     reuse_data_ok: bool, skip_baseline: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in (sweep.get("base_overrides") or {}).items():
        set_dotted(cfg, k, v)
    for k, v in overrides.items():
        set_dotted(cfg, k, v)
    if sweep.get("tie_lora_alpha_to_r", True) and "lora.r" in overrides \
            and "lora.lora_alpha" not in overrides:
        cfg["lora"]["lora_alpha"] = cfg["lora"]["r"]
    if reuse_data_ok:
        for step in ("run_clustering", "run_preprocessing", "run_label_generation"):
            cfg["pipeline"][step] = False
    if skip_baseline:
        cfg["pipeline"]["run_baseline_eval"] = False
    cfg["run_tag"] = f"sw{run_idx:02d}"
    cfg["checkpoints"]["use_checkpoints"] = True
    return cfg


# ── Running one config ────────────────────────────────────────────────────────

def run_one(cfg: dict, cfg_path: Path, log_path: Path, device_mode: str | None,
            code_dir: Path) -> int:
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cmd = [sys.executable, "main.py", "--phase", "1", "--config", str(cfg_path)]
    if device_mode:
        cmd += ["--device_mode", device_mode]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=code_dir, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for line in proc.stdout:           # stream to notebook AND log file
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def find_run_dir(outputs_dir: Path, run_tag: str) -> Path | None:
    hits = sorted(outputs_dir.glob(f"*_{run_tag}"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def write_report(results: pd.DataFrame, sweep: dict, sweep_dir: Path) -> None:
    pm = sweep.get("primary_metric", "judge_composite")
    ok = results[results["status"] == "ok"].copy()
    results.to_csv(sweep_dir / "sweep_results.csv", index=False)
    if ok.empty:
        logger.warning("[sweep] No successful runs - nothing to report.")
        return

    penalty = float(sweep.get("overfit_penalty", 0.0) or 0.0)
    ok["primary"] = ok["metrics"].apply(lambda m: primary_value(m, pm, penalty))
    bad = ok[ok["metrics"].apply(lambda m: not m.get("metrics_valid", True))]
    for _, r in bad.iterrows():
        logger.warning(f"[sweep] {r['run_tag']} has incomplete/corrupt metrics: "
                       f"{r['metrics'].get('metrics_warnings')}")
    ok = ok.sort_values("primary", ascending=lower_is_better(pm)).reset_index(drop=True)

    # Flatten metrics into columns of the CSV
    flat = []
    for _, r in ok.iterrows():
        row = {"rank": len(flat) + 1, "config": r["label"], "run_tag": r["run_tag"],
               f"primary:{pm}": r["primary"]}
        row.update({f"ovr:{k}": v for k, v in r["overrides"].items()})
        row.update({k: v for k, v in r["metrics"].items() if k != "run_dir"})
        row["run_dir"] = r["metrics"].get("run_dir", "")
        flat.append(row)
    flat_df = pd.DataFrame(flat)
    flat_df.to_csv(sweep_dir / "sweep_results.csv", index=False)

    # Best config
    best = ok.iloc[0]
    (sweep_dir / "best_config.yaml").write_text(yaml.safe_dump(
        {"primary_metric": pm, "value": float(best["primary"]),
         "overrides": {k: _coerce(v) for k, v in best["overrides"].items()},
         "run_dir": best["metrics"].get("run_dir", "")}, sort_keys=False), encoding="utf-8")

    # Console table
    cols = ["rank", "config", f"primary:{pm}", "judge_composite_baseline",
            "cosine_sim_same_finetuned", "rouge_l_same_finetuned",
            "latency_mean_s_finetuned", "finetune_minutes", "train_final_loss",
            "val_best_eval_loss", "gap_eval_loss", "gap_judge_composite_finetuned",
            "metrics_valid"]
    show = flat_df[[c for c in cols if c in flat_df.columns]]
    try:
        from tabulate import tabulate
        table = tabulate(show, headers="keys", tablefmt="rounded_outline",
                         floatfmt=".3f", showindex=False)
    except ImportError:
        table = show.to_string(index=False)
    logger.info(f"\n{'=' * 78}\n  SWEEP RESULTS ({sweep['sweep_name']}) - ranked by {pm}\n"
                f"{'=' * 78}\n{table}\n{'=' * 78}")

    _plot(ok, sweep, sweep_dir, pm)
    _stats_report(ok, sweep, sweep_dir)


def _plot(ok: pd.DataFrame, sweep: dict, sweep_dir: Path, pm: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("[sweep] matplotlib not installed - skipping plots (pip install matplotlib).")
        return

    base_val = None
    for _, r in ok.iterrows():            # baseline is only evaluated in the first run
        v = r["metrics"].get("judge_composite_baseline")
        if v is not None and not math.isnan(v):
            base_val = v
            break

    lat = ok["metrics"].apply(lambda m: m.get("latency_mean_s_finetuned", float("nan")))
    sweep_keys = list(ok.iloc[0]["overrides"].keys())
    n_panels = 2 + (1 if len(sweep_keys) >= 1 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))

    # 1) ranked bar chart
    ax = axes[0]
    y = np.arange(len(ok))[::-1]
    ax.barh(y, ok["primary"], color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(ok["label"], fontsize=8)
    if base_val is not None and pm == "judge_composite":
        ax.axvline(base_val, color="crimson", ls="--", label=f"baseline SLM ({base_val:.2f})")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel(pm)
    ax.set_title("Configs ranked by primary metric")
    for yi, v in zip(y, ok["primary"]):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)

    # 2) score vs latency scatter
    ax = axes[1]
    ax.scatter(lat, ok["primary"], s=70, color="#F58518")
    for (x, yv), lab in zip(zip(lat, ok["primary"]), ok["run_tag"]):
        if not (math.isnan(x) or math.isnan(yv)):
            ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("mean fine-tuned inference latency (s/label)")
    ax.set_ylabel(pm)
    ax.set_title("Quality vs latency (top-left is best)")
    ax.grid(alpha=0.3)

    # 3) heatmap (2 swept params) or line plot (1 swept param)
    if n_panels == 3:
        ax = axes[2]
        if len(sweep_keys) >= 2:
            kx, ky = sweep_keys[0], sweep_keys[1]
            tbl = pd.DataFrame({
                "x": ok["overrides"].apply(lambda o: _coerce(o.get(kx))),
                "y": ok["overrides"].apply(lambda o: _coerce(o.get(ky))),
                "v": ok["primary"]})
            pivot = tbl.groupby(["y", "x"])["v"].mean().unstack("x")
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(i) for i in pivot.index])
            ax.set_xlabel(kx.split(".")[-1])
            ax.set_ylabel(ky.split(".")[-1])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    if not math.isnan(pivot.values[i, j]):
                        ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center",
                                va="center", color="white", fontsize=9)
            fig.colorbar(im, ax=ax, label=pm)
            ax.set_title(f"{pm}: {kx.split('.')[-1]} x {ky.split('.')[-1]}"
                         + (" (mean over other params)" if len(sweep_keys) > 2 else ""))
        else:
            k = sweep_keys[0]
            xs = ok["overrides"].apply(lambda o: _coerce(o.get(k)))
            order = xs.argsort()
            ax.plot(xs.iloc[order], ok["primary"].iloc[order], marker="o")
            ax.set_xlabel(k.split(".")[-1])
            ax.set_ylabel(pm)
            ax.set_title(f"{pm} vs {k.split('.')[-1]}")
            ax.grid(alpha=0.3)

    fig.suptitle(f"Hyper-parameter sweep: {sweep['sweep_name']}", fontsize=13)
    fig.tight_layout()
    out = sweep_dir / "sweep_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"[sweep] Plot -> {out}")


def _stats_report(ok: pd.DataFrame, sweep: dict, sweep_dir: Path) -> None:
    """Friedman / RM-ANOVA over configs, paired on (cluster, prompt) judge composite."""
    from phase1.evaluation.stats import format_pvalue, multi_group_compare

    series = {}
    for _, r in ok.iterrows():
        s = per_item_scores(r["metrics"]["run_dir"], "finetuned", "composite")
        if s.empty:   # judge disabled -> fall back to cosine
            s = per_item_scores(r["metrics"]["run_dir"], "finetuned", "cosine_sim_same")
        if not s.empty:
            series[r["run_tag"]] = s
    lines = []
    if len(series) < 2:
        lines.append("Need >= 2 runs with per-item scores for statistical tests.")
    else:
        wide = pd.DataFrame(series)
        res = multi_group_compare(wide)
        lines.append(f"Per-item scores: fine-tuned SLM, {res['n_items']} paired items "
                     f"(cluster x prompt), {wide.shape[1]} configs")
        lines.append("\nMean score per config:\n" + res["means"].round(3).to_string())
        lines.append("\nOmnibus tests (H0: all configs score the same):")
        for k, v in res["omnibus"].items():
            lines.append(f"  {k}: {format_pvalue(v) if 'pvalue' in k else v}")
        pw = res["pairwise"]
        if not pw.empty:
            keep = ["group_a", "group_b", "mean_diff", "ci95_low", "ci95_high",
                    "cohens_dz", "wilcoxon_pvalue", "wilcoxon_p_holm"]
            lines.append("\nPairwise (B - A), Wilcoxon signed-rank, Holm-corrected:")
            lines.append(pw[[c for c in keep if c in pw.columns]].round(4).to_string(index=False))
        lines.append("\nInterpretation: a config is only *reliably* better if the omnibus test "
                     "and the Holm-corrected pairwise p-value are < 0.05 AND the CI excludes 0. "
                     "With few test clusters, treat small differences as noise.")
    text = "\n".join(lines)
    (sweep_dir / "sweep_stats.txt").write_text(text, encoding="utf-8")
    logger.info("\n" + text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Hyper-parameter sweep for the SLM distillation pipeline")
    ap.add_argument("--sweep",  default="configs/sweep_config.yaml")
    ap.add_argument("--config", default="configs/phase1_config.yaml", help="Base pipeline config")
    ap.add_argument("--device_mode", choices=["colab", "local_mps", "local_cpu"], default=None)
    ap.add_argument("--dry_run", action="store_true", help="Print the planned runs and exit")
    ap.add_argument("--resume", action="store_true", help="Skip configs that already finished")
    ap.add_argument("--report_only", action="store_true", help="Skip training; rebuild report")
    ap.add_argument("--max_runs", type=int, default=None, help="Only run the first N configs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    code_dir = Path(__file__).resolve().parent
    sweep    = yaml.safe_load(open(args.sweep, encoding="utf-8"))
    base_cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.device_mode:
        base_cfg["device_mode"] = args.device_mode

    # Resolve paths the same way main.py does (handles Colab drive_root override)
    from main import resolve_paths
    resolved   = resolve_paths(copy.deepcopy(base_cfg))
    outputs    = Path(resolved["paths"]["outputs"])
    processed  = Path(resolved["paths"]["data_processed"])
    sweep_dir  = outputs / "sweeps" / sweep["sweep_name"]
    state_path = sweep_dir / "sweep_state.json"
    state      = json.loads(state_path.read_text()) if state_path.exists() else {}

    runs = expand_runs(sweep)
    if args.max_runs:
        runs = runs[:args.max_runs]

    print(f"\nSweep '{sweep['sweep_name']}': {len(runs)} runs | primary metric: "
          f"{sweep.get('primary_metric', 'judge_composite')} | output: {sweep_dir}")
    for i, o in enumerate(runs):
        print(f"  [{i:02d}] {label_for(o)}")
    if args.dry_run:
        return
    sweep_dir.mkdir(parents=True, exist_ok=True)

    records = []
    first_baseline_done = any(v.get("status") == "ok" and v.get("has_baseline")
                              for v in state.values())
    t_sweep = time.time()
    for idx, overrides in enumerate(runs):
        label   = label_for(overrides)
        tag     = f"sw{idx:02d}"
        entry   = state.get(tag, {})
        rec     = {"run_tag": tag, "label": label, "overrides": overrides}

        done = entry.get("status") == "ok" and Path(entry.get("run_dir", "")).exists()
        if args.report_only or (args.resume and done):
            run_dir = Path(entry["run_dir"]) if done else find_run_dir(outputs, tag)
            if run_dir is None:
                rec["status"] = "skipped"
                records.append(rec)
                continue
            logger.info(f"[sweep] {tag} ({label}): using existing run {run_dir.name}")
        else:
            labeled_exists = (processed / "bitext_labeled.csv").exists()
            reuse = bool(sweep.get("reuse_data", True)) and labeled_exists \
                and (processed / "bitext_clustered.csv").exists()
            skip_baseline = bool(sweep.get("baseline_once", True)) and first_baseline_done
            cfg = build_run_config(base_cfg, sweep, overrides, idx, reuse, skip_baseline)
            logger.info(f"\n{'#' * 78}\n# RUN {idx + 1}/{len(runs)}  {tag}  {label}\n{'#' * 78}")
            t0 = time.time()
            rc = run_one(cfg, sweep_dir / "configs" / f"{tag}.yaml",
                         sweep_dir / "logs" / f"{tag}.log", args.device_mode, code_dir)
            run_dir = find_run_dir(outputs, tag)
            if rc != 0 or run_dir is None:
                logger.error(f"[sweep] {tag} FAILED (exit {rc}). See logs/{tag}.log")
                rec.update(status="failed", error=f"exit code {rc}")
                state[tag] = {"status": "failed", "label": label}
                state_path.write_text(json.dumps(state, indent=2))
                records.append(rec)
                continue
            if not skip_baseline:
                first_baseline_done = True
            state[tag] = {"status": "ok", "label": label, "run_dir": str(run_dir),
                          "has_baseline": not skip_baseline,
                          "wall_min": round((time.time() - t0) / 60, 1)}
            state_path.write_text(json.dumps(state, indent=2))

        rec["status"] = "ok"
        rec["metrics"] = collect_run_metrics(run_dir, sweep.get("split", "test"))
        records.append(rec)

        # incremental report after each finished run (survives Colab disconnects)
        write_report(pd.DataFrame(records), sweep, sweep_dir)

    results = pd.DataFrame(records)
    write_report(results, sweep, sweep_dir)
    failed = results[results["status"] == "failed"]
    if len(failed):
        logger.warning(f"[sweep] {len(failed)} run(s) failed: {', '.join(failed['label'])}")
    logger.info(f"[sweep] Done in {(time.time() - t_sweep) / 60:.1f} min -> {sweep_dir}")


if __name__ == "__main__":
    main()
