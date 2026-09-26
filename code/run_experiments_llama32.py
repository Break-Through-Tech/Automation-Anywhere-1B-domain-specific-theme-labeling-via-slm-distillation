"""
run_experiments_llama32.py

Isolation sweep for Llama 3.2-3B on the RAW Bitext split, n_samples=500 fixed.

Baseline ("baseline_v2") = current llama_3.2_3b.yaml as-is:
  completion_only_loss=True, load_in_4bit=False, include_domain_in_prompt=True,
  lr=2e-4, epochs=3, r=16, alpha=16, dropout=0.05, warmup_ratio=0.05

Each sweep experiment changes exactly ONE field from baseline_v2 — nothing else.
Run this on raw first. Once you've picked a winning config, re-run this same
script with the clean-split dataset path swapped in to confirm.

Completion / resume semantics
------------------------------
An experiment is only skipped as "already done" if the FINAL adapter save
exists at <run_out>/models/lora_adapter/adapter_model.safetensors — the
location trainer.py writes to via trainer.save_model() at the very end of
run_finetuning(). This deliberately does NOT check intermediate
checkpoint-N/ subdirectories (save_strategy='epoch' writes adapter weights
into those too), because a crashed run that only reached checkpoint-1/ would
otherwise be mistaken for complete and skipped forever — discarding the
partially-trained run instead of letting trainer.py's
resume_from_checkpoint logic pick it back up.
"""

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO         = Path("/content/project")
CODE_DIR     = REPO / "code"
CONFIG_BASE  = CODE_DIR / "configs/llama_3.2_3b.yaml"
EXP_DIR      = REPO / "experiments/llama_3.2_3b_sweep_raw"
DRIVE_ROOT   = Path("/content/drive/MyDrive/slm-distillation")
RUNS_BASE    = DRIVE_ROOT / "runs/llama_3.2_3b_sweep_raw"
RAW_DATASET  = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Absolute path mapping to prevent local relative path lookup errors
DRIVE_PROCESSED_RAW = DRIVE_ROOT / "data/processed_raw"

# ── Baseline (fixed point every sweep experiment deviates from by ONE field) ──
BASELINE = {
    "lr":               2.0e-4,
    "epochs":           3,
    "r":                16,
    "alpha":            16,
    "dropout":          0.05,
    "warmup_ratio":     0.05,
    "warmup_steps":     None,   # None = baseline derives warmup from warmup_ratio
}

# ── Sweep definitions: (param_name, [alternate values to try]) ───────────────
# Each value here generates ONE experiment that changes only that field.
SWEEPS = [
    ("lr",           [1.0e-4, 4.0e-4]),
    ("epochs",       [5]),
    ("r_alpha",      [32]),                # r and alpha move together, kept at 1:1 ratio
    ("dropout",      [0.0]),
    ("warmup_steps", [15]),                # explicit steps, not ratio — guarantees
                                           # separation at small step counts
]


def build_experiments():
    """Build the full experiment list: baseline_v2 first, then one-param sweeps."""
    experiments = [{
        "name": "llama32_raw_baseline_v2",
        **BASELINE,
    }]

    for param, values in SWEEPS:
        for v in values:
            exp = dict(BASELINE)  # copy — everything else stays at baseline
            if param == "r_alpha":
                exp["r"], exp["alpha"] = v, v
                tag = f"r{v}_a{v}"
            else:
                exp[param] = v
                tag = f"{param}{v}"
            exp["name"] = f"llama32_raw_{tag}"
            experiments.append(exp)

    return experiments


EXPERIMENTS = build_experiments()


def clear_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def find_latest_dir(parent: Path, since_ts: float):
    if not parent.exists():
        return None
    dirs = [d for d in parent.iterdir() if d.is_dir() and d.stat().st_mtime >= since_ts]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def already_completed(run_out: Path) -> bool:
    """
    An experiment is fully done only if the FINAL adapter save exists directly
    in <run_out>/models/lora_adapter/ — not in a checkpoint-N/ subfolder.
    """
    adapter_dir = run_out / "models" / "lora_adapter"
    if not adapter_dir.exists():
        return False
    return (adapter_dir / "adapter_model.safetensors").exists() or \
           (adapter_dir / "adapter_model.bin").exists()


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    dest_split = EXP_DIR / "raw"
    dest_split.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_BASE, "r") as f:
        base_cfg = yaml.safe_load(f)

    print(f"\n{len(EXPERIMENTS)} experiments queued (1 baseline + "
          f"{len(EXPERIMENTS) - 1} isolated sweeps):")
    for e in EXPERIMENTS:
        print(f"  - {e['name']}")

    for exp in EXPERIMENTS:
        name    = exp["name"]
        run_out = RUNS_BASE / name / "outputs"

        if already_completed(run_out):
            print(f"\n[SKIP] {name} — final adapter weights already exist at "
                  f"{run_out / 'models' / 'lora_adapter'}.")
            continue

        print("\n" + "=" * 75)
        print(f"[EXECUTING]: {name}  |  lr={exp['lr']}  epochs={exp['epochs']}  "
              f"r={exp['r']}  alpha={exp['alpha']}  dropout={exp['dropout']}  "
              f"warmup_steps={exp.get('warmup_steps')}  warmup_ratio={exp['warmup_ratio']}")
        print("=" * 75)

        clear_vram()
        cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy — original file untouched

        # Fixed dataset — raw split, n_samples stays 500 throughout the sweep
        cfg["dataset"]["name"]      = RAW_DATASET
        cfg["dataset"]["n_samples"] = 500

        # Fixed Absolute Path pointing directly to Drive processed files
        cfg["paths"]["data_processed"] = str(DRIVE_PROCESSED_RAW)
        cfg["paths"]["checkpoints"]    = f"{DRIVE_ROOT}/runs/llama_3.2_3b_sweep_raw/{name}/checkpoints"
        cfg["paths"]["outputs"]        = str(run_out)
        cfg["paths"]["labels_out"]     = str(run_out / "labels")
        cfg["paths"]["models_out"]     = str(run_out / "models")
        cfg["paths"]["evaluation_out"] = str(run_out / "evaluation")
        cfg["paths"]["hf_cache"]       = "/root/.cache/huggingface"

        # Apply this experiment's single deviation from BASELINE
        cfg["training"]["learning_rate"]        = exp["lr"]
        cfg["training"]["num_train_epochs"]     = exp["epochs"]
        cfg["training"]["warmup_ratio"]         = exp["warmup_ratio"]
        cfg["training"]["warmup_steps"]         = exp.get("warmup_steps")  # None → derive from ratio
        cfg["training"]["completion_only_loss"] = True   # constant across the whole sweep
        cfg["lora"]["r"]                        = exp["r"]
        cfg["lora"]["lora_alpha"]               = exp["alpha"]
        cfg["lora"]["lora_dropout"]             = exp["dropout"]

        # Force a real train+eval run for THIS experiment's own config.
        cfg["evaluation"]["existing_run_dir"] = None

        # Data already clustered/labeled for raw n=500 — safely skip those pipeline stages
        has_labeled = (DRIVE_PROCESSED_RAW / "bitext_labeled.csv").exists()
        cfg["pipeline"]["run_clustering"]       = False
        cfg["pipeline"]["run_preprocessing"]    = False
        cfg["pipeline"]["run_label_generation"] = not has_labeled

        trial_cfg_path = CODE_DIR / f"configs/{name}.yaml"
        with open(trial_cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"
        start_ts = time.time()

        res = subprocess.run(
            [sys.executable, str(CODE_DIR / "main.py"),
             "--phase", "1", "--config", str(trial_cfg_path),
             "--device_mode", "colab"],
            cwd=str(CODE_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        duration = (time.time() - start_ts) / 60.0
        clear_vram()

        if res.returncode != 0:
            print(f"[ERROR] Trial '{name}' failed:\n{res.stderr[-2000:]}")
            print(f"        Rerun the script — completed experiments will be "
                  f"skipped, and this one will resume from its last checkpoint "
                  f"(trainer.py) rather than retraining from scratch.")
            continue

        run_dir = find_latest_dir(run_out, start_ts)
        if not run_dir:
            print(f"[WARNING] Could not find run output for {name}")
            continue

        eval_dir = run_dir / "evaluation"
        for fname in ["metrics_summary.csv", "judge_summary.csv", "business_eval.csv"]:
            src = eval_dir / fname
            if src.exists():
                shutil.copy2(src, dest_split / f"{name}_{fname}")

        shutil.copy2(trial_cfg_path, dest_split / f"{name}.yaml")
        print(f"✔ Completed {name} in {duration:.2f} min. Exported to {dest_split}")

    _build_comparison_table(dest_split)
    print("\n" + "=" * 75)
    print("SWEEP COMPLETE — see comparison_raw.csv for the ranked results")
    print("=" * 75)


def _build_comparison_table(dest_split: Path) -> None:
    """Collect every experiment's finetuned composite score into one CSV,
    tagged with which single parameter changed vs. baseline_v2."""
    rows = []
    baseline_params = {
        "lr":           BASELINE["lr"],
        "epochs":       BASELINE["epochs"],
        "r":            BASELINE["r"],
        "alpha":        BASELINE["alpha"],
        "dropout":      BASELINE["dropout"],
        "warmup_ratio": BASELINE["warmup_ratio"],
        "warmup_steps": BASELINE["warmup_steps"],
    }

    for exp in EXPERIMENTS:
        name       = exp["name"]
        judge_path = dest_split / f"{name}_judge_summary.csv"
        if not judge_path.exists():
            continue
        df     = pd.read_csv(judge_path)
        ft_row = df[df["model"] == "finetuned"]
        if ft_row.empty:
            continue

        # Identify which single field differs from baseline (blank for baseline itself)
        changed = [k for k in baseline_params if exp.get(k) != baseline_params[k]]
        varied  = ", ".join(changed) if changed else "baseline_v2"

        row = ft_row.iloc[0].to_dict()
        row["experiment"]   = name
        row["varied_param"] = varied
        row["lr"]           = exp["lr"]
        row["epochs"]       = exp["epochs"]
        row["r"]            = exp["r"]
        row["alpha"]        = exp["alpha"]
        row["dropout"]      = exp["dropout"]
        row["warmup_ratio"] = exp["warmup_ratio"]
        row["warmup_steps"] = exp.get("warmup_steps")
        rows.append(row)

    if not rows:
        print("[WARNING] No completed experiments found for comparison table.")
        return

    comp_df  = pd.DataFrame(rows).sort_values("composite", ascending=False)
    out_path = dest_split / "comparison_raw.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"\nRanked comparison → {out_path}")
    print(comp_df[["experiment", "varied_param", "composite"]].to_string(index=False))


if __name__ == "__main__":
    main()
