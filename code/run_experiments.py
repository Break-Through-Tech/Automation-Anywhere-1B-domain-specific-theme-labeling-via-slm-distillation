"""
run_experiments.py — Automated fine-tuning experiment runner and baseline comparison.

WHAT THIS DOES:
1. Loads your base YAML config.
2. Runs a series of controlled hyperparameter experiments using main.py.
3. Extracts metrics defensively (Cosine Sim, ROUGE-L, LLM Judge composite scores).
4. Compares each run against the baseline model (evaluating deltas: Δ Cosine, Δ ROUGE).
5. Exports results to experiment_comparison.csv.
"""

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# ── 1. CONFIGURATION & PATHS ──────────────────────────────────────────────────
REPO = "/content/project"
CODE_DIR = "/content/project/code"
BASE_CONFIG_PATH = f"{CODE_DIR}/configs/phase1_config_vd.yaml"
DRIVE_ROOT = "/content/drive/MyDrive/slm-distillation"
DEVICE_MODE = "colab"  # "colab", "local_mps", or "local_cpu"

# Standard target modules for LLaMA / SmolLM architectures
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ── 2. HYPERPARAMETER EXPERIMENT GRID ─────────────────────────────────────────
EXPERIMENTS = [
    # Baseline configuration
    {
        "label": "baseline_default",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    # Learning Rate Ablations
    {
        "label": "lr_low_5e-5",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 5.0e-5,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    {
        "label": "lr_high_5e-4",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 5.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    # Epoch Sweep (Testing memorization / convergence)
    {
        "label": "epochs_5",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 5,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    # LoRA Capacity (Rank + Alpha scaling)
    {
        "label": "lora_r32_a32",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
    # Model Capacity Comparison
    {
        "label": "model_smollm_1.7B",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
]


# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def set_nested(cfg: dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] = value from the string 'a.b'."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def build_experiment_config(base_cfg: dict, overrides: dict) -> dict:
    """Clones base config and prepares pipeline flags for training & evaluation."""
    cfg = copy.deepcopy(base_cfg)

    # Freeze pre-training stages (reuse generated labels & clusters)
    cfg["pipeline"]["run_clustering"] = False
    cfg["pipeline"]["run_preprocessing"] = False
    cfg["pipeline"]["run_label_generation"] = False

    # Execute training and downstream evaluations
    cfg["pipeline"]["run_finetuning"] = True
    cfg["pipeline"]["run_baseline_eval"] = True
    cfg["pipeline"]["run_finetuned_eval"] = True
    cfg["pipeline"]["run_llm_judge"] = True
    cfg["pipeline"]["run_business_eval"] = True
    cfg["device_mode"] = DEVICE_MODE

    # Clear eval-only locks so new models train and evaluate cleanly
    cfg.setdefault("evaluation", {})["existing_run_dir"] = None

    # Prevent recursive drive.mount() calls in subprocess
    cfg.setdefault("colab", {})["mount_drive"] = False

    # Guard against string-based target_modules crashing PEFT
    lora_cfg = cfg.setdefault("lora", {})
    if isinstance(lora_cfg.get("target_modules"), str):
        lora_cfg["target_modules"] = DEFAULT_TARGET_MODULES

    for key, value in overrides.items():
        if key == "label":
            continue
        set_nested(cfg, key, value)

    return cfg


def find_latest_run_dir(outputs_dir: Path, since_ts: float) -> Path | None:
    """Locate the newly created output folder under outputs/."""
    if not outputs_dir.exists():
        return None
    candidates = [
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.stat().st_mtime >= since_ts
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def get_metric_val(df: pd.DataFrame, model_name: str, *candidate_cols, default=None):
    """Safely extracts a metric across different schema variants and prevents KeyError."""
    if df.empty:
        return default

    sub = df[df["model"] == model_name] if "model" in df.columns else df
    if "split" in sub.columns and (sub["split"] == "test").any():
        sub = sub[sub["split"] == "test"]

    if sub.empty:
        return default

    for col in candidate_cols:
        if col in sub.columns:
            val = sub[col].iloc[0]
            try:
                return round(float(val), 4) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default

    return default


def run_one_experiment(exp: dict, base_cfg: dict) -> dict:
    """Runs a single experiment subprocess and collects metric deltas."""
    label = exp["label"]
    print(f"\n{'='*75}\n[STARTING EXPERIMENT]: {label}\n{'='*75}")

    cfg = build_experiment_config(base_cfg, exp)
    config_path = f"{CODE_DIR}/configs/exp_{label}.yaml"
    Path(f"{CODE_DIR}/configs").mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{CODE_DIR}:{current_pp}" if current_pp else CODE_DIR

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            f"{CODE_DIR}/main.py",
            "--phase", "1",
            "--config", config_path,
            "--device_mode", DEVICE_MODE,
            "--no_checkpoints",
        ],
        cwd=CODE_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] Experiment '{label}' failed with returncode {result.returncode}")
        print(result.stderr[-2500:])
        return {"label": label, "status": "failed", "elapsed_min": round(elapsed / 60, 2)}

    outputs_dir = Path(f"{DRIVE_ROOT}/outputs")
    run_dir = find_latest_run_dir(outputs_dir, start_time)
    if run_dir is None:
        print(f"[WARNING] Could not identify output run folder for '{label}'")
        return {"label": label, "status": "missing_output", "elapsed_min": round(elapsed / 60, 2)}

    metrics_path = run_dir / "evaluation" / "metrics_summary.csv"
    judge_path = run_dir / "evaluation" / "judge_summary.csv"

    row = {
        "label": label,
        "status": "success",
        "elapsed_min": round(elapsed / 60, 2),
        "model": exp.get("student_slm.model_id", base_cfg.get("student_slm", {}).get("model_id")),
        "lr": exp.get("training.learning_rate", base_cfg.get("training", {}).get("learning_rate")),
        "epochs": exp.get("training.num_train_epochs", base_cfg.get("training", {}).get("num_train_epochs")),
        "lora_r": exp.get("lora.r", base_cfg.get("lora", {}).get("r")),
    }

    # ── Non-LLM Metrics ───────────────────────────────────────────────────────
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        row["base_cosine_sim"] = get_metric_val(metrics_df, "baseline", "cosine_sim_same", "cosine_sim")
        row["base_rouge_l"]    = get_metric_val(metrics_df, "baseline", "rouge_l_same", "rouge_l")
        row["ft_cosine_sim"]   = get_metric_val(metrics_df, "finetuned", "cosine_sim_same", "cosine_sim")
        row["ft_rouge_l"]      = get_metric_val(metrics_df, "finetuned", "rouge_l_same", "rouge_l")

        if row.get("ft_cosine_sim") is not None and row.get("base_cosine_sim") is not None:
            row["Δ_cosine_sim"] = round(row["ft_cosine_sim"] - row["base_cosine_sim"], 4)
        if row.get("ft_rouge_l") is not None and row.get("base_rouge_l") is not None:
            row["Δ_rouge_l"] = round(row["ft_rouge_l"] - row["base_rouge_l"], 4)

    # ── LLM-as-a-Judge Metrics ────────────────────────────────────────────────
    if judge_path.exists():
        judge_df = pd.read_csv(judge_path)
        row["judge_composite"]    = get_metric_val(judge_df, "finetuned", "composite", "composite_score")
        row["judge_faithfulness"] = get_metric_val(judge_df, "finetuned", "faithfulness")
        row["judge_specificity"]  = get_metric_val(judge_df, "finetuned", "specificity")

    print(f"[COMPLETE] '{label}' finished in {row['elapsed_min']} min")
    return row


# ── 4. EXECUTION LOOP WITH AUTO-RESUME ─────────────────────────────────────────
def main():
    drive_root = Path(DRIVE_ROOT)
    if not drive_root.exists():
        raise RuntimeError(
            f"Directory {DRIVE_ROOT} not found. Mount drive first:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=True)"
        )

    for subpath in ["phase1", "phase1/data"]:
        init_file = Path(CODE_DIR) / subpath / "__init__.py"
        if init_file.parent.exists() and not init_file.exists():
            init_file.touch()

    if not Path(BASE_CONFIG_PATH).exists():
        raise FileNotFoundError(f"Base config not found at: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    out_path = Path(f"{CODE_DIR}/experiment_comparison.csv")
    
    # ── Check for existing completed trials to resume cleanly ────────────────
    if out_path.exists():
        results_df = pd.read_csv(out_path)
        completed_labels = set(results_df[results_df["status"] == "success"]["label"].tolist())
        results = results_df.to_dict("records")
        print(f"[RESUME] Found existing results. Skipping {len(completed_labels)} already completed experiment(s).")
    else:
        results = []
        completed_labels = set()

    for exp in EXPERIMENTS:
        label = exp["label"]
        if label in completed_labels:
            print(f"[SKIP] Experiment '{label}' already succeeded in previous run.")
            continue

        row = run_one_experiment(exp, base_cfg)
        
        # Remove previous failed attempt if retrying
        results = [r for r in results if r.get("label") != label]
        results.append(row)

        # ── Checkpoint: Save immediately to disk after EACH experiment ───────
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_path, index=False)
        print(f"[CHECKPOINT] Progress saved to {out_path}")

    print(f"\n\n{'='*75}\nEXPERIMENT SWEEP COMPLETE — SUMMARY TABLE\n{'='*75}")
    display_cols = [
        "label", "model", "lr", "epochs", "lora_r", 
        "base_cosine_sim", "ft_cosine_sim", "Δ_cosine_sim",
        "judge_composite", "elapsed_min", "status"
    ]
    cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
