"""
run_experiments.py — High-efficiency, resilient hyperparameter runner.
"""

import copy
import gc
import os
import shutil
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
DEVICE_MODE = "colab"

MASTER_CSV_PATH = Path(f"{DRIVE_ROOT}/experiment_comparison.csv")
SHARED_EVAL_DIR = Path(f"{DRIVE_ROOT}/data/checkpoints/shared_baseline_eval")

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ── 2. HYPERPARAMETER EXPERIMENT GRID ─────────────────────────────────────────
EXPERIMENTS = [
    {
        "label": "baseline_default",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
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

    boundary_experiments = [
    {
        "label": "lr_higher_8e-4",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 8.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    {
        "label": "lr_extreme_1e-3",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 1.0e-3,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    }
]

    {
        "label": "epochs_5",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 5,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    {
        "label": "lora_r32_a32",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
    {
        "label": "model_smollm_1.7B",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "training.learning_rate": 2.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },

    {
        "label": "model_1.7B_lr_5e-4",
        "student_slm.model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "training.learning_rate": 5.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
]

# ── 3. HELPERS ────────────────────────────────────────────────────────────────
def clear_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def set_nested(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def build_experiment_config(base_cfg: dict, overrides: dict, skip_baseline_eval: bool = False) -> dict:
    cfg = copy.deepcopy(base_cfg)

    # Freeze pre-training stages
    cfg["pipeline"]["run_clustering"] = False
    cfg["pipeline"]["run_preprocessing"] = False
    cfg["pipeline"]["run_label_generation"] = False

    # Dynamic pipeline flags
    cfg["pipeline"]["run_finetuning"] = True
    cfg["pipeline"]["run_baseline_eval"] = not skip_baseline_eval
    cfg["pipeline"]["run_finetuned_eval"] = True
    cfg["pipeline"]["run_llm_judge"] = True
    cfg["pipeline"]["run_business_eval"] = True
    cfg["device_mode"] = DEVICE_MODE

    # Clear run locks
    cfg.setdefault("evaluation", {})["existing_run_dir"] = None
    cfg.setdefault("colab", {})["mount_drive"] = False

    # Ensure target_modules is a list
    lora_cfg = cfg.setdefault("lora", {})
    if isinstance(lora_cfg.get("target_modules"), str):
        lora_cfg["target_modules"] = DEFAULT_TARGET_MODULES

    for key, value in overrides.items():
        if key == "label":
            continue
        set_nested(cfg, key, value)

    # Dynamic VRAM guard for larger models
    target_model = cfg.get("student_slm", {}).get("model_id", "")
    if "1.7B" in target_model:
        cfg["training"]["per_device_train_batch_size"] = 2
        cfg["training"]["gradient_accumulation_steps"] = 8
    elif "3.8B" in target_model or "Phi-3" in target_model:
        cfg["training"]["per_device_train_batch_size"] = 1
        cfg["training"]["gradient_accumulation_steps"] = 16

    return cfg


def find_latest_run_dir(outputs_dir: Path, since_ts: float) -> Path | None:
    if not outputs_dir.exists():
        return None
    candidates = [
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.stat().st_mtime >= since_ts
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def get_metric_val(df: pd.DataFrame, model_name: str, *candidate_cols, default=None):
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


def seed_shared_baseline_if_needed(run_dir: Path):
    """Caches baseline artifacts to avoid repeating baseline evaluation."""
    SHARED_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["baseline_predictions.jsonl", "evaluation/nonllm_baseline.csv", "evaluation/llm_baseline.csv"]:
        src = run_dir / fname
        dest = SHARED_EVAL_DIR / Path(fname).name
        if src.exists() and not dest.exists():
            shutil.copy(src, dest)


def copy_shared_baseline_to_run(run_dir: Path):
    """Copies precomputed baseline artifacts into the current trial."""
    if not SHARED_EVAL_DIR.exists():
        return
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    mappings = {
        "baseline_predictions.jsonl": run_dir / "baseline_predictions.jsonl",
        "nonllm_baseline.csv": eval_dir / "nonllm_baseline.csv",
        "llm_baseline.csv": eval_dir / "llm_baseline.csv",
    }
    for src_name, dest_path in mappings.items():
        src_path = SHARED_EVAL_DIR / src_name
        if src_path.exists() and not dest_path.exists():
            shutil.copy(src_path, dest_path)


def run_one_experiment(exp: dict, base_cfg: dict, shared_baseline_available: bool) -> dict:
    label = exp["label"]
    is_360m = "360M" in exp.get("student_slm.model_id", "360M")
    skip_baseline = shared_baseline_available and is_360m

    print(f"\n{'='*75}\n[STARTING EXPERIMENT]: {label}")
    if skip_baseline:
        print("[OPTIMIZATION] Reusing precomputed baseline evaluation.")
    print('='*75)
    clear_vram()

    cfg = build_experiment_config(base_cfg, exp, skip_baseline_eval=skip_baseline)
    config_path = f"{CODE_DIR}/configs/exp_{label}.yaml"
    Path(f"{CODE_DIR}/configs").mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{CODE_DIR}:{current_pp}" if current_pp else CODE_DIR

    # Only enable offline mode for 360M models to prevent network checks
    if is_360m:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

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
    clear_vram()

    if result.returncode != 0:
        print(f"[ERROR] Experiment '{label}' failed with returncode {result.returncode}")
        print(result.stderr[-2500:])
        return {"label": label, "status": "failed", "elapsed_min": round(elapsed / 60, 2)}

    outputs_dir = Path(f"{DRIVE_ROOT}/outputs")
    run_dir = find_latest_run_dir(outputs_dir, start_time)
    if run_dir is None:
        print(f"[WARNING] Could not identify output run folder for '{label}'")
        return {"label": label, "status": "missing_output", "elapsed_min": round(elapsed / 60, 2)}

    # Save or reuse baseline artifacts
    if not skip_baseline and is_360m:
        seed_shared_baseline_if_needed(run_dir)
    elif skip_baseline:
        copy_shared_baseline_to_run(run_dir)

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

    # Extract non-LLM metrics
    if metrics_path.exists():
        m_df = pd.read_csv(metrics_path)
        row["base_cosine_sim"] = get_metric_val(m_df, "baseline", "cosine_sim_same", "cosine_sim")
        row["base_rouge_l"]    = get_metric_val(m_df, "baseline", "rouge_l_same", "rouge_l")
        row["ft_cosine_sim"]   = get_metric_val(m_df, "finetuned", "cosine_sim_same", "cosine_sim")
        row["ft_rouge_l"]      = get_metric_val(m_df, "finetuned", "rouge_l_same", "rouge_l")

        if row.get("ft_cosine_sim") is not None and row.get("base_cosine_sim") is not None:
            row["Δ_cosine_sim"] = round(row["ft_cosine_sim"] - row["base_cosine_sim"], 4)
        if row.get("ft_rouge_l") is not None and row.get("base_rouge_l") is not None:
            row["Δ_rouge_l"] = round(row["ft_rouge_l"] - row["base_rouge_l"], 4)

    # Extract LLM judge metrics
    if judge_path.exists():
        j_df = pd.read_csv(judge_path)
        row["judge_composite"]    = get_metric_val(j_df, "finetuned", "composite", "composite_score")
        row["judge_faithfulness"] = get_metric_val(j_df, "finetuned", "faithfulness")
        row["judge_specificity"]  = get_metric_val(j_df, "finetuned", "specificity")

    print(f"[COMPLETE] '{label}' finished in {row['elapsed_min']} min")
    time.sleep(2)
    return row


# ── 4. EXECUTION LOOP ─────────────────────────────────────────────────────────
def main():
    drive_root = Path(DRIVE_ROOT)
    if not drive_root.exists():
        raise RuntimeError(f"Drive root not found at {DRIVE_ROOT}. Mount Google Drive first.")

    for subpath in ["phase1", "phase1/data"]:
        init_file = Path(CODE_DIR) / subpath / "__init__.py"
        if init_file.parent.exists() and not init_file.exists():
            init_file.touch()

    if not Path(BASE_CONFIG_PATH).exists():
        raise FileNotFoundError(f"Base config not found at: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Resume from existing Drive master CSV
    if MASTER_CSV_PATH.exists():
        results_df = pd.read_csv(MASTER_CSV_PATH)
        completed_labels = set(results_df[results_df["status"] == "success"]["label"].tolist())
        results = results_df.to_dict("records")
        print(f"[RESUME] Found existing results on Drive. Skipping {len(completed_labels)} completed trial(s).")
    else:
        results = []
        completed_labels = set()

    shared_baseline_available = (SHARED_EVAL_DIR / "baseline_predictions.jsonl").exists()

    for exp in EXPERIMENTS:
        label = exp["label"]
        if label in completed_labels:
            print(f"[SKIP] Experiment '{label}' already completed successfully.")
            continue

        row = run_one_experiment(exp, base_cfg, shared_baseline_available)

        if row.get("status") == "success" and "360M" in row.get("model", ""):
            shared_baseline_available = True

        results = [r for r in results if r.get("label") != label]
        results.append(row)

        results_df = pd.DataFrame(results)
        results_df.to_csv(MASTER_CSV_PATH, index=False)
        print(f"[CHECKPOINT] Progress saved to Drive: {MASTER_CSV_PATH}")

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
