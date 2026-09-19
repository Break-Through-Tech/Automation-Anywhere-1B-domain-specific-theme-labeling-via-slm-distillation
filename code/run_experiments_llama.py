"""
run_experiments_llama.py — Dedicated multi-dataset hyperparameter runner for Llama 3.2-3B.
Uses configs/llama_3.2_3b.yaml as the base configuration.
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

# ── 1. PATHS & SETTINGS ────────────────────────────────────────────────────────
REPO = Path("/content/project")
CODE_DIR = REPO / "code"
DRIVE_ROOT = Path("/content/drive/MyDrive/slm-distillation")
DEVICE_MODE = "colab"

# Check code/configs first, then repo root configs/
if (CODE_DIR / "configs/llama_3.2_3b.yaml").exists():
    BASE_CONFIG_PATH = CODE_DIR / "configs/llama_3.2_3b.yaml"
elif (REPO / "configs/llama_3.2_3b.yaml").exists():
    BASE_CONFIG_PATH = REPO / "configs/llama_3.2_3b.yaml"
else:
    raise FileNotFoundError("Could not find llama_3.2_3b.yaml in code/configs/ or configs/")

MASTER_CSV_PATH = DRIVE_ROOT / "runs/llama_3.2_3b/experiment_comparison_llama32.csv"

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ── 2. HYPERPARAMETER TRIALS (Run across BOTH Raw & Clean) ─────────────────────
LLAMA_TRIALS = [
    {
        "name": "default_baseline",
        "training.learning_rate": 3.0e-4,
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
    },
    {
        "name": "lr_optimal_3.5e-4",
        "training.learning_rate": 3.5e-4,
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
    {
        "name": "high_capacity_4e-4_ep4",   # Target score >= 4.0 - 4.5
        "training.learning_rate": 4.0e-4,
        "training.num_train_epochs": 4,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
]

# ── 3. HELPERS ─────────────────────────────────────────────────────────────────
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

def build_config_for_split(base_cfg: dict, trial: dict, data_split: str, skip_baseline: bool) -> tuple[dict, Path, Path]:
    cfg = copy.deepcopy(base_cfg)
    is_clean = (data_split == "clean")

    cfg["student_slm"]["model_id"] = "meta-llama/Llama-3.2-3B-Instruct"

    out_dir = DRIVE_ROOT / f"runs/llama_3.2_3b/outputs_{data_split}"
    chk_dir = DRIVE_ROOT / f"runs/llama_3.2_3b/checkpoints_{data_split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = (
        "/content/drive/MyDrive/slm-distillation/data/cleaned/bitext_cleaned_support.csv"
        if is_clean else
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )

    cfg["dataset"]["name"] = dataset_name
    cfg["dataset"]["it_categories"] = ["ACCOUNT", "DELIVERY", "CONTACT"]
    cfg["paths"]["data_processed"] = f"{{drive_root}}/data/processed_{data_split}"
    cfg["paths"]["checkpoints"] = str(chk_dir)
    cfg["paths"]["outputs"] = str(out_dir)
    cfg["paths"]["labels_out"] = str(out_dir / "labels")
    cfg["paths"]["models_out"] = str(out_dir / "models")
    cfg["paths"]["evaluation_out"] = str(out_dir / "evaluation")
    cfg["paths"]["hf_cache"] = "/root/.cache/huggingface"

    proc_dir = DRIVE_ROOT / f"data/processed_{data_split}"
    if (proc_dir / "bitext_labeled.csv").exists():
        cfg["pipeline"]["run_clustering"] = False
        cfg["pipeline"]["run_preprocessing"] = False
        cfg["pipeline"]["run_label_generation"] = False
    else:
        cfg["pipeline"]["run_clustering"] = True
        cfg["pipeline"]["run_preprocessing"] = True
        cfg["pipeline"]["run_label_generation"] = True

    cfg["pipeline"]["run_finetuning"] = True
    cfg["pipeline"]["run_baseline_eval"] = not skip_baseline
    cfg["pipeline"]["run_finetuned_eval"] = True
    cfg["pipeline"]["run_llm_judge"] = True
    cfg["pipeline"]["run_business_eval"] = True
    cfg["device_mode"] = DEVICE_MODE

    lora_cfg = cfg.setdefault("lora", {})
    if isinstance(lora_cfg.get("target_modules"), str) or not lora_cfg.get("target_modules"):
        lora_cfg["target_modules"] = DEFAULT_TARGET_MODULES

    for k, v in trial.items():
        if k != "name":
            set_nested(cfg, k, v)

    cfg["training"]["per_device_train_batch_size"] = 2
    cfg["training"]["gradient_accumulation_steps"] = 8
    cfg["training"]["fp16"] = True
    cfg["training"]["bf16"] = False
    cfg["training"]["gradient_checkpointing"] = True
    cfg["student_slm"]["max_seq_length"] = 1024

    return cfg, out_dir, chk_dir

def find_latest_run_dir(outputs_dir: Path, since_ts: float) -> Path | None:
    if not outputs_dir.exists():
        return None
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and p.stat().st_mtime >= since_ts]
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

def run_single_trial_split(trial: dict, data_split: str, base_cfg: dict) -> dict:
    trial_name = trial["name"]
    full_label = f"llama32_{data_split}_{trial_name}"
    
    shared_baseline_dir = DRIVE_ROOT / f"runs/llama_3.2_3b/checkpoints_{data_split}/shared_baseline"
    shared_baseline_available = (shared_baseline_dir / "baseline_predictions.jsonl").exists()

    print(f"\n{'='*75}\n[EXECUTING]: {full_label} | Split: {data_split.upper()}\n{'='*75}")
    if shared_baseline_available:
        print(f"[CACHE] Reusing baseline predictions for {data_split} split.")
    clear_vram()

    cfg, out_dir, chk_dir = build_config_for_split(base_cfg, trial, data_split, skip_baseline=shared_baseline_available)
    config_path = CODE_DIR / f"configs/active_{full_label}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / "main.py"),
            "--phase", "1",
            "--config", str(config_path),
            "--device_mode", DEVICE_MODE,
            "--no_checkpoints"
        ],
        cwd=str(CODE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time
    clear_vram()

    if result.returncode != 0:
        print(f"[ERROR] Trial '{full_label}' failed:\n{result.stderr[-2500:]}")
        return {"label": full_label, "dataset": data_split, "status": "failed", "elapsed_min": round(elapsed / 60, 2)}

    run_dir = find_latest_run_dir(out_dir, start_time)
    if run_dir is None:
        return {"label": full_label, "dataset": data_split, "status": "missing_output", "elapsed_min": round(elapsed / 60, 2)}

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    shared_baseline_dir.mkdir(parents=True, exist_ok=True)
    if not shared_baseline_available:
        for fname in ["baseline_predictions.jsonl", "evaluation/nonllm_baseline.csv", "evaluation/llm_baseline.csv"]:
            src = run_dir / fname
            if src.exists():
                shutil.copy2(src, shared_baseline_dir / Path(fname).name)
    else:
        for fname in ["baseline_predictions.jsonl", "nonllm_baseline.csv", "llm_baseline.csv"]:
            src = shared_baseline_dir / fname
            dest = run_dir / fname if fname.endswith(".jsonl") else eval_dir / fname
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)

    git_dest = REPO / f"experiments/llama_3.2_3b/{data_split}"
    git_dest.mkdir(parents=True, exist_ok=True)
    for fname in ["business_eval.csv", "judge_summary.csv", "metrics_summary.csv"]:
        if (eval_dir / fname).exists():
            shutil.copy2(eval_dir / fname, git_dest / fname)
    shutil.copy2(config_path, git_dest / f"{full_label}.yaml")
    print(f"✔ Exported metrics to experiments/llama_3.2_3b/{data_split}/")

    m_path = eval_dir / "metrics_summary.csv"
    j_path = eval_dir / "judge_summary.csv"
    row = {
        "label": full_label,
        "dataset": data_split,
        "trial_name": trial_name,
        "status": "success",
        "elapsed_min": round(elapsed / 60, 2),
        "lr": trial.get("training.learning_rate", cfg["training"]["learning_rate"]),
        "epochs": trial.get("training.num_train_epochs", cfg["training"]["num_train_epochs"]),
        "lora_r": trial.get("lora.r", cfg["lora"]["r"]),
    }

    if m_path.exists():
        m_df = pd.read_csv(m_path)
        row["base_cosine_sim"] = get_metric_val(m_df, "baseline", "cosine_sim_same", "cosine_sim")
        row["ft_cosine_sim"]   = get_metric_val(m_df, "finetuned", "cosine_sim_same", "cosine_sim")
        if row.get("ft_cosine_sim") is not None and row.get("base_cosine_sim") is not None:
            row["Δ_cosine_sim"] = round(row["ft_cosine_sim"] - row["base_cosine_sim"], 4)

    if j_path.exists():
        j_df = pd.read_csv(j_path)
        row["judge_composite"]    = get_metric_val(j_df, "finetuned", "composite", "composite_score")
        row["judge_faithfulness"] = get_metric_val(j_df, "finetuned", "faithfulness")
        row["judge_specificity"]  = get_metric_val(j_df, "finetuned", "specificity")

    print(f"[COMPLETE] {full_label} in {row['elapsed_min']}m | Judge Composite: {row.get('judge_composite')}")
    return row

# ── 4. MAIN DISPATCHER ────────────────────────────────────────────────────────
def main():
    print(f"Using base config: {BASE_CONFIG_PATH}")
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV_PATH.exists():
        results_df = pd.read_csv(MASTER_CSV_PATH)
        completed = set(results_df[results_df["status"] == "success"]["label"].tolist())
        results = results_df.to_dict("records")
    else:
        results = []
        completed = set()

    DATA_SPLITS = ["raw", "clean"]

    for trial in LLAMA_TRIALS:
        for split in DATA_SPLITS:
            full_label = f"llama32_{split}_{trial['name']}"
            if full_label in completed:
                print(f"[SKIP] Trial '{full_label}' already completed.")
                continue

            row = run_single_trial_split(trial, split, base_cfg)
            results = [r for r in results if r.get("label") != full_label]
            results.append(row)
            pd.DataFrame(results).to_csv(MASTER_CSV_PATH, index=False)

    print("\n" + "="*75 + "\nALL LLAMA 3.2 DUAL-DATASET EXPERIMENTS COMPLETED\n" + "="*75)
    final_df = pd.DataFrame(results)
    disp = [c for c in ["label", "dataset", "lr", "epochs", "base_cosine_sim", "ft_cosine_sim", "Δ_cosine_sim", "judge_composite", "elapsed_min", "status"] if c in final_df.columns]
    print(final_df[disp].to_string(index=False))

if __name__ == "__main__":
    main()
