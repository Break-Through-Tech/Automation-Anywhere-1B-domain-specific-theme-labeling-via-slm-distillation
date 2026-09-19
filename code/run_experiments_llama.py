import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import yaml

REPO = Path("/content/project")
CODE_DIR = REPO / "code"
CONFIG_BASE = CODE_DIR / "configs/llama_3.2_3b.yaml"
EXP_DIR = REPO / "experiments/llama_3.2_3b"
DRIVE_ROOT = Path("/content/drive/MyDrive/slm-distillation")
RUNS_BASE = DRIVE_ROOT / "runs/llama_3.2_3b"
CSV_PATH = EXP_DIR / "experiment_comparison_llama32.csv"

EXPERIMENTS = [
    # ── CLEAN DATASET ──
    {
        "name": "llama32_clean_default_baseline",
        "split": "clean",
        "dataset_name": "/content/drive/MyDrive/slm-distillation/data/cleaned/bitext_cleaned_support.csv",
        "lr": 3.0e-4, "epochs": 3, "r": 16, "alpha": 16
    },
    {
        "name": "llama32_clean_lr_optimal_3.5e-4",
        "split": "clean",
        "dataset_name": "/content/drive/MyDrive/slm-distillation/data/cleaned/bitext_cleaned_support.csv",
        "lr": 3.5e-4, "epochs": 3, "r": 32, "alpha": 32
    },
    {
        "name": "llama32_clean_high_capacity_4e-4_ep4",
        "split": "clean",
        "dataset_name": "/content/drive/MyDrive/slm-distillation/data/cleaned/bitext_cleaned_support.csv",
        "lr": 4.0e-4, "epochs": 4, "r": 32, "alpha": 32
    },
    # ── RAW DATASET ──
    {
        "name": "llama32_raw_default_baseline",
        "split": "raw",
        "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "lr": 3.0e-4, "epochs": 3, "r": 16, "alpha": 16
    },
    {
        "name": "llama32_raw_lr_optimal_3.5e-4",
        "split": "raw",
        "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "lr": 3.5e-4, "epochs": 3, "r": 32, "alpha": 32
    },
    {
        "name": "llama32_raw_high_capacity_4e-4_ep4",
        "split": "raw",
        "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "lr": 4.0e-4, "epochs": 4, "r": 32, "alpha": 32
    },
]

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

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_BASE, "r") as f:
        base_cfg = yaml.safe_load(f)

    for exp in EXPERIMENTS:
        name = exp["name"]
        split = exp["split"]
        print("\n" + "=" * 75)
        print(f"[EXECUTING]: {name} | Split: {split.upper()} | LR: {exp['lr']} | Epochs: {exp['epochs']}")
        print("=" * 75)

        clear_vram()
        run_out = RUNS_BASE / f"{split}_{name}/outputs"
        cfg = yaml.safe_load(yaml.dump(base_cfg))

        # Direct paths
        cfg["paths"]["data_processed"] = f"{{drive_root}}/data/processed_{split}"
        cfg["paths"]["checkpoints"] = f"{{drive_root}}/runs/llama_3.2_3b/{split}_{name}/checkpoints"
        cfg["paths"]["outputs"] = str(run_out)
        cfg["paths"]["labels_out"] = str(run_out / "labels")
        cfg["paths"]["models_out"] = str(run_out / "models")
        cfg["paths"]["evaluation_out"] = str(run_out / "evaluation")
        cfg["paths"]["hf_cache"] = "/root/.cache/huggingface"

        cfg["dataset"]["name"] = exp["dataset_name"]
        cfg["training"]["learning_rate"] = exp["lr"]
        cfg["training"]["num_train_epochs"] = exp["epochs"]
        cfg["lora"]["r"] = exp["r"]
        cfg["lora"]["lora_alpha"] = exp["alpha"]

        # Ensure label generation is skipped if bitext_labeled.csv already exists
        has_labeled = (DRIVE_ROOT / f"data/processed_{split}/bitext_labeled.csv").exists()
        cfg["pipeline"]["run_label_generation"] = not has_labeled

        # Save specific trial config
        trial_cfg_path = CODE_DIR / f"configs/{name}.yaml"
        with open(trial_cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Launch subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"
        start_ts = time.time()

        res = subprocess.run(
            [sys.executable, str(CODE_DIR / "main.py"), "--phase", "1", "--config", str(trial_cfg_path), "--device_mode", "colab", "--no_checkpoints"],
            cwd=str(CODE_DIR),
            env=env,
            capture_output=True,
            text=True
        )
        duration = (time.time() - start_ts) / 60.0
        clear_vram()

        if res.returncode != 0:
            print(f"[ERROR] Trial '{name}' failed:\n{res.stderr[-2000:]}")
            continue

        run_dir = find_latest_dir(run_out, start_ts)
        if not run_dir:
            print(f"[WARNING] Could not find run output for {name}")
            continue

        eval_dir = run_dir / "evaluation"
        dest_split = EXP_DIR / split
        dest_split.mkdir(parents=True, exist_ok=True)

        for fname in ["metrics_summary.csv", "judge_summary.csv", "business_eval.csv"]:
            src = eval_dir / fname
            if src.exists():
                shutil.copy2(src, dest_split / f"{name}_{fname}")
                shutil.copy2(src, dest_split / fname)  # Also keep canonical active file

        shutil.copy2(trial_cfg_path, dest_split / f"{name}.yaml")
        print(f"✔ Completed {name} in {duration:.2f} min. Exported to {dest_split}")

    print("\n" + "=" * 75)
    print("ALL RUNS COMPLETE ON THE TEAM PROMPT!")
    print("=" * 75)

if __name__ == "__main__":
    main()
