"""
run_teama_test.py — Runs the 1:1 Team A replica benchmark on Llama 3.2-3B.
"""

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

REPO = Path("/content/project")
CODE_DIR = REPO / "code"
CONFIG_PATH = CODE_DIR / "configs/teama_exact_config.yaml"
DRIVE_ROOT = Path("/content/drive/MyDrive/slm-distillation")
OUT_DIR = DRIVE_ROOT / "runs/llama_3.2_3b/teama_replication/outputs"
DEST_GIT = REPO / "experiments/llama_3.2_3b/teama_replication"

def clear_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

def find_latest_run_dir(outputs_dir: Path, since_ts: float) -> Path | None:
    if not outputs_dir.exists():
        return None
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and p.stat().st_mtime >= since_ts]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

def main():
    print("=" * 80)
    print("LAUNCHING TEAM A REPLICATION BENCHMARK")
    print("Dataset: Bitext Raw | Categories: ACCOUNT, TECHNICAL_SUPPORT, DELIVERY, CONTACT")
    print("Model: meta-llama/Llama-3.2-3B-Instruct (3 epochs, lr 2e-4)")
    print("=" * 80)

    clear_vram()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / "main.py"),
            "--phase", "1",
            "--config", str(CONFIG_PATH),
            "--device_mode", "colab",
            "--no_checkpoints"
        ],
        cwd=str(CODE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = (time.time() - start_time) / 60.0
    clear_vram()

    if result.returncode != 0:
        print(f"\n[ERROR] Replication run failed:\n{result.stderr[-3000:]}")
        return

    run_dir = find_latest_run_dir(OUT_DIR, start_time)
    if not run_dir:
        print("[ERROR] Could not locate output directory.")
        return

    eval_dir = run_dir / "evaluation"
    DEST_GIT.mkdir(parents=True, exist_ok=True)

    for fname in ["metrics_summary.csv", "judge_summary.csv", "business_eval.csv"]:
        src = eval_dir / fname
        if src.exists():
            shutil.copy2(src, DEST_GIT / fname)

    shutil.copy2(CONFIG_PATH, DEST_GIT / "teama_replica_config.yaml")
    print(f"\n✔ Run completed in {elapsed:.2f} minutes!")
    print(f"✔ Benchmark metrics exported to: {DEST_GIT}")

    print("\n" + "=" * 80)
    print("COMPARISON: YOUR REPLICATION VS REPORTED TEAM A NUMBERS")
    print("=" * 80)

    m_path = eval_dir / "metrics_summary.csv"
    j_path = eval_dir / "judge_summary.csv"

    if m_path.exists():
        print("\n[Similarity Metrics]")
        print(pd.read_csv(m_path).to_string(index=False))

    if j_path.exists():
        print("\n[LLM Judge Metrics]")
        print(pd.read_csv(j_path).to_string(index=False))

if __name__ == "__main__":
    main()
