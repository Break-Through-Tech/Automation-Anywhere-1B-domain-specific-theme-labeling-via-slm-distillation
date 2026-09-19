"""
run_experiments_llama.py — High-efficiency, isolated hyperparameter runner for Llama 3.2-3B.
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
REPO = Path("/content/project")
CODE_DIR = REPO / "code"
BASE_CONFIG_PATH = CODE_DIR / "configs/phase1_config_vd.yaml"
DRIVE_ROOT = Path("/content/drive/MyDrive/slm-distillation")
DEVICE_MODE = "colab"

MASTER_CSV_PATH = (
    DRIVE_ROOT / "runs/llama_3.2_3b/experiment_comparison_llama32.csv"
)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── 2. HYPERPARAMETER EXPERIMENT GRID ─────────────────────────────────────────
EXPERIMENTS = [
    # 1. Raw Dataset Benchmark (Baseline vs LoRA)
    {
        "label": "llama32_raw_default_3.5e-4",
        "dataset_type": "raw",
        "student_slm.model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "training.learning_rate": 3.5e-4,
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
    # 2. Clean Dataset Standard Adaptation
    {
        "label": "llama32_clean_std_3.5e-4",
        "dataset_type": "clean",
        "student_slm.model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "training.learning_rate": 3.5e-4,
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
    },
    # 3. Clean Dataset High-Score Target (Targeting >= 4.0 - 4.5)
    {
        "label": "llama32_clean_target_4e-4_ep4",
        "dataset_type": "clean",
        "student_slm.model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "training.learning_rate": 4.0e-4,
        "training.num_train_epochs": 4,
        "lora.r": 32,
        "lora.lora_alpha": 32,
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


def build_experiment_config(
    base_cfg: dict, overrides: dict, skip_baseline_eval: bool = False
) -> tuple[dict, Path, Path]:
  cfg = copy.deepcopy(base_cfg)
  data_tag = overrides.get("dataset_type", "clean")
  is_clean = data_tag == "clean"

  # Path isolation
  out_dir = DRIVE_ROOT / f"runs/llama_3.2_3b/outputs_{data_tag}"
  chk_dir = DRIVE_ROOT / f"runs/llama_3.2_3b/checkpoints_{data_tag}"
  out_dir.mkdir(parents=True, exist_ok=True)
  chk_dir.mkdir(parents=True, exist_ok=True)

  # Dataset selection
  dataset_name = (
      "/content/drive/MyDrive/slm-distillation/data/cleaned/bitext_cleaned_support.csv"
      if is_clean
      else "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
  )
  cfg["dataset"]["name"] = dataset_name
  cfg["dataset"]["it_categories"] = ["ACCOUNT", "DELIVERY", "CONTACT"]

  cfg["paths"]["data_processed"] = f"{{drive_root}}/data/processed_{data_tag}"
  cfg["paths"]["checkpoints"] = str(chk_dir)
  cfg["paths"]["outputs"] = str(out_dir)
  cfg["paths"]["labels_out"] = str(out_dir / "labels")
  cfg["paths"]["models_out"] = str(out_dir / "models")
  cfg["paths"]["evaluation_out"] = str(out_dir / "evaluation")
  cfg["paths"]["hf_cache"] = "/root/.cache/huggingface"

  # Reuse clusters/labels if they already exist
  proc_dir = DRIVE_ROOT / f"data/processed_{data_tag}"
  if (proc_dir / "bitext_labeled.csv").exists():
    cfg["pipeline"]["run_clustering"] = False
    cfg["pipeline"]["run_preprocessing"] = False
    cfg["pipeline"]["run_label_generation"] = False
  else:
    cfg["pipeline"]["run_clustering"] = True
    cfg["pipeline"]["run_preprocessing"] = True
    cfg["pipeline"]["run_label_generation"] = True

  # Pipeline evaluation & training flags
  cfg["pipeline"]["run_finetuning"] = True
  cfg["pipeline"]["run_baseline_eval"] = not skip_baseline_eval
  cfg["pipeline"]["run_finetuned_eval"] = True
  cfg["pipeline"]["run_llm_judge"] = True
  cfg["pipeline"]["run_business_eval"] = True
  cfg["device_mode"] = DEVICE_MODE

  # Apply module overrides
  lora_cfg = cfg.setdefault("lora", {})
  if isinstance(lora_cfg.get("target_modules"), str) or not lora_cfg.get(
      "target_modules"
  ):
    lora_cfg["target_modules"] = DEFAULT_TARGET_MODULES

  for key, value in overrides.items():
    if key in ["label", "dataset_type"]:
      continue
    set_nested(cfg, key, value)

  # Hardware guards for Llama 3.2-3B on T4
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
  candidates = [
      p
      for p in outputs_dir.iterdir()
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


def run_one_experiment(exp: dict, base_cfg: dict) -> dict:
  label = exp["label"]
  data_tag = exp["dataset_type"]
  shared_dir = (
      DRIVE_ROOT / f"runs/llama_3.2_3b/checkpoints_{data_tag}/shared_baseline"
  )
  shared_baseline_available = (
      shared_dir / "baseline_predictions.jsonl"
  ).exists()

  print(
      f"\n{'='*75}\n[STARTING EXPERIMENT]: {label} (Split: {data_tag.upper()})"
  )
  if shared_baseline_available:
    print(f"[CACHE] Reusing shared baseline predictions from: {shared_dir.name}")
  print("=" * 75)
  clear_vram()

  cfg, out_dir, chk_dir = build_experiment_config(
      base_cfg, exp, skip_baseline_eval=shared_baseline_available
  )
  config_path = CODE_DIR / f"configs/exp_{label}.yaml"
  config_path.parent.mkdir(parents=True, exist_ok=True)

  with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False)

  env = os.environ.copy()
  env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"

  start_time = time.time()
  result = subprocess.run(
      [
          sys.executable,
          str(CODE_DIR / "main.py"),
          "--phase",
          "1",
          "--config",
          str(config_path),
          "--device_mode",
          DEVICE_MODE,
          "--no_checkpoints",
      ],
      cwd=str(CODE_DIR),
      env=env,
      capture_output=True,
      text=True,
  )
  elapsed = time.time() - start_time
  clear_vram()

  if result.returncode != 0:
    print(f"[ERROR] Experiment '{label}' failed:\n{result.stderr[-2500:]}")
    return {
        "label": label,
        "status": "failed",
        "elapsed_min": round(elapsed / 60, 2),
    }

  run_dir = find_latest_run_dir(out_dir, start_time)
  if run_dir is None:
    print(f"[WARNING] Could not identify output run folder for '{label}'")
    return {
        "label": label,
        "status": "missing_output",
        "elapsed_min": round(elapsed / 60, 2),
    }

  eval_dir = run_dir / "evaluation"
  eval_dir.mkdir(parents=True, exist_ok=True)

  # Cache baseline predictions for subsequent runs
  shared_dir.mkdir(parents=True, exist_ok=True)
  if not shared_baseline_available:
    for fname in [
        "baseline_predictions.jsonl",
        "evaluation/nonllm_baseline.csv",
        "evaluation/llm_baseline.csv",
    ]:
      src = run_dir / fname
      if src.exists():
        shutil.copy2(src, shared_dir / Path(fname).name)
  else:
    for fname in [
        "baseline_predictions.jsonl",
        "nonllm_baseline.csv",
        "llm_baseline.csv",
    ]:
      src = shared_dir / fname
      dest = run_dir / fname if fname.endswith(".jsonl") else eval_dir / fname
      if src.exists() and not dest.exists():
        shutil.copy2(src, dest)

  # Export summaries to GitHub experiments directory
  git_exp_dir = REPO / f"experiments/llama_3.2_3b/{data_tag}"
  git_exp_dir.mkdir(parents=True, exist_ok=True)

  for fname in [
      "business_eval.csv",
      "judge_summary.csv",
      "metrics_summary.csv",
  ]:
    if (eval_dir / fname).exists():
      shutil.copy2(eval_dir / fname, git_exp_dir / fname)

  shutil.copy2(config_path, git_exp_dir / f"{label}.yaml")
  print(f"✔ Staged evaluation CSVs to: experiments/llama_3.2_3b/{data_tag}/")

  # Parse metrics
  metrics_path = eval_dir / "metrics_summary.csv"
  judge_path = eval_dir / "judge_summary.csv"

  row = {
      "label": label,
      "status": "success",
      "dataset": data_tag,
      "elapsed_min": round(elapsed / 60, 2),
      "model": cfg["student_slm"]["model_id"],
      "lr": exp.get("training.learning_rate", cfg["training"]["learning_rate"]),
      "epochs": exp.get(
          "training.num_train_epochs", cfg["training"]["num_train_epochs"]
      ),
      "lora_r": exp.get("lora.r", cfg["lora"]["r"]),
  }

  if metrics_path.exists():
    m_df = pd.read_csv(metrics_path)
    row["base_cosine_sim"] = get_metric_val(
        m_df, "baseline", "cosine_sim_same", "cosine_sim"
    )
    row["ft_cosine_sim"] = get_metric_val(
        m_df, "finetuned", "cosine_sim_same", "cosine_sim"
    )
    row["base_rouge_l"] = get_metric_val(
        m_df, "baseline", "rouge_l_same", "rouge_l"
    )
    row["ft_rouge_l"] = get_metric_val(
        m_df, "finetuned", "rouge_l_same", "rouge_l"
    )
    if (
        row.get("ft_cosine_sim") is not None
        and row.get("base_cosine_sim") is not None
    ):
      row["Δ_cosine_sim"] = round(
          row["ft_cosine_sim"] - row["base_cosine_sim"], 4
      )

  if judge_path.exists():
    j_df = pd.read_csv(judge_path)
    row["judge_composite"] = get_metric_val(
        j_df, "finetuned", "composite", "composite_score"
    )
    row["judge_faithfulness"] = get_metric_val(
        j_df, "finetuned", "faithfulness"
    )
    row["judge_specificity"] = get_metric_val(j_df, "finetuned", "specificity")

  print(
      f"[COMPLETE] '{label}' finished in {row['elapsed_min']} min | Judge"
      f" Composite: {row.get('judge_composite')}"
  )
  return row


# ── 4. MAIN LOOP ──────────────────────────────────────────────────────────────
def main():
  if not DRIVE_ROOT.exists():
    raise RuntimeError(f"Drive root not found at {DRIVE_ROOT}.")

  with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

  MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
  if MASTER_CSV_PATH.exists():
    results_df = pd.read_csv(MASTER_CSV_PATH)
    completed = set(
        results_df[results_df["status"] == "success"]["label"].tolist()
    )
    results = results_df.to_dict("records")
  else:
    results = []
    completed = set()

  for exp in EXPERIMENTS:
    label = exp["label"]
    if label in completed:
      print(f"[SKIP] Experiment '{label}' already succeeded.")
      continue

    row = run_one_experiment(exp, base_cfg)
    results = [r for r in results if r.get("label") != label]
    results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(MASTER_CSV_PATH, index=False)

  print(f"\n\n{'='*75}\nLLAMA 3.2 EXPERIMENT SWEEP COMPLETE\n{'='*75}")
  disp_cols = [
      "label",
      "dataset",
      "lr",
      "epochs",
      "base_cosine_sim",
      "ft_cosine_sim",
      "Δ_cosine_sim",
      "judge_composite",
      "elapsed_min",
      "status",
  ]
  cols = [c for c in disp_cols if c in results_df.columns]
  print(results_df[cols].to_string(index=False))


if __name__ == "__main__":
  main()