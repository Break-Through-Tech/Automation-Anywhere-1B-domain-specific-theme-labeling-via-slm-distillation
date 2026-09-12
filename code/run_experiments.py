"""
run_experiments.py — Automated fine-tuning experiment runner for Phase 1.

WHAT THIS DOES
---------------
Instead of manually editing phase1_config.yaml and re-running main.py for
every parameter combination you want to test, this script:

  1. Takes a BASE config (your own copy, e.g. phase1_config_vd.yaml)
  2. Takes a GRID of parameter combinations you want to test
  3. For each combination:
       - copies the base config
       - overrides just the parameters you're testing
       - runs main.py with that config
       - reads the resulting metrics_summary.csv and judge_summary.csv
  4. Collects everything into ONE comparison table you can read/plot/share

WHY THIS EXISTS
----------------
Manually running 6-8 experiments one at a time is slow and error-prone
(easy to forget which config produced which result). This script makes
each experiment reproducible and comparable, and produces the evidence
table you'll want for Phase 3 (Analysis, Findings and Discussion).

HOW TO USE
-----------
1. Set BASE_CONFIG_PATH below to YOUR OWN config copy (never the shared one).
2. Edit the EXPERIMENTS list at the bottom to define what you want to test.
3. Run this script from a Colab cell:
       !python run_experiments.py
4. When done, check experiment_results.csv for the full comparison table.

COST/TIME NOTE
----------------
Each experiment with run_finetuning=True re-trains the model from scratch.
Clustering/preprocessing/label_generation are set to False in the template
below since you should only need to do those once (they don't change based
on training hyperparameters). This keeps each experiment fast and avoids
burning extra Anthropic API budget on unnecessary re-labeling.
"""

import copy
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

# ── CONFIGURE THESE ─────────────────────────────────────────────────────────
# These match the same variables your setup_repo() / commit_file() helpers use.
# If you're running this from a cell that already ran setup_repo(), you can
# just reuse REPO, CODE_DIR, BRANCH directly instead of re-typing them.

REPO = "/content/project"
CODE_DIR = "/content/project/code"       # main.py and configs/ live here
BASE_CONFIG_PATH = f"{CODE_DIR}/configs/phase1_config_vd.yaml"  # YOUR OWN copy
DRIVE_ROOT = "/content/drive/MyDrive/slm-distillation"
DEVICE_MODE = "colab"

# Each dict below is ONE experiment. Only include the keys you want to
# override from the base config — everything else stays as in your base file.
# "label" becomes part of the experiment's identifying name in the results table.
EXPERIMENTS = [
    {
        "label": "baseline",
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
        "training.learning_rate": 2.0e-4,
    },
    {
        "label": "more_epochs",
        "training.num_train_epochs": 6,
        "lora.r": 16,
        "lora.lora_alpha": 16,
        "training.learning_rate": 2.0e-4,
    },
    {
        "label": "higher_lora_rank",
        "training.num_train_epochs": 3,
        "lora.r": 32,
        "lora.lora_alpha": 32,
        "training.learning_rate": 2.0e-4,
    },
    {
        "label": "higher_learning_rate",
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
        "training.learning_rate": 5.0e-4,
    },
    {
        "label": "lower_learning_rate",
        "training.num_train_epochs": 3,
        "lora.r": 16,
        "lora.lora_alpha": 16,
        "training.learning_rate": 5.0e-5,
    },
]

# ── Implementation — you shouldn't need to edit below this line ────────────


def set_nested(cfg: dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] = value from the string 'a.b'."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def build_experiment_config(base_cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)

    # These stay fixed across all experiments in this script:
    # data prep already happened once, don't redo it or call the API again.
    cfg["pipeline"]["run_clustering"] = False
    cfg["pipeline"]["run_preprocessing"] = False
    cfg["pipeline"]["run_label_generation"] = False
    cfg["pipeline"]["run_finetuning"] = True
    cfg["pipeline"]["run_baseline_eval"] = True
    cfg["pipeline"]["run_finetuned_eval"] = True
    cfg["pipeline"]["run_llm_judge"] = True
    cfg["pipeline"]["run_business_eval"] = True
    cfg["device_mode"] = DEVICE_MODE

    for key, value in overrides.items():
        if key == "label":
            continue
        set_nested(cfg, key, value)

    return cfg


def find_latest_run_dir(outputs_dir: Path, since_ts: float) -> Path | None:
    """Find the most recently created run folder under outputs/."""
    candidates = [
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.stat().st_mtime >= since_ts
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_one_experiment(exp: dict, base_cfg: dict) -> dict:
    label = exp["label"]
    print(f"\n{'='*70}\nRunning experiment: {label}\n{'='*70}")

    cfg = build_experiment_config(base_cfg, exp)
    config_path = f"{CODE_DIR}/configs/exp_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    start_time = time.time()
    result = subprocess.run(
        [
            "python", f"{CODE_DIR}/main.py",
            "--phase", "1",
            "--config", config_path,
            "--device_mode", DEVICE_MODE,
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] Experiment '{label}' failed:")
        print(result.stderr[-2000:])  # last 2000 chars of error output
        return {"label": label, "status": "failed", "elapsed_sec": elapsed}

    # locate the run folder this experiment just created
    outputs_dir = Path(f"{DRIVE_ROOT}/outputs")
    run_dir = find_latest_run_dir(outputs_dir, start_time)
    if run_dir is None:
        print(f"[WARNING] Could not locate output folder for '{label}'")
        return {"label": label, "status": "no_output_found", "elapsed_sec": elapsed}

    metrics_path = run_dir / "evaluation" / "metrics_summary.csv"
    judge_path = run_dir / "evaluation" / "judge_summary.csv"

    row = {
        "label": label,
        "status": "success",
        "elapsed_sec": round(elapsed, 1),
        "run_dir": str(run_dir),
        **{k: v for k, v in exp.items() if k != "label"},
    }

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        test_finetuned = metrics_df[
            (metrics_df["split"] == "test") & (metrics_df["model"] == "finetuned")
        ]
        if len(test_finetuned) > 0:
            row["cosine_sim_same"] = test_finetuned["cosine_sim_same"].iloc[0]
            row["rouge_l_same"] = test_finetuned["rouge_l_same"].iloc[0]

    if judge_path.exists():
        judge_df = pd.read_csv(judge_path)
        test_finetuned = judge_df[
            (judge_df["split"] == "test") & (judge_df["model"] == "finetuned")
        ]
        if len(test_finetuned) > 0:
            row["judge_faithfulness"] = test_finetuned["faithfulness"].iloc[0]
            row["judge_specificity"] = test_finetuned["specificity"].iloc[0]
            row["judge_equivalence"] = test_finetuned["equivalence"].iloc[0]
            row["judge_composite"] = test_finetuned["composite"].iloc[0]

    print(f"[DONE] '{label}' finished in {elapsed/60:.1f} min")
    return row


def main():
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    results = []
    for exp in EXPERIMENTS:
        row = run_one_experiment(exp, base_cfg)
        results.append(row)

    results_df = pd.DataFrame(results)
    out_path = f"{CODE_DIR}/experiment_results.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n\n{'='*70}\nALL EXPERIMENTS COMPLETE\n{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
