"""
phase1/evaluation/run_summary.py

Read the finished output files of ONE pipeline run directory and flatten them into a
single dict of scalar metrics. Used by hparam_sweep.py (one row per config) and
analyze_run.py.

Keys in the returned dict
-------------------------
  hp_*                         independent variables (lora_r, learning_rate, epochs, ...)
  {split}_{metric}_{model}     e.g. test_judge_composite_finetuned, val_cosine_sim_same_baseline
                               for split in train / val / test
  {metric}_{model}             un-prefixed copy of the `primary_split` values (default: test)
  train_final_loss, val_best_eval_loss, val_final_eval_loss
  gap_eval_loss                val_best_eval_loss - train_final_loss        (>0 => overfitting)
  gap_{metric}_finetuned       train - test score gap (needs eval_all_splits: true)
  metrics_valid / metrics_warnings   False + reasons if any file was corrupt/unreadable

NOTE: train/val rows of judge_summary.csv / metrics_summary.csv only exist when the run used
`evaluation.eval_all_splits: true`; otherwise those keys are simply absent (NaN in the CSV).

Files read (all optional - missing ones simply give NaN):
  config_used.yaml                  -> hp_* (written by pipeline.py); falls back to adapter_config.json
  evaluation/judge_summary.csv      -> LLM-judge composite + dimensions per (split, model)
  evaluation/metrics_summary.csv    -> cosine / ROUGE-L / BERTScore per (split, model)
  evaluation/latency_by_cluster.csv -> per-call latency for teacher / baseline / finetuned
  evaluation/business_eval.csv      -> fine-tuning time
  models/lora_adapter/checkpoint-*/trainer_state.json -> train / eval loss
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

NAN = float("nan")
SPLITS = ("train", "val", "test")
MODELS = ("baseline", "finetuned", "teacher")
JUDGE_DIMS = ("composite", "faithfulness", "specificity", "equivalence", "coherence")
NONLLM_COLS = ("cosine_sim_same", "cosine_sim_multi", "rouge_l_same",
               "rouge_l_multi", "bertscore_f1_same", "bertscore_f1_multi")

# config_used.yaml dotted path -> hp_* key
HP_FIELDS = {
    "student_slm.model_id":                   "hp_model_id",
    "lora.r":                                 "hp_lora_r",
    "lora.lora_alpha":                        "hp_lora_alpha",
    "lora.lora_dropout":                      "hp_lora_dropout",
    "training.learning_rate":                 "hp_learning_rate",
    "training.num_train_epochs":              "hp_epochs",
    "training.per_device_train_batch_size":   "hp_batch_size",
    "training.gradient_accumulation_steps":   "hp_grad_accum",
    "training.lr_scheduler_type":             "hp_lr_scheduler",
    "training.warmup_ratio":                  "hp_warmup_ratio",
    "top_k":                                  "hp_top_k",
    "seed":                                   "hp_seed",
}


def _row(df: pd.DataFrame | None, split: str, model: str):
    if df is None or df.empty or "split" not in df.columns:
        return None
    sel = df[(df["split"] == split) & (df["model"] == model)]
    return sel.iloc[0] if len(sel) else None


def _read(path: Path, warnings: list | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        _warn(warnings, f"{path.name} unreadable ({type(e).__name__}: {e})")
        return None


def _warn(warnings: list | None, msg: str) -> None:
    logger.warning(f"[run_summary] {msg}")
    if warnings is not None:
        warnings.append(msg)


# ── Independent variables ─────────────────────────────────────────────────────

def _get_dotted(cfg: dict, dotted: str):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def collect_hyperparameters(run_dir: Path, warnings: list) -> dict:
    """Read the configuration that PRODUCED this run (config_used.yaml, else adapter_config.json)."""
    hp: dict = {}
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            for dotted, key in HP_FIELDS.items():
                v = _get_dotted(cfg, dotted)
                if v is not None:
                    hp[key] = v.strip() if isinstance(v, str) else v
            return hp
        except yaml.YAMLError as e:
            _warn(warnings, f"config_used.yaml unreadable ({e}) - falling back to adapter_config.json")

    # Fallback for older runs that predate config_used.yaml
    ac = run_dir / "models" / "lora_adapter" / "adapter_config.json"
    try:
        a = json.loads(ac.read_text(encoding="utf-8"))
        hp.update(hp_lora_r=a.get("r"), hp_lora_alpha=a.get("lora_alpha"),
                  hp_lora_dropout=a.get("lora_dropout"),
                  hp_model_id=a.get("base_model_name_or_path"))
    except FileNotFoundError:
        _warn(warnings, "no config_used.yaml or adapter_config.json - hyper-parameters unknown")
    except json.JSONDecodeError as e:
        _warn(warnings, f"adapter_config.json corrupt ({e})")
    # epochs / lr / batch size can still be recovered from trainer_state.json
    st = _load_trainer_state(run_dir, warnings)
    if st:
        hp.setdefault("hp_epochs", st.get("num_train_epochs"))
        hp.setdefault("hp_batch_size", st.get("train_batch_size"))
    return {k: v for k, v in hp.items() if v is not None}


# ── Losses (trainer_state.json) ───────────────────────────────────────────────

def _checkpoint_step(p: Path) -> int:
    mt = re.search(r"checkpoint-(\d+)", str(p))
    return int(mt.group(1)) if mt else -1


def _load_trainer_state(run_dir: Path, warnings: list) -> dict | None:
    """
    Load the trainer_state.json of the HIGHEST-step checkpoint (numeric sort: checkpoint-10 > -9).
    Returns None and records a warning if absent or corrupt - never a silent pass.
    """
    root   = run_dir / "models" / "lora_adapter"
    states = sorted(root.glob("checkpoint-*/trainer_state.json"), key=_checkpoint_step)
    if not states:
        _warn(warnings, "no trainer_state.json found - losses unavailable (run crashed before first save?)")
        return None
    try:
        return json.loads(states[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _warn(warnings, f"{states[-1]} is corrupt/half-written ({e}) - loss metrics set to NaN")
    except FileNotFoundError:
        _warn(warnings, f"{states[-1]} vanished while reading")
    except OSError as e:
        _warn(warnings, f"{states[-1]} unreadable ({e})")
    return None


def collect_losses(run_dir: Path, warnings: list) -> dict:
    out = {"train_final_loss": NAN, "val_best_eval_loss": NAN, "val_final_eval_loss": NAN,
           "gap_eval_loss": NAN}
    st = _load_trainer_state(run_dir, warnings)
    if st is None:
        return out
    hist = st.get("log_history", [])
    train = [h["loss"] for h in hist if "loss" in h]
    evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
    if train:
        out["train_final_loss"] = float(train[-1])
    if evals:
        out["val_final_eval_loss"] = float(evals[-1])
        out["val_best_eval_loss"]  = float(st["best_metric"]) if st.get("best_metric") is not None \
            else float(min(evals))
    else:
        _warn(warnings, "trainer_state.json has no eval_loss entries")
    if not (np.isnan(out["train_final_loss"]) or np.isnan(out["val_best_eval_loss"])):
        out["gap_eval_loss"] = out["val_best_eval_loss"] - out["train_final_loss"]
    return out


# ── Main entry ────────────────────────────────────────────────────────────────

def collect_run_metrics(run_dir: str | Path, split: str = "test") -> dict:
    """
    Flatten one run into a dict. All of train/val/test are extracted (prefixed keys);
    `split` only chooses which one is ALSO exposed un-prefixed for backwards compatibility.
    """
    run_dir  = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    warnings: list[str] = []
    m: dict = {"run_dir": str(run_dir)}

    # 1) independent variables
    m.update(collect_hyperparameters(run_dir, warnings))

    # 2) scores per split
    judge   = _read(eval_dir / "judge_summary.csv", warnings)
    metrics = _read(eval_dir / "metrics_summary.csv", warnings)
    for sp in SPLITS:
        for tag in MODELS:
            jr = _row(judge, sp, tag)
            if jr is not None:
                for dim in JUDGE_DIMS:
                    if dim in jr.index and pd.notna(jr[dim]):
                        m[f"{sp}_judge_{dim}_{tag}"] = float(jr[dim])
            mr = _row(metrics, sp, tag)
            if mr is not None:
                for col in NONLLM_COLS:
                    if col in mr.index and pd.notna(mr[col]):
                        m[f"{sp}_{col}_{tag}"] = float(mr[col])

    # un-prefixed copy of the primary split (used by sweep ranking / existing code)
    pref = f"{split}_"
    for k in [k for k in m if k.startswith(pref)]:
        m[k[len(pref):]] = m[k]

    # generalisation gap in score space (train - test); positive => memorising
    for metric in ("judge_composite", "cosine_sim_same", "rouge_l_same"):
        a, b = m.get(f"train_{metric}_finetuned"), m.get(f"test_{metric}_finetuned")
        if a is not None and b is not None:
            m[f"gap_{metric}_finetuned"] = a - b

    # 3) latency (seconds per label)
    lat = _read(eval_dir / "latency_by_cluster.csv", warnings)
    if lat is not None:
        if "split" in lat.columns:
            lat_split = lat[lat["split"] == split]
            lat = lat_split if len(lat_split) else lat
        for tag in ("teacher", "baseline", "finetuned"):
            col = f"{tag}_latency_s"
            if col in lat.columns and lat[col].notna().any():
                m[f"latency_mean_s_{tag}"] = float(lat[col].mean())
                m[f"latency_median_s_{tag}"] = float(lat[col].median())
                m[f"latency_p95_s_{tag}"]  = float(lat[col].quantile(0.95))

    # 4) fine-tuning time
    biz = _read(eval_dir / "business_eval.csv", warnings)
    if biz is not None and "metric" in biz.columns:
        d = dict(zip(biz["metric"], biz["value"]))
        if "finetuning_wall_time_min" in d:
            m["finetune_minutes"] = float(d["finetuning_wall_time_min"])

    # 5) losses
    m.update(collect_losses(run_dir, warnings))

    m["metrics_valid"]    = not warnings
    m["metrics_warnings"] = " | ".join(warnings)
    return m


def primary_value(metrics: dict, primary_metric: str, overfit_penalty: float = 0.0) -> float:
    """
    Map a sweep `primary_metric` name to the fine-tuned-SLM value.
    overfit_penalty > 0 subtracts penalty * max(0, train-test gap) so memorising configs rank lower
    (lower-is-better metrics get the penalty added instead). Gap needs eval_all_splits: true;
    if unavailable it falls back to the val-loss gap, else no penalty.
    """
    mapping = {
        "judge_composite": "judge_composite_finetuned",
        "cosine_sim":      "cosine_sim_same_finetuned",
        "rouge_l":         "rouge_l_same_finetuned",
        "latency_s":       "latency_mean_s_finetuned",
        "eval_loss":       "val_best_eval_loss",
    }
    key = mapping.get(primary_metric, primary_metric)
    val = float(metrics.get(key, NAN))
    if overfit_penalty and not np.isnan(val):
        gap = metrics.get({"judge_composite": "gap_judge_composite_finetuned",
                           "cosine_sim": "gap_cosine_sim_same_finetuned",
                           "rouge_l": "gap_rouge_l_same_finetuned"}.get(primary_metric, ""), NAN)
        if gap is None or np.isnan(gap):
            gap = metrics.get("gap_eval_loss", NAN)    # weaker proxy, loss scale
        if gap is not None and not np.isnan(gap):
            pen = overfit_penalty * max(0.0, float(gap))
            val = val + pen if lower_is_better(primary_metric) else val - pen
    return val


def lower_is_better(primary_metric: str) -> bool:
    return primary_metric in ("latency_s", "eval_loss")


def per_item_scores(run_dir: str | Path, tag: str = "finetuned",
                    metric: str = "composite") -> pd.Series:
    """
    Per-(cluster, prompt) score series for stats tests.
    metric: "composite" (LLM judge) | "cosine_sim_same" | "rouge_l_same" | "latency"
    Index = "cluster_id|prompt_id".
    """
    eval_dir = Path(run_dir) / "evaluation"
    if metric == "composite":
        df, col = _read(eval_dir / "llm_by_cluster.csv"), f"composite_{tag}"
    elif metric == "latency":
        df, col = _read(eval_dir / "latency_by_cluster.csv"), f"{tag}_latency_s"
    else:
        df, col = _read(eval_dir / "nonllm_by_cluster.csv"), f"{metric}_{tag}"
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    key = df["cluster_id"].astype(str) + "|" + df["prompt_id"].astype(str)
    return pd.Series(df[col].values, index=key, name=tag).dropna()
