"""
phase1/evaluation/latency.py

THE place that measures latency for all three models.

  timed_slm_generate()   one SLM label (baseline OR fine-tuned) with GPU-synchronised timing
  timed_teacher_call()   one teacher-LLM API label (wall-clock incl. network + retries)
  benchmark_latency()    side-by-side benchmark of Teacher / Baseline SLM / Fine-tuned SLM
                         on the same prompts, returning a DataFrame + summary stats

Why synchronise?  CUDA/MPS kernels are asynchronous; without torch.cuda.synchronize() a timer
can stop before the GPU has finished. generate() usually syncs when it copies tokens back, but
doing it explicitly makes the measurement defensible.

What is (not) included
  included    : tokenisation, prefill, decoding, detokenisation (SLM); network + server time (teacher)
  NOT included: model load time (recorded separately by BusinessEvaluator.record_model_load_time)

Usage
-----
from phase1.evaluation.latency import benchmark_latency
df, summary = benchmark_latency(cfg, adapter_dir="outputs/<run>/models/lora_adapter",
                                n_prompts=20, merge_adapter=True)
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sync_device(device_mode: str) -> None:
    import torch
    try:
        if device_mode == "colab" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_mode == "local_mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001  (never let timing code break a run)
        pass


def timed_slm_generate(prompt_str: str, model, tokenizer, cfg: dict, device_mode: str) -> dict:
    """Generate one label and time it. Returns label, latency_s, prompt/new tokens, tokens_per_s."""
    from phase1.finetuning.trainer import generate_label

    sync_device(device_mode)
    t0 = time.perf_counter()
    label = generate_label(prompt_str, model, tokenizer, cfg, device_mode)
    sync_device(device_mode)
    latency = time.perf_counter() - t0

    n_new = len(tokenizer(label, add_special_tokens=False)["input_ids"]) if label else 0
    n_in  = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
    return {"label": label, "latency_s": latency, "prompt_tokens": n_in, "new_tokens": n_new,
            "tokens_per_s": n_new / latency if latency > 0 else float("nan")}


def timed_teacher_call(llm_caller, messages: list[dict], llm_cfg: dict) -> dict:
    """Call the teacher API once and time it (network + queueing + generation)."""
    t0 = time.perf_counter()
    label = llm_caller(messages, llm_cfg)
    return {"label": label.strip(), "latency_s": time.perf_counter() - t0}


def summarize(lat: pd.Series) -> dict:
    lat = lat.dropna()
    if lat.empty:
        return {}
    return {"n": len(lat), "mean_s": lat.mean(), "median_s": lat.median(),
            "std_s": lat.std(ddof=1) if len(lat) > 1 else float("nan"),
            "p50_s": lat.quantile(.5), "p95_s": lat.quantile(.95), "p99_s": lat.quantile(.99),
            "min_s": lat.min(), "max_s": lat.max(), "labels_per_min": 60.0 / lat.mean()}


def benchmark_latency(
    cfg: dict,
    adapter_dir: str,
    n_prompts: int = 20,
    n_warmup: int = 3,
    prompt_id: str = "P1",
    include_teacher: bool = True,
    merge_adapter: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Benchmark Teacher LLM, Baseline SLM and Fine-tuned SLM on identical prompts.

    merge_adapter=True also benchmarks the fine-tuned model after merge_and_unload()
    ("finetuned_merged") to show how much latency the un-merged LoRA layers cost.

    Needs a finished run (labeled_data.csv is read from the adapter's run directory).
    Returns (per_call_df, summary_df).
    """
    from pathlib import Path

    from peft import PeftModel

    from phase1.data.schema import CLUSTER_ID, TICKET_DETAILS, TICKET_RANK
    from phase1.finetuning.trainer import load_model_and_tokenizer
    from phase1.labeling.frontier_llm import _get_llm_caller
    from phase1.prompts.templates import build_inference_prompt, build_messages

    device_mode = cfg["device_mode"]
    domain, k   = cfg["dataset"]["domain"], cfg["top_k"]
    run_dir     = Path(adapter_dir).parent.parent
    labeled     = run_dir / "labeled_data.csv"
    df = pd.read_csv(labeled)
    sets = [g.sort_values(TICKET_RANK)[TICKET_DETAILS].tolist()
            for _, g in df[df[TICKET_RANK] <= k].groupby(CLUSTER_ID)][:n_prompts]
    logger.info(f"[latency] {len(sets)} prompt(s), {n_warmup} warm-up, prompt {prompt_id}")

    rows: list[dict] = []

    def run_slm(name, model, tok):
        prompts = [build_inference_prompt(prompt_id, t, cfg, tok, domain) for t in sets]
        for p in prompts[:1] * n_warmup:                      # warm-up (not recorded)
            timed_slm_generate(p, model, tok, cfg, device_mode)
        for i, p in enumerate(prompts):
            r = timed_slm_generate(p, model, tok, cfg, device_mode)
            rows.append({"model": name, "item": i, **{k_: v for k_, v in r.items() if k_ != "label"}})

    # Teacher
    if include_teacher:
        caller = _get_llm_caller(cfg["teacher_llm"]["provider"], cfg["teacher_llm"])
        for i, t in enumerate(sets):
            r = timed_teacher_call(caller, build_messages(prompt_id, t, cfg, domain), cfg["teacher_llm"])
            rows.append({"model": "teacher", "item": i, "latency_s": r["latency_s"]})
            time.sleep(cfg["teacher_llm"].get("sleep_between_calls", 0))  # sleep NOT timed

    # Baseline SLM
    base, tok = load_model_and_tokenizer(cfg)
    run_slm("baseline", base, tok)

    # Fine-tuned SLM (adapter attached, un-merged)
    ft = PeftModel.from_pretrained(base, str(adapter_dir))
    ft.eval()
    run_slm("finetuned", ft, tok)

    # Fine-tuned SLM merged into base weights
    if merge_adapter:
        merged = ft.merge_and_unload()
        merged.eval()
        run_slm("finetuned_merged", merged, tok)

    per_call = pd.DataFrame(rows)
    summary = pd.DataFrame([{"model": m, **summarize(g["latency_s"])}
                            for m, g in per_call.groupby("model")])
    if "teacher" in set(summary["model"]):
        t_mean = summary.loc[summary["model"] == "teacher", "mean_s"].iloc[0]
        summary["speedup_vs_teacher"] = t_mean / summary["mean_s"]
    return per_call, summary
