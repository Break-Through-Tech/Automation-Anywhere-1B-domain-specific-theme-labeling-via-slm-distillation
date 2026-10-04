#!/usr/bin/env python3
"""Benchmark saved Gemma 3 adapters, baseline and Anthropic teacher; never train.

Add this file to code/benchmark_latency.py. Run --help for options.
The default workload is the saved cleaned test split, not a new quality holdout.
Pure preparation/statistics functions use only the Python standard library.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

RUNS = {
    "cleaned_2e5": "gemma3_2.0e-5_2epochs_2000sample_auged_2var_completion_only_loss_0.1dropout_cleaned",
    "cleaned_5e5": "gemma3_5.0e-5_2epochs_2000sample_auged_3var_completion_only_loss_0.1dropout_cleaned",
    "original_5e5": "gemma3_5.0e-5_2epochs_2000sample_auged_3var_completion_only_loss_0.1dropout",
}
PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICE_CHECKED = "2026-10-03"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(target)


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def messages_for(cfg, pid, tickets):
    # Matches phase1/prompts/templates.py; deliberately no training imports.
    p = cfg["prompts"]
    system = p["system"].strip()
    if p.get("include_domain_in_prompt") and cfg["dataset"].get("domain"):
        system = f"You are analysing {cfg['dataset']['domain']} support tickets. " + system
    numbered = "\n".join(f"Ticket {i + 1}: {t.strip()}" for i, t in enumerate(tickets))
    if "{tickets}" not in p[pid]:
        raise ValueError(f"{pid} template has no tickets placeholder")
    return [{"role": "system", "content": system},
            {"role": "user", "content": p[pid].replace("{tickets}", numbered)}]


def prepare_workload(run, cfg, prompt_ids):
    labels = read_csv(Path(run) / "evaluation/labels_by_cluster.csv")
    selected = [r for r in labels if r["split"] == "test" and r["prompt_id"] in prompt_ids]
    keys = [(int(r["cluster_id"]), r["prompt_id"]) for r in selected]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Saved test split is empty or contains duplicate cluster/prompt keys")
    clusters = sorted({k[0] for k in keys})
    expected = {(c, p) for c in clusters for p in prompt_ids}
    if set(keys) != expected:
        raise ValueError("Saved test split does not cover every requested prompt for every cluster")
    if any(int(r["cluster_id"]) in clusters and r["split"] != "test" for r in labels):
        raise ValueError("A test cluster also appears in another saved split")
    rows = read_csv(Path(run) / "labeled_data.csv")
    k = int(cfg["top_k"])
    if k < 1:
        raise ValueError("top_k must be positive")
    requests = []
    for cid in clusters:
        ranked = sorted([r for r in rows if int(r["cluster_id"]) == cid
                         and 1 <= int(r["ticket_rank_in_cluster"]) <= k],
                        key=lambda r: int(r["ticket_rank_in_cluster"]))
        if [int(r["ticket_rank_in_cluster"]) for r in ranked] != list(range(1, k + 1)):
            raise ValueError(f"Cluster {cid}: missing/duplicate top-{k} ticket ranks")
        tickets = [r["ticket_details"].strip() for r in ranked]
        if not all(tickets):
            raise ValueError(f"Cluster {cid}: empty ticket")
        for pid in prompt_ids:
            requests.append({"cluster_id": cid, "prompt_id": pid, "tickets": tickets,
                             "messages": messages_for(cfg, pid, tickets)})
    return requests


def adapter_info(path):
    path = Path(path)
    cfg_path = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    config = json.loads(cfg_path.read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"Unexpected adapter type: {path}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return {"path": str(path), "config": config, "config_sha256": sha_file(cfg_path),
            "weights_sha256": sha_file(weights)}


def resolve_base_model(infos, cfg, override=None):
    known = {v["config"].get("base_model_name_or_path") for v in infos.values()}
    known = {v.strip() for v in known if isinstance(v, str) and v.strip()}
    if len(known) > 1:
        raise ValueError(f"Adapter base IDs differ: {known}; resolve before comparing")
    explicit = override.strip() if isinstance(override, str) else None
    if explicit and known and explicit not in known:
        raise ValueError("Explicit base model conflicts with saved adapter metadata")
    configured = cfg.get("student_slm", {}).get("model_id")
    chosen = explicit or next(iter(known), None) or configured
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("No valid base model ID: supply --base-model or student_slm.model_id")
    return chosen.strip()


def native_bf16_supported(cuda):
    # Recent PyTorch defaults to including emulation, which is unsuitable for
    # choosing the benchmark's fast compute dtype on T4 (compute capability 7.5).
    try:
        return cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return cuda.get_device_capability(0)[0] >= 8


def redact_error(text):
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def quantile(values, p):
    a = sorted(values)
    if not a:
        return None
    x = (len(a) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def api_cost(usage, input_rate, output_rate):
    """USD from API usage; support Haiku cache multipliers, never assume missing input/output."""
    if not isinstance(usage, dict) or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    read = usage.get("cache_read_input_tokens") or 0
    created = usage.get("cache_creation_input_tokens") or 0
    details = usage.get("cache_creation") or {}
    short = details.get("ephemeral_5m_input_tokens") or 0
    long = details.get("ephemeral_1h_input_tokens") or 0
    if created != short + long:
        # No caching is requested by this script. Unexpected usage cannot be priced safely.
        return None
    return (input_rate * (usage["input_tokens"] + .1 * read + 1.25 * short + 2 * long)
            + output_rate * usage["output_tokens"]) / 1_000_000


def summarize(records, hourly_rate):
    result = []
    for name in sorted({r["model"] for r in records}):
        measured = [r for r in records if r["model"] == name and r["phase"] == "measured"]
        ok = [r for r in measured if r["status"] == "ok"]
        if not measured:
            continue
        lat = [r["latency_s"] for r in ok]
        gen = [r["generation_s"] for r in ok if r.get("generation_s") is not None]
        costs = [r["estimated_api_cost_usd"] for r in ok if r.get("estimated_api_cost_usd") is not None]
        per_label = (statistics.mean(costs) if name == "teacher" and len(costs) == len(ok) and ok
                     else statistics.mean(lat) / 3600 * hourly_rate
                     if name != "teacher" and hourly_rate is not None and lat else None)
        result.append({
            "model": name, "attempts": len(measured), "successful_labels": len(ok),
            "errors_or_empty": len(measured) - len(ok),
            "mean_s": statistics.mean(lat) if lat else None,
            "median_s": quantile(lat, .5), "p95_s": quantile(lat, .95),
            "stdev_s": statistics.stdev(lat) if len(lat) > 1 else None,
            "generation_mean_s": statistics.mean(gen) if gen else None,
            "mean_input_tokens": statistics.mean(r["input_tokens"] for r in ok) if ok else None,
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in ok) if ok else None,
            "output_tokens_per_generation_second": sum(r["output_tokens"] for r in ok) / sum(gen) if gen else None,
            "serial_labels_per_minute": 60 / statistics.mean(lat) if lat else None,
            "token_limit_reached": sum(bool(r.get("token_limit_reached")) for r in ok),
            "peak_allocated_gib": max((r.get("peak_allocated_gib", 0) for r in ok), default=0) if name != "teacher" else None,
            "estimated_serving_usd_per_1000_labels": per_label * 1000 if per_label is not None else None,
            "cost_basis": "API-reported usage, configured list prices" if name == "teacher" else "busy request seconds × supplied GPU hourly rate; excludes loading/idle/training",
        })
    return result


def format_num(x):
    return "N/A" if x is None else f"{x:.4f}"


def save_reports(out, records, loads, manifest, rate):
    summary = summarize(records, rate)
    write_csv(out / "summary.csv", summary)
    rounds = []
    for rnd in sorted({r["round"] for r in records if r["phase"] == "measured"}):
        for row in summarize([r for r in records if r["round"] == rnd], rate):
            rounds.append({"round": rnd, **row})
    write_csv(out / "summary_by_round.csv", rounds)
    write_csv(out / "requests.csv", records)
    write_csv(out / "model_loads.csv", loads)
    teacher = [r for r in records if r["model"] == "teacher"]
    known_cost = sum(r.get("estimated_api_cost_usd") or 0 for r in teacher)
    unknown = sum(r.get("estimated_api_cost_usd") is None for r in teacher)
    wall = manifest.get("elapsed_s", 0)
    cost = {
        "teacher_cost_with_known_usage_usd": known_cost,
        "teacher_warmup_cost_with_known_usage_usd": sum(r.get("estimated_api_cost_usd") or 0 for r in teacher if r["phase"] == "warmup"),
        "teacher_calls_without_priceable_usage": unknown,
        "teacher_cost_complete": unknown == 0,
        "gpu_hourly_usd_supplied": rate,
        "script_elapsed_s": wall,
        "estimated_GPU_allocation_cost_during_script_usd": wall / 3600 * rate if rate is not None else None,
        "estimated_combined_script_cost_usd": known_cost + wall / 3600 * rate if rate is not None and not unknown else None,
        "notes": "API cost is a usage-based list-price estimate, not an invoice. Unknown/error calls may still be billed. GPU wall cost includes loading, warmups, API waits and logging while this script runs, but excludes notebook setup/idle outside the script. Serving estimates assume sequential fully utilized requests. No training cost or subscription allocation is inferred. Tokenizers differ across teacher/student; token throughput is not directly interchangeable.",
    }
    write_json(out / "cost_summary.json", cost)
    write_json(out / "manifest.json", manifest)
    lines = ["# Latency and cost benchmark", "", f"Status: **{manifest['status']}**", "",
             "All latencies are completed-label latency, not time-to-first-token. Warmups and failed/empty requests are excluded from latency summaries; failures are counted separately. Teacher time includes network/API wait; SLM time includes tokenization, transfer, generation and decoding, excluding prompt rendering and disk writes.", "",
             "| Model | Successful / attempted | Mean s | Median s | p95 s | Estimated USD / 1,000 labels |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['successful_labels']}/{r['attempts']} | {format_num(r['mean_s'])} | {format_num(r['median_s'])} | {format_num(r['p95_s'])} | {format_num(r['estimated_serving_usd_per_1000_labels'])} |")
    lines += ["", f"Teacher known-usage cost, including warmup: ${known_cost:.6f}. Calls without priceable usage: {unknown}.",
              "", "GPU cost is N/A unless an hourly rate was supplied. A zero rate explicitly means zero incremental cash cost, not zero resource use.",
              "", "See manifest.json for exact settings, versions, workload hashes and hardware; requests.jsonl for incremental raw records; model_loads.csv for loading/validation/warmup times; summary_by_round.csv for repeat variability. Repetitions measure runtime variability, not new independent quality examples.",
              "", "No judge calls, retraining, adapter merging, or original experiment file overwrites were performed."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).parent / "configs/phase1_config.yaml"))
    p.add_argument("--drive-root", default="/content/drive/MyDrive/slm-distillation")
    p.add_argument("--source-run", help="Saved run supplying test requests; default cleaned_2e5")
    p.add_argument("--adapter", action="append", metavar="NAME=PATH", help="Override all three default adapters; repeat for each")
    p.add_argument("--prompts", default="P1,P2,P3,P4,P5")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=5, help="Per local model load; excluded from request summaries")
    p.add_argument("--teacher-warmups", type=int, default=1, help="API calls; billed but excluded from request summaries")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--teacher-model", help="Default from config, must be Claude Haiku 4.5")
    p.add_argument("--teacher-input-usd-per-million", type=float, default=1.0)
    p.add_argument("--teacher-output-usd-per-million", type=float, default=5.0)
    p.add_argument("--teacher-gap-s", type=float, default=1.0, help="Rate-limit pacing outside measured latency")
    p.add_argument("--teacher-timeout-s", type=float, default=60.0)
    p.add_argument("--gpu-hourly-usd", type=float, default=None, help="Your actual/effective GPU rate; omitted = unknown, NOT zero")
    p.add_argument("--compute-dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--base-revision", default=None, help="HF commit/tag; resolved to a commit before timing")
    p.add_argument("--base-model", default=None, help="Explicit base ID when adapter metadata is missing; otherwise falls back to student_slm.model_id")
    p.add_argument("--cache-dir", default="/content/slm_latency_hf_cache", help="Local Colab disk cache for repeated loads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Local file checks only; no model downloads, API calls or output writes")
    a = p.parse_args()
    if a.repeats < 1 or a.warmups < 1 or a.teacher_warmups < 0 or a.max_new_tokens < 1:
        p.error("Invalid repeats, warmups or token limit")
    for k in ["teacher_input_usd_per_million", "teacher_output_usd_per_million", "teacher_gap_s", "gpu_hourly_usd"]:
        v = getattr(a, k)
        if v is not None and (not math.isfinite(v) or v < 0):
            p.error(f"{k} must be finite and nonnegative")
    if not math.isfinite(a.teacher_timeout_s) or a.teacher_timeout_s <= 0:
        p.error("teacher timeout must be positive")
    return a


def main():
    args = arguments()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.drive_root).expanduser().resolve()
    source = Path(args.source_run).expanduser().resolve() if args.source_run else root / "outputs" / RUNS["cleaned_2e5"]
    paths = {k: root / "outputs" / v / "models/lora_adapter" for k, v in RUNS.items()}
    if args.adapter:
        paths = {}
        for entry in args.adapter:
            name, sep, path = entry.partition("=")
            if not sep or not name or name in paths or name in ("baseline", "teacher"):
                raise ValueError("Use unique --adapter NAME=PATH; baseline and teacher are reserved")
            paths[name] = Path(path).expanduser().resolve()
    infos = {name: adapter_info(path) for name, path in paths.items()}
    base_id = resolve_base_model(infos, cfg, args.base_model)
    first = next(iter(paths.values()))
    # A common saved tokenizer keeps input IDs identical for all local models.
    tokenizer_hashes = {}
    for name, path in paths.items():
        tokenizer_hashes[name] = {f: sha_file(path / f) for f in ("tokenizer.json", "chat_template.jinja")}
    if any(v != next(iter(tokenizer_hashes.values())) for v in tokenizer_hashes.values()):
        raise ValueError("Saved tokenizers/chat templates differ; choose a deliberate shared tokenizer before benchmarking")
    pids = args.prompts.split(",")
    if len(set(pids)) != len(pids) or any(p not in ("P1", "P2", "P3", "P4", "P5") for p in pids):
        raise ValueError("--prompts must be unique P1–P5 IDs separated by commas")
    workload = prepare_workload(source, cfg, pids)
    teacher_model = args.teacher_model or cfg["teacher_llm"]["model"]
    if not args.skip_teacher and not teacher_model.startswith("claude-haiku-4-5"):
        raise ValueError("This benchmark's teacher pricing is for Claude Haiku 4.5 only")
    print(f"Workload: {len(workload)} requests; {len({r['cluster_id'] for r in workload})} clusters", flush=True)
    print(f"SLM: {len(paths)+1} models × {len(workload)} × {args.repeats} measured calls", flush=True)
    print(f"Teacher: {0 if args.skip_teacher else len(workload)*args.repeats+args.teacher_warmups} maximum paid calls; no automatic retries", flush=True)
    print(f"Base: {base_id}; GPU hourly rate: {args.gpu_hourly_usd if args.gpu_hourly_usd is not None else 'unknown'}", flush=True)
    if args.dry_run:
        print("File checks passed; no GPU/model/API work performed.")
        return
    if not args.skip_teacher and not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Set ANTHROPIC_API_KEY from Colab Secrets first")
    import torch
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration, GenerationConfig
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; CPU fallback is deliberately disabled")
    torch.cuda.set_device(0)
    bf16 = native_bf16_supported(torch.cuda)
    dtype_name = ("bfloat16" if bf16 else "float16") if args.compute_dtype == "auto" else args.compute_dtype
    if dtype_name == "bfloat16" and not bf16:
        raise RuntimeError("This GPU does not natively support bfloat16; use --compute-dtype float16")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = root / "latency_benchmarks" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"Results → {out}", flush=True)
    records, loads = [], []
    manifest = {"status": "running", "created_utc": stamp, "arguments": vars(args), "source_run": str(source),
                "adapters": infos, "saved_tokenizer_hashes": tokenizer_hashes,
                "workload_sha256": digest(workload), "script_sha256": sha_file(__file__),
                "source_labels_sha256": sha_file(source / "evaluation/labels_by_cluster.csv"),
                "source_tickets_sha256": sha_file(source / "labeled_data.csv"),
                "base_model": base_id, "compute_dtype": dtype_name,
                "backend": "Transformers Gemma3ForConditionalGeneration + PEFT, NF4 double quant, unmerged adapters",
                "gpu": torch.cuda.get_device_name(0), "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory/2**30,
                "cuda": torch.version.cuda, "python": sys.version,
                "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "bitsandbytes", "safetensors", "huggingface-hub"]},
                "price_source": PRICE_SOURCE, "price_checked": PRICE_CHECKED,
                "teacher_model_requested": teacher_model, "teacher_prompt_cache_requested": False,
                "timing_definition": "SLM: tokenize+transfer+generate+decode; teacher: complete nonstreaming API request including network. Prompt construction, pacing and writes excluded. No TTFT metric.",
                "config_used": {k: cfg[k] for k in ["prompts", "dataset", "student_slm", "top_k", "teacher_llm"]}}
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(args.config).resolve().parent, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    write_json(out / "manifest.json", manifest)
    client = None
    try:
        # Pin the resolved base revision across all twelve model loads.
        bc = AutoConfig.from_pretrained(base_id, revision=args.base_revision, cache_dir=args.cache_dir)
        if bc.model_type != "gemma3":
            raise ValueError(f"Expected Gemma 3 conditional-generation base, got {bc.model_type}")
        revision = getattr(bc, "_commit_hash", None)
        if not revision:
            raise ValueError("Could not resolve an immutable base-model revision")
        manifest["base_revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(str(first), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(cfg["student_slm"]["max_seq_length"])
        for r in workload:
            r["prompt_text"] = tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            # Match the repo's tokenizer call, including its default special-token behavior.
            ids = tokenizer(r["prompt_text"], truncation=False)["input_ids"]
            if len(ids) > max_length:
                raise ValueError(f"Cluster {r['cluster_id']}/{r['prompt_id']} exceeds {max_length} tokens; refuse silent truncation")
            r["input_tokens"] = len(ids)
            r["input_ids_sha256"] = digest(ids)
        manifest["rendered_workload_sha256"] = digest(workload)
        write_json(out / "workload.json", workload)
        # Avoid inheriting sampling or suppress-token settings from unrelated generations.
        gen_cfg = GenerationConfig.from_pretrained(base_id, revision=revision, cache_dir=args.cache_dir)
        eos = gen_cfg.eos_token_id or tokenizer.eos_token_id
        if not isinstance(eos, list):
            eos = [eos]
        eos = [int(e) for e in eos if e is not None]
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos = sorted(set(eos + [eot]))
        if not eos:
            raise ValueError("No EOS token IDs found")
        common_generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False,
                                             num_beams=1, use_cache=True, eos_token_id=eos,
                                             pad_token_id=tokenizer.eos_token_id)
        manifest["generation_config"] = common_generation.to_dict()
        teacher_temp = float(cfg["teacher_llm"].get("temperature", .3))
        manifest["teacher_temperature"] = teacher_temp
        if not args.skip_teacher:
            import anthropic
            manifest["versions"]["anthropic"] = anthropic.__version__
            client = anthropic.Anthropic(max_retries=0, timeout=args.teacher_timeout_s)

        def record(row):
            records.append(row)
            append_json(out / "requests.jsonl", row)

        def teacher_call(r, rnd, phase):
            t0 = time.perf_counter()
            try:
                response = client.messages.create(model=teacher_model, max_tokens=args.max_new_tokens,
                    temperature=teacher_temp, system=r["messages"][0]["content"], messages=r["messages"][1:])
                label = "".join(b.text for b in response.content if b.type == "text").strip()
                elapsed = time.perf_counter() - t0
                usage = response.usage.model_dump()
                row = {"model": "teacher", "round": rnd, "phase": phase, "cluster_id": r["cluster_id"], "prompt_id": r["prompt_id"],
                       "status": "ok" if label else "empty", "latency_s": elapsed,
                       "input_tokens": usage["input_tokens"] + (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0),
                       "output_tokens": usage["output_tokens"], "label": label, "usage": usage,
                       "estimated_api_cost_usd": api_cost(usage, args.teacher_input_usd_per_million, args.teacher_output_usd_per_million),
                       "resolved_model": response.model, "stop_reason": response.stop_reason,
                       "token_limit_reached": response.stop_reason == "max_tokens", "request_id": getattr(response, "_request_id", None)}
            except Exception as e:
                row = {"model": "teacher", "round": rnd, "phase": phase, "cluster_id": r["cluster_id"], "prompt_id": r["prompt_id"],
                       "status": "error", "latency_s": time.perf_counter() - t0,
                       "error_type": type(e).__name__, "http_status": getattr(e, "status_code", None), "estimated_api_cost_usd": None}
                # No retry: avoids hidden latency and duplicate billed requests.
            record(row)
            if row["status"] != "ok":
                print(f"Teacher {row['status']}: {row.get('error_type', 'empty output')}", flush=True)
                if row.get("http_status") in (400, 401, 403, 404):
                    raise RuntimeError("Teacher configuration/authentication error; see saved request metadata")
            if args.teacher_gap_s:
                time.sleep(args.teacher_gap_s)

        def local_request(model, r):
            torch.cuda.synchronize()
            start = time.perf_counter()
            inputs = tokenizer(r["prompt_text"], return_tensors="pt", truncation=False)
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            torch.cuda.synchronize()
            g0 = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=common_generation)
            torch.cuda.synchronize()
            generation_s = time.perf_counter() - g0
            ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            label = tokenizer.decode(ids, skip_special_tokens=True).strip()
            elapsed = time.perf_counter() - start
            return {"status": "ok" if label else "empty", "latency_s": elapsed, "generation_s": generation_s,
                    "input_tokens": r["input_tokens"], "output_tokens": len(ids), "label": label,
                    "token_limit_reached": len(ids) >= args.max_new_tokens and (not ids or ids[-1] not in eos)}

        names = ["baseline"] + list(paths)
        random.Random(args.seed).shuffle(names)
        if client:
            for w in range(args.teacher_warmups):
                teacher_call(workload[w % len(workload)], 0, "warmup")
        for rnd in range(1, args.repeats + 1):
            requests = workload.copy()
            random.Random(args.seed + rnd).shuffle(requests)
            order = names[(rnd - 1) % len(names):] + names[:(rnd - 1) % len(names)]
            for name in order:
                print(f"Round {rnd}/{args.repeats}: loading {name}", flush=True)
                model = None
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    t0 = time.perf_counter()
                    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
                    model = Gemma3ForConditionalGeneration.from_pretrained(base_id, revision=revision, cache_dir=args.cache_dir,
                        quantization_config=bnb, torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
                    if not getattr(model, "is_loaded_in_4bit", False):
                        raise RuntimeError("The base did not load in 4-bit mode")
                    if name != "baseline":
                        model = PeftModel.from_pretrained(model, str(paths[name]), is_trainable=False)
                    model.eval()
                    model.requires_grad_(False)
                    torch.cuda.synchronize()
                    load_s = time.perf_counter() - t0
                    validation_start = time.perf_counter()
                    if any(p.device.type != "cuda" for p in model.parameters()):
                        raise RuntimeError("CPU/disk offload detected; results would not be comparable")
                    if name != "baseline":
                        expected = load_file(str(paths[name] / "adapter_model.safetensors"), device="cpu")
                        actual = get_peft_model_state_dict(model)
                        if set(expected) != set(actual):
                            raise RuntimeError("Loaded adapter tensor names do not match saved weights")
                        if not any(torch.count_nonzero(v).item() for k, v in expected.items() if "lora_B" in k):
                            raise RuntimeError("All LoRA B weights are zero; adapter is ineffective")
                        for key, value in expected.items():
                            if not torch.equal(actual[key].detach().cpu(), value.to(actual[key].dtype)):
                                raise RuntimeError(f"Adapter tensor did not load exactly: {key}")
                        if any(getattr(m, "disable_adapters", False) for m in model.modules() if hasattr(m, "lora_A")):
                            raise RuntimeError("Adapter layers are disabled")
                        del actual, expected
                    validation_s = time.perf_counter() - validation_start
                    t0 = time.perf_counter()
                    for w in range(args.warmups):
                        local_request(model, requests[w % len(requests)])
                    warmup_s = time.perf_counter() - t0
                    loads.append({"model": name, "round": rnd, "load_s": load_s,
                                  "adapter_validation_s": validation_s, "warmup_s": warmup_s,
                                  "warmup_calls": args.warmups, "cache_dir": args.cache_dir,
                                  "note": "Load may include downloads on first use; not a standardized cold-start measure"})
                    write_csv(out / "model_loads.csv", loads)
                    for j, r in enumerate(requests):
                        torch.cuda.reset_peak_memory_stats()
                        try:
                            row = local_request(model, r)
                        except Exception as e:
                            record({"model": name, "round": rnd, "phase": "measured",
                                    "cluster_id": r["cluster_id"], "prompt_id": r["prompt_id"],
                                    "status": "error", "error_type": type(e).__name__})
                            raise
                        record({"model": name, "round": rnd, "phase": "measured", "cluster_id": r["cluster_id"],
                                "prompt_id": r["prompt_id"], **row,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30})
                        if (j + 1) % 10 == 0:
                            print(f"  {j+1}/{len(requests)} labels complete", flush=True)
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            if client:
                print(f"Round {rnd}: teacher API, {len(requests)} calls", flush=True)
                for j, r in enumerate(requests):
                    teacher_call(r, rnd, "measured")
                    if (j + 1) % 10 == 0:
                        print(f"  Teacher {j+1}/{len(requests)} complete", flush=True)
            manifest["elapsed_s"] = time.perf_counter() - started
            save_reports(out, records, loads, manifest, args.gpu_hourly_usd)
        manifest["status"] = "completed_with_request_errors" if any(r["status"] != "ok" for r in records) else "completed"
    except BaseException as e:
        manifest["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
        manifest["failure_type"] = type(e).__name__
        manifest["failure_message"] = redact_error(str(e))
        (out / "error.txt").write_text(redact_error(traceback.format_exc()), encoding="utf-8")
        raise
    finally:
        if client is not None:
            client.close()
        manifest["elapsed_s"] = time.perf_counter() - started
        summary = save_reports(out, records, loads, manifest, args.gpu_hourly_usd)
        print(f"\nStatus: {manifest['status']}; saved results: {out}", flush=True)
        for r in summary:
            print(f"{r['model']:18s} mean={format_num(r['mean_s'])}s p95={format_num(r['p95_s'])}s cost/1000=${format_num(r['estimated_serving_usd_per_1000_labels'])}", flush=True)


if __name__ == "__main__":
    main()
