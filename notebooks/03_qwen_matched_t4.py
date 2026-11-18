# %% [markdown]
# # Qwen3-8B matched 500-pair evaluation (Colab T4)
# Run all cells on a fresh Colab T4. This does not train a model, regenerate
# teacher labels, modify the original notebooks, or push to GitHub.
#
# The first run asks you to upload `qwen3_evaluation_handoff_2026-11-17.zip`.
# It saves the ZIP and evaluation progress in Google Drive. Re-running resumes
# completed batches and cached judge calls. The judge stage makes paid
# Anthropic API requests for distinct ticket-list/label pairs.
#
# This evaluates the separate n=2000 completion-only Qwen run at checkpoint-40.
# Unsloth CUDA decode graphs are disabled for the entire sweep after a T4
# launch failure when changing batch size; this has its own results folder.
# Its recorded best checkpoint-50 has no weights. The original 4.20/5 Qwen
# result used a different adapter, test set, and rubric; do not attribute that
# score to this checkpoint or compare it directly to scores produced here.

# %%
# 1. Settings — the available n=2000 Qwen checkpoint and frozen handoff.
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-8B"
DRIVE_ROOT = Path("/content/drive/MyDrive/slm-distillation")
SOURCE_RUN = DRIVE_ROOT / "experiments/qwen3_n2000_completion_latest_v1"
ADAPTER_DIR = SOURCE_RUN / "model/checkpoints/checkpoint-40"
CLUSTER_SPLITS = SOURCE_RUN / "cluster_splits.json"
GROUPED_CSV = SOURCE_RUN / "data_processed/bitext_grouped.csv"
SOURCE_SAMPLE_CSV = SOURCE_RUN / "sample_2000.csv"
TRAINING_RECORD = SOURCE_RUN / "training_complete.json"
TRAINING_PLAN = SOURCE_RUN / "plan.json"
HANDOFF_ZIP = DRIVE_ROOT / "matched_qwen_eval/qwen3_evaluation_handoff_2026-11-17.zip"
RESULT_DIR = DRIVE_ROOT / "matched_qwen_eval/qwen3-n2000-checkpoint40-500pairs-eager-v2"
BATCH_SIZES = (16, 8, 4, 2, 1)  # handoff measurement order
MAX_NEW_TOKENS = 60
MAX_PROMPT_TOKENS = 2048  # no silent truncation; actual lengths are recorded
RUN_JUDGE = True  # paid Haiku calls; cached by exact ticket-list/label pair
RUN_BERTSCORE = True
RETRY_OOM = False  # set True only after changing the environment and rerunning
STUDENT_GPU_USD_PER_HOUR = 0.526  # handoff's historical full-utilization assumption

# %%
# 2. Install dependencies, verify T4, mount Drive, and load secrets.
import importlib.util, os, subprocess, sys

# Set both cache variables before importing Hugging Face libraries. Model
# downloads stay on this Colab VM; the adapter and results stay on Drive.
os.environ["HF_HOME"] = "/content/qwen_matched_eval_hf_cache"
os.environ["HF_HUB_CACHE"] = str(Path(os.environ["HF_HOME"]) / "hub")
os.environ["UNSLOTH_DECODE_CUDA_GRAPHS"] = "0"
Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

needed = {
    "unsloth": "unsloth[colab-new]", "peft": "peft", "anthropic": "anthropic",
    "sentence_transformers": "sentence-transformers==5.1.2",
    "rouge_score": "rouge-score==0.1.2", "bert_score": "bert-score==0.3.13",
    "huggingface_hub": "huggingface-hub", "pandas": "pandas", "yaml": "pyyaml",
}
missing = [spec for module, spec in needed.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

from google.colab import drive, userdata
if not (DRIVE_ROOT / "data").exists():
    drive.mount("/content/drive")
assert DRIVE_ROOT.is_dir(), f"Drive project folder not found: {DRIVE_ROOT}"
for secret in ("HF_TOKEN", "ANTHROPIC_API_KEY"):
    value = os.environ.get(secret) or userdata.get(secret)
    assert value, f"Add {secret} to Colab Secrets and enable notebook access."
    os.environ[secret] = value

import torch
assert torch.cuda.is_available(), "Select a GPU runtime in Colab."
gpu_name = torch.cuda.get_device_name(0)
assert "T4" in gpu_name, f"This notebook is validated for a T4; found {gpu_name}."
print("GPU:", gpu_name, "| VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print("Adapter exists:", ADAPTER_DIR.is_dir())

# %%
# 3. Load and validate the frozen handoff; upload it once if absent from Drive.
import hashlib, json, re, unicodedata, zipfile

HANDOFF_ZIP.parent.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
if not HANDOFF_ZIP.is_file():
    from google.colab import files
    print("Upload qwen3_evaluation_handoff_2026-11-17.zip")
    uploaded = files.upload()
    assert set(uploaded) == {HANDOFF_ZIP.name}, f"Upload only {HANDOFF_ZIP.name}."
    HANDOFF_ZIP.write_bytes(uploaded[HANDOFF_ZIP.name])

with zipfile.ZipFile(HANDOFF_ZIP) as bundle:
    expected_hashes = json.loads(bundle.read("SHA256.json"))
    bad = [name for name, digest in expected_hashes.items()
           if hashlib.sha256(bundle.read(name)).hexdigest() != digest]
    assert not bad, f"Handoff ZIP checksum mismatch: {bad}"
    inputs = json.loads(bundle.read("workload/evaluation_inputs.json"))
    provenance = json.loads(bundle.read("workload/selected_ticket_provenance.json"))
    shared_config = json.loads(bundle.read("serving_config.yaml"))
    metric_config = json.loads(bundle.read("metric_settings.json"))
    judge_system = bundle.read("judge_system_prompt.txt").decode().strip()
    teacher = [json.loads(line) for line in bundle.read("comparison_reference/teacher_requests.jsonl").splitlines() if line]

def pair_key(row):
    return (str(row["cluster_id"]), str(row["prompt_id"]))

input_by_key = {pair_key(row): row for row in inputs}
teacher_by_key = {pair_key(row): row for row in teacher}
assert len(inputs) == len(input_by_key) == len(teacher) == len(teacher_by_key) == 500
assert set(input_by_key) == set(teacher_by_key)
assert len({row["cluster_id"] for row in inputs}) == 100
assert all(len(row["tickets"]) == 10 for row in inputs)
assert len(provenance) == 1000
assert {row["prompt_id"] for row in inputs} == {f"P{i}" for i in range(1, 6)}
assert all(str(row["label"]).strip() for row in teacher)
assert shared_config["top_k"] == 10

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())

eval_tickets = {normalize(row["ticket"]) for row in provenance}
assert len(eval_tickets) == 1000
print("Verified handoff: 100 clusters, 1,000 distinct tickets, 500 keyed inputs and teacher labels.")

# %%
# 4. Verify the checkpoint and check this run's training/source ticket overlap.
import pandas as pd

assert (ADAPTER_DIR / "adapter_config.json").is_file(), f"Missing adapter metadata: {ADAPTER_DIR}"

def adapter_weight_files(directory):
    return [p for p in (directory / "adapter_model.safetensors", directory / "adapter_model.bin")
            if p.is_file() and p.stat().st_size > 0]

weights = adapter_weight_files(ADAPTER_DIR)
assert len(weights) == 1, f"Expected one nonempty adapter weight file in {ADAPTER_DIR}; found {weights}"
if weights[0].suffix == ".safetensors":
    from safetensors import safe_open
    with safe_open(str(weights[0]), framework="pt", device="cpu") as adapter_weights:
        assert list(adapter_weights.keys()), f"Empty adapter weights: {weights[0]}"
assert TRAINING_RECORD.is_file(), f"Missing run-specific training record: {TRAINING_RECORD}"
assert TRAINING_PLAN.is_file(), f"Missing run-specific training plan: {TRAINING_PLAN}"
record = json.loads(TRAINING_RECORD.read_text())
training_plan = json.loads(TRAINING_PLAN.read_text())
assert (training_plan["model"], training_plan["n"], training_plan["epochs"],
        training_plan["lr"], training_plan["top_k"],
        training_plan["completion_only"], training_plan["augmentation"]) == (
        MODEL_ID, 2000, 5, 1e-4, 10, True, False
    ), f"Unexpected training plan: {training_plan}"
assert Path(record["best_checkpoint"]).name == "checkpoint-50", record
print("Using checkpoint-40 (available weights; validation loss 0.9741).")
print("Recorded best checkpoint-50 has validation loss 0.9667, but no saved weights.")

adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
adapter_base = str(adapter_config.get("base_model_name_or_path") or "").lower()
assert "qwen3-8b" in adapter_base, f"Adapter is not for Qwen3-8B: {adapter_base!r}"

assert CLUSTER_SPLITS.is_file() and GROUPED_CSV.is_file() and SOURCE_SAMPLE_CSV.is_file(), (
    "Run-specific training source files are missing; cannot check ticket overlap."
)
splits = json.loads(CLUSTER_SPLITS.read_text())
assert {"train", "val", "test"} <= set(splits)
train_ids = {int(value) for value in splits["train"]}
assert train_ids and not (train_ids & {int(value) for value in splits["val"] + splits["test"]})
grouped = pd.read_csv(GROUPED_CSV)
assert {"cluster_id", "top_10_ticket_details"} <= set(grouped.columns)
assert not grouped["cluster_id"].duplicated().any()
assert train_ids <= set(grouped["cluster_id"].astype(int)), "Training cluster IDs missing from grouped data."
training_tickets = set()
for tickets_json in grouped.loc[grouped["cluster_id"].isin(train_ids), "top_10_ticket_details"]:
    tickets = json.loads(tickets_json)
    assert isinstance(tickets, list) and len(tickets) == 10
    training_tickets.update(normalize(ticket) for ticket in tickets)

sample = pd.read_csv(SOURCE_SAMPLE_CSV)
assert "text" in sample.columns and len(sample) == 2000
source_tickets = {normalize(x) for x in sample["text"].dropna()}
assert training_tickets <= source_tickets, (
    "The run's training cluster tickets do not match its saved 2,000-ticket sample; "
    "inspect data provenance before evaluating."
)
train_hits = sorted(eval_tickets & training_tickets)
source_hits = sorted(eval_tickets & source_tickets)
affected_train = sorted({row["cluster_id"] for row in provenance if normalize(row["ticket"]) in train_hits})
affected_source = sorted({row["cluster_id"] for row in provenance if normalize(row["ticket"]) in source_hits})
overlap_report = {
    "run": str(SOURCE_RUN), "adapter_checkpoint": ADAPTER_DIR.name,
    "training_sources": [str(CLUSTER_SPLITS), str(GROUPED_CSV)],
    "train_clusters": len(train_ids),
    "extracted_train_ticket_count": len(training_tickets),
    "sample_source": str(SOURCE_SAMPLE_CSV),
    "source_ticket_count": len(source_tickets),
    "train_ticket_matches": len(train_hits), "train_affected_clusters": affected_train,
    "conservative_source_matches": len(source_hits), "conservative_source_affected_clusters": affected_source,
    "normalization": "NFKC + casefold + whitespace collapse; near-duplicates not detected",
}
(RESULT_DIR / "overlap_report.json").write_text(json.dumps(overlap_report, indent=2))
print(json.dumps(overlap_report, indent=2))
if train_hits:
    print("CAUTION: training overlap found; use the separate training-overlap-free summary.")
else:
    print("No exact normalized overlap found with this Qwen run's training tickets.")
print("Source-sample overlap is reported separately; it does not imply a gradient update.")

# %%
# 5. Freeze run identity, model revision, and a deterministic 500-input order.
import random, time, uuid
from datetime import datetime, timezone
from huggingface_hub import model_info

ordered_inputs = list(inputs)
random.Random(101).shuffle(ordered_inputs)
ordered_keys = [pair_key(row) for row in ordered_inputs]
model_revision = model_info(MODEL_ID, token=os.environ["HF_TOKEN"]).sha

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "model_id": MODEL_ID, "model_revision_at_start": model_revision,
    "source_run": str(SOURCE_RUN), "checkpoint": ADAPTER_DIR.name,
    "adapter_dir": str(ADAPTER_DIR), "adapter_sha256": sha256_file(weights[0]),
    "workload_sha256": expected_hashes["workload/evaluation_inputs.json"],
    "batch_sizes": list(BATCH_SIZES), "shuffle_seed": 101,
    "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
    "reasoning_mode": "Qwen3 enable_thinking=False", "quantization": "4-bit NF4 / FP16 via Unsloth",
    "decode_cuda_graphs": False,
    "judge_system_sha256": hashlib.sha256(judge_system.encode()).hexdigest(),
}
manifest_path = RESULT_DIR / "run_manifest.json"
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, (
        "Existing result directory has different model, adapter, workload, or settings. "
        "Choose a new RESULT_DIR; do not mix runs."
    )
else:
    manifest_path.write_text(json.dumps(manifest, indent=2))
session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
session_info = {
    "session_id": session_id, "gpu_name": gpu_name,
    "torch_version": torch.__version__, "model_revision_at_start": model_revision,
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
session_dir = RESULT_DIR / "sessions"
session_dir.mkdir(exist_ok=True)
(session_dir / f"{session_id}.json").write_text(json.dumps(session_info, indent=2))
print("Results:", RESULT_DIR)
print("Model revision:", model_revision)

# %%
# 6. Load one 4-bit Qwen base plus the saved adapter; prepare exact chat prompts.
import gc
from unsloth import FastLanguageModel
from peft import PeftModel

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_PROMPT_TOKENS + MAX_NEW_TOKENS,
    dtype=torch.float16,
    load_in_4bit=True,
    token=os.environ["HF_TOKEN"],
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False)
model.eval()
FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.eos_token_id is not None
eos_ids = [tokenizer.eos_token_id]
qwen_turn_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
if isinstance(qwen_turn_end, int) and qwen_turn_end != tokenizer.unk_token_id:
    eos_ids = sorted(set(eos_ids + [qwen_turn_end]))

def messages_for(row):
    pid = row["prompt_id"]
    tickets_block = "\n".join(f"Ticket {i}: {ticket.strip()}"
                              for i, ticket in enumerate(row["tickets"], 1))
    return [
        {"role": "system", "content": shared_config["prompts"]["system"].strip()},
        {"role": "user", "content": shared_config["prompts"][pid].replace("{tickets}", tickets_block)},
    ]

for row in ordered_inputs:
    row["messages"] = messages_for(row)
    assert row["messages"] == teacher_by_key[pair_key(row)]["request_messages"], (
        f"Prompt mismatch for {pair_key(row)}; stop before inference."
    )
    row["rendered_prompt"] = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    row["prompt_tokens"] = len(tokenizer.encode(row["rendered_prompt"], add_special_tokens=False))

lengths = [row["prompt_tokens"] for row in ordered_inputs]
assert max(lengths) <= MAX_PROMPT_TOKENS, (
    f"Prompt length {max(lengths)} exceeds MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}. "
    "Do not truncate or omit tickets; raise the limit and use a new RESULT_DIR."
)
print("Prompt token lengths: min", min(lengths), "median", sorted(lengths)[len(lengths)//2], "max", max(lengths))
print("Loaded backend: Unsloth + PEFT; adapter is switched off only for baseline inference.")
session_info["resolved_base_model"] = str(getattr(base_model.config, "_name_or_path", MODEL_ID))
session_info["loaded_config_commit_hash"] = getattr(base_model.config, "_commit_hash", None)
session_info["tokenizer_commit_hash"] = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
session_info["transformers_version"] = __import__("transformers").__version__
session_info["peft_version"] = __import__("peft").__version__
(session_dir / f"{session_id}.json").write_text(json.dumps(session_info, indent=2))

# %%
# 7. Measured, resumable baseline and fine-tuned inference for each batch size.
import contextlib

prediction_dir = RESULT_DIR / "prediction_batches"
prediction_dir.mkdir(exist_ok=True)
oom_path = RESULT_DIR / "oom_status.json"
oom_status = json.loads(oom_path.read_text()) if oom_path.exists() else {}

def save_json_atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def batch_plan(batch_size):
    return [ordered_inputs[i:i+batch_size] for i in range(0, len(ordered_inputs), batch_size)]

def read_completed(path, plan, variant, nominal):
    if not path.exists():
        return []
    completed = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(completed) <= len(plan)
    for index, batch in enumerate(completed):
        expected = [pair_key(row) for row in plan[index]]
        found = [pair_key(row) for row in batch["rows"]]
        assert batch["batch_index"] == index and found == expected
        assert batch["variant"] == variant and batch["nominal_batch_size"] == nominal
        assert batch["actual_batch_size"] == len(expected)
    return completed

def predict(batch, variant, model, tokenizer):
    prompts = [row["rendered_prompt"] for row in batch]
    torch.cuda.synchronize()
    started = time.perf_counter()
    tokenized = tokenizer(prompts, padding=True, truncation=False,
                          add_special_tokens=False, return_tensors="pt")
    assert tokenized["input_ids"].shape[1] <= MAX_PROMPT_TOKENS
    tokenized = {key: value.to("cuda") for key, value in tokenized.items()}
    input_width = tokenized["input_ids"].shape[1]
    adapter_context = model.disable_adapter() if variant == "baseline" else contextlib.nullcontext()
    with torch.inference_mode(), adapter_context:
        generated = model.generate(
            **tokenized, do_sample=False, num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    labels = [tokenizer.decode(seq[input_width:], skip_special_tokens=True).strip()
              for seq in generated]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started  # tokenization, transfer, generation, decoding
    rows = []
    for source, sequence, label in zip(batch, generated, labels):
        new_tokens = sequence[input_width:]
        stopped = any(token_id in eos_ids for token_id in new_tokens.tolist())
        word_count = len(label.split())
        rows.append({
            "cluster_id": source["cluster_id"], "prompt_id": source["prompt_id"],
            "tickets": source["tickets"], "messages": source["messages"],
            "label": label, "prompt_tokens": source["prompt_tokens"],
            "generated_tokens": len(new_tokens), "empty": not bool(label),
            "truncated": len(new_tokens) >= MAX_NEW_TOKENS and not stopped,
            "format_violation": not (5 <= word_count <= 15) or "\n" in label,
        })
    return rows, elapsed

def validate_logits(batch, variant, model, tokenizer):
    tokenized = tokenizer([row["rendered_prompt"] for row in batch],
                          padding=True, truncation=False, add_special_tokens=False,
                          return_tensors="pt")
    tokenized = {key: value.to("cuda") for key, value in tokenized.items()}
    adapter_context = model.disable_adapter() if variant == "baseline" else contextlib.nullcontext()
    with torch.inference_mode(), adapter_context:
        logits = model(**tokenized).logits[:, -1, :]
        assert torch.isfinite(logits).all().item(), "Non-finite next-token logits."
    del logits, tokenized
    torch.cuda.empty_cache()

for variant in ("baseline", "finetuned"):
    for nominal in BATCH_SIZES:
        config_key = f"{variant}_b{nominal}"
        plan = batch_plan(nominal)
        path = prediction_dir / f"{config_key}.jsonl"
        completed = read_completed(path, plan, variant, nominal)
        if len(completed) == len(plan):
            print("COMPLETE", config_key, "500/500")
            continue
        if config_key in oom_status and not RETRY_OOM:
            print("SKIP recorded OOM", config_key, "after", len(completed), "batches")
            continue
        print("RUN", config_key, "resume at batch", len(completed), "of", len(plan))
        try:
            validate_logits(plan[0][:1], variant, model, tokenizer)  # one-input finite-logit check, not timed
            predict(plan[0], variant, model, tokenizer)  # one untimed warmup per configuration
            torch.cuda.empty_cache()
            with path.open("a", encoding="utf-8") as handle:
                for index in range(len(completed), len(plan)):
                    rows, seconds = predict(plan[index], variant, model, tokenizer)
                    record = {
                        "batch_index": index, "variant": variant,
                        "nominal_batch_size": nominal, "actual_batch_size": len(rows),
                        "full_batch_seconds": seconds, "session_id": session_id,
                        "rows": rows,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    if (index + 1) % max(1, 50 // nominal) == 0 or index + 1 == len(plan):
                        print(f"  {config_key}: {sum(len(b) for b in plan[:index+1])}/500")
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                oom_status[config_key] = {
                    "failed_batch": len(read_completed(path, plan, variant, nominal)),
                    "message": str(exc)[:500],
                }
                save_json_atomic(oom_path, oom_status)
                gc.collect()
                torch.cuda.empty_cache()
                print("CUDA OOM:", config_key, "— recorded; continuing to the next configuration.")
                continue
            if "CUDA error" in str(exc) or "CUDA_ERROR_" in str(exc):
                raise RuntimeError(
                    f"CUDA launch failed in {config_key}; completed batches are saved on Drive. "
                    "Restart the Colab runtime and rerun this notebook from the top to resume."
                ) from exc
            raise

del model, base_model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("Inference stage ended; Qwen released from VRAM.")

# %%
# 8. Collect only complete 500-row configurations and keep full batch timing.
student_rows = []
complete_configs = []
all_measured_batches = []
for variant in ("baseline", "finetuned"):
    for nominal in BATCH_SIZES:
        plan = batch_plan(nominal)
        path = prediction_dir / f"{variant}_b{nominal}.jsonl"
        completed = read_completed(path, plan, variant, nominal)
        for batch in completed:
            all_measured_batches.append({
                "variant": variant, "nominal_batch_size": nominal,
                "actual_batch_size": batch["actual_batch_size"],
                "batch_index": batch["batch_index"],
                "session_id": batch.get("session_id", "unknown"),
                "full_batch_seconds": batch["full_batch_seconds"],
                "seconds_per_label": batch["full_batch_seconds"] / batch["actual_batch_size"],
                "labels_per_second": batch["actual_batch_size"] / batch["full_batch_seconds"],
            })
        if len(completed) != len(plan):
            print("INCOMPLETE", variant, nominal, sum(len(batch["rows"]) for batch in completed), "/500")
            continue
        complete_configs.append((variant, nominal))
        for batch in completed:
            for row in batch["rows"]:
                student_rows.append({
                    **row, "variant": variant, "nominal_batch_size": nominal,
                    "actual_batch_size": batch["actual_batch_size"],
                    "batch_index": batch["batch_index"],
                    "session_id": batch.get("session_id", "unknown"),
                    "full_batch_seconds": batch["full_batch_seconds"],
                    "latency_s": batch["full_batch_seconds"] / batch["actual_batch_size"],
                })
assert complete_configs, "No complete configuration yet. Inspect oom_status.json and rerun inference."
assert all(sum(r["variant"] == v and r["nominal_batch_size"] == b for r in student_rows) == 500
           for v, b in complete_configs)
pd.DataFrame(all_measured_batches).to_csv(RESULT_DIR / "batch_timing.csv", index=False)
print("Complete configurations:", complete_configs, "| student rows:", len(student_rows))
print("Saved teacher rows:", len(teacher), "| expected final rows:", len(student_rows) + len(teacher))

# %%
# 9. Blind, reference-free judge; cache identical ticket-list/label pairs.
import anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"
# The current Anthropic API no longer accepts temperature. Keep the rubric
# and pinned Haiku snapshot; record this sampling-setting deviation.
judge_settings = {
    "model": JUDGE_MODEL,
    "max_tokens": 300,
    "temperature": "omitted (unsupported by current Anthropic API)",
    "judge_system_sha256": hashlib.sha256(judge_system.encode()).hexdigest(),
    "anthropic_sdk_version": getattr(anthropic, "__version__", "unknown"),
}
save_json_atomic(RESULT_DIR / "judge_settings.json", judge_settings)

judge_cache_path = RESULT_DIR / "judge_cache.jsonl"
judge_cache = {}
if judge_cache_path.exists():
    for line in judge_cache_path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            assert entry["key"] not in judge_cache or judge_cache[entry["key"]] == entry
            judge_cache[entry["key"]] = entry

def judge_key(tickets, label):
    payload = json.dumps({"tickets": tickets, "label": label}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()

def judge_user_message(tickets, label):
    ticket_block = "\n".join(f"Ticket {i}: {ticket}" for i, ticket in enumerate(tickets, 1))
    return f"Tickets:\n{ticket_block}\n\nCandidate label: {label}"

judge_requests = {}
for row in teacher:
    tickets = input_by_key[pair_key(row)]["tickets"]
    judge_requests[judge_key(tickets, row["label"])] = (tickets, row["label"])
for row in student_rows:
    judge_requests[judge_key(row["tickets"], row["label"])] = (row["tickets"], row["label"])
missing_judge = [key for key in judge_requests if key not in judge_cache]
print("Distinct judge pairs:", len(judge_requests), "| cached:", len(judge_cache), "| API calls remaining:", len(missing_judge))

if RUN_JUDGE:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=90.0)
    with judge_cache_path.open("a", encoding="utf-8") as handle:
        for index, key in enumerate(missing_judge, 1):
            tickets, label = judge_requests[key]
            for attempt in range(5):
                try:
                    response = client.messages.create(
                        model=JUDGE_MODEL, max_tokens=300,
                        system=judge_system,
                        messages=[{"role": "user", "content": judge_user_message(tickets, label)}],
                    )
                    raw = "".join(block.text for block in response.content if block.type == "text").strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
                    scores = json.loads(raw)
                    dims = ("faithfulness", "relevance", "specificity")
                    values = {dim: int(scores[dim]) for dim in dims}
                    assert all(1 <= value <= 5 for value in values.values()), scores
                    entry = {
                        "key": key, **values,
                        "composite": sum(values.values()) / 3,
                        "reason": str(scores.get("reason", "")),
                        "judge_model": response.model,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    judge_cache[key] = entry
                    break
                except (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.APITimeoutError, anthropic.InternalServerError,
                        json.JSONDecodeError, KeyError, ValueError, AssertionError) as exc:
                    if attempt == 4:
                        raise RuntimeError(f"Judge stopped after retries; progress is saved. Last error: {exc}") from exc
                    time.sleep(min(60, 2 ** (attempt + 2)))
            if index % 25 == 0 or index == len(missing_judge):
                print(f"Judged {index}/{len(missing_judge)} new pairs")
else:
    print("RUN_JUDGE=False: inference is saved, but final quality CSV requires judge scores.")

judge_versions = sorted({entry["judge_model"] for entry in judge_cache.values()})
teacher_versions = sorted({row["raw_response"].get("model") for row in teacher})
print("Saved teacher version(s):", teacher_versions, "| judge version(s):", judge_versions)

# %%
# 10. Pinned cosine, ROUGE-L, and BERTScore against each same-key Haiku label.
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from huggingface_hub import snapshot_download
import numpy as np

metric_cache_path = RESULT_DIR / "metric_cache.json"
metric_cache = json.loads(metric_cache_path.read_text()) if metric_cache_path.exists() else {}
metric_pairs = {}
for row in teacher:
    metric_pairs[(row["label"], row["label"])] = None
for row in student_rows:
    reference = teacher_by_key[pair_key(row)]["label"]
    metric_pairs[(row["label"], reference)] = None

def metric_key(label, reference):
    return hashlib.sha256(json.dumps([label, reference], ensure_ascii=False).encode()).hexdigest()

missing_metrics = [(label, ref) for label, ref in metric_pairs
                   if metric_key(label, ref) not in metric_cache]
print("Distinct label/reference pairs:", len(metric_pairs), "| metric pairs remaining:", len(missing_metrics))

if missing_metrics:
    mpnet = SentenceTransformer(
        metric_config["cosine_model"], revision=metric_config["cosine_revision"],
        device="cpu", cache_folder=os.environ["HF_HOME"],
    )
    all_labels = sorted({text for pair in missing_metrics for text in pair})
    vectors = mpnet.encode(all_labels, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=True)
    vector_by_label = dict(zip(all_labels, vectors))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for label, reference in missing_metrics:
        key = metric_key(label, reference)
        metric_cache[key] = {
            "cosine": float(np.dot(vector_by_label[label], vector_by_label[reference])),
            "rouge_l": float(rouge.score(reference, label)["rougeL"].fmeasure),
            "bertscore_f1": None,
        }
    save_json_atomic(metric_cache_path, metric_cache)
    del mpnet, vectors, vector_by_label, rouge
    gc.collect()

if RUN_BERTSCORE:
    bert_todo = [(label, ref) for label, ref in metric_pairs
                 if metric_cache[metric_key(label, ref)]["bertscore_f1"] is None]
    if bert_todo:
        from bert_score import BERTScorer
        bert_path = snapshot_download(
            repo_id=metric_config["bertscore_model"],
            revision=metric_config["bertscore_revision"],
            token=os.environ["HF_TOKEN"],
        )
        scorer = BERTScorer(
            model_type=bert_path, num_layers=17, idf=False,
            rescale_with_baseline=False, use_fast_tokenizer=True,
            device="cuda", batch_size=32,
        )
        for offset in range(0, len(bert_todo), 64):
            chunk = bert_todo[offset:offset+64]
            candidates = [label if label else " " for label, _ in chunk]
            references = [ref for _, ref in chunk]
            _, _, f1 = scorer.score(candidates, references, batch_size=32)
            for (label, reference), score in zip(chunk, f1.tolist()):
                metric_cache[metric_key(label, reference)]["bertscore_f1"] = float(score)
            save_json_atomic(metric_cache_path, metric_cache)
            print("BERTScore:", min(offset + len(chunk), len(bert_todo)), "/", len(bert_todo))
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
print("Metric cache complete for", len(metric_pairs), "distinct label/reference pairs.")

# %%
# 11. Export the 17-column CSV, row provenance, and quality/latency summaries.
import csv

assert all(key in judge_cache for key in judge_requests), (
    "Judge cache is incomplete. Set RUN_JUDGE=True and rerun from Cell 9."
)
assert all(metric_key(label, ref) in metric_cache for label, ref in metric_pairs)
if RUN_BERTSCORE:
    assert all(metric_cache[metric_key(label, ref)]["bertscore_f1"] is not None
               for label, ref in metric_pairs)

columns = [
    "Name of Model", "Model Owner", "Prompt ID", "Actual Prompt", "Input to the Model",
    "Output of the Model", "Latency", "Batch", "Batch Size", "Faithfulness", "Relevance",
    "Specificity", "Composite", "Cosine Score", "ROUGE", "BertScore", "Estimated/Approx Cost",
]
csv_rows = []
row_provenance = []
summary_source = []
training_overlap_clusters = set(affected_train)
source_overlap_clusters = set(affected_source)

def append_result(source, variant, nominal, actual, full_seconds, latency, teacher_row=None):
    key = pair_key(source)
    label = source["label"]
    tickets = input_by_key[key]["tickets"]
    ref = teacher_by_key[key]["label"]
    j = judge_cache[judge_key(tickets, label)]
    m = metric_cache[metric_key(label, ref)]
    if teacher_row is None:
        model_variant = (f"LoRA {ADAPTER_DIR.name} (n2000 run)" if variant == "finetuned"
                         else "base (adapter disabled)")
        model_name = f"{MODEL_ID} | {model_variant} | Unsloth NF4/FP16"
        prompt_messages = source["messages"]
        cost = latency * STUDENT_GPU_USD_PER_HOUR / 3600
        prompt_hash = hashlib.sha256(json.dumps(prompt_messages, ensure_ascii=False).encode()).hexdigest()
    else:
        model_name = f"{teacher_row['raw_response'].get('model', 'claude-haiku-4-5')} | teacher"
        prompt_messages = teacher_row["request_messages"]
        usage = teacher_row["raw_response"]["usage"]
        cost = (usage["input_tokens"] * 1 + usage["output_tokens"] * 5) / 1_000_000
        prompt_hash = hashlib.sha256(json.dumps(prompt_messages, ensure_ascii=False).encode()).hexdigest()
    csv_rows.append({
        "Name of Model": model_name, "Model Owner": "Aashish Khosla",
        "Prompt ID": key[1],
        "Actual Prompt": json.dumps(prompt_messages, ensure_ascii=False),
        "Input to the Model": json.dumps(tickets, ensure_ascii=False),
        "Output of the Model": label, "Latency": latency,
        "Batch": "YES" if actual > 1 else "NO", "Batch Size": actual,
        "Faithfulness": j["faithfulness"], "Relevance": j["relevance"],
        "Specificity": j["specificity"], "Composite": j["composite"],
        "Cosine Score": m["cosine"], "ROUGE": m["rouge_l"],
        "BertScore": m["bertscore_f1"], "Estimated/Approx Cost": cost,
    })
    row_provenance.append({
        "csv_line_number": len(csv_rows) + 1, "cluster_id": key[0], "prompt_id": key[1],
        "variant": variant, "nominal_batch_size": nominal, "actual_batch_size": actual,
        "session_id": source.get("session_id") if teacher_row is None else "teacher_saved",
        "full_batch_response_seconds": full_seconds, "latency_seconds_per_label": latency,
        "prompt_sha256": prompt_hash, "teacher_reference": ref,
        "judge_model": j["judge_model"], "judge_reason": j["reason"],
        "training_overlap_free": key[0] not in training_overlap_clusters,
        "source_sample_overlap_free": key[0] not in source_overlap_clusters,
        "prompt_tokens": source.get("prompt_tokens"),
        "generated_tokens": source.get("generated_tokens"),
        "empty": source.get("empty"), "truncated": source.get("truncated"),
        "format_violation": source.get("format_violation"),
    })
    summary_source.append({
        "variant": variant, "nominal_batch_size": nominal,
        "cluster_id": key[0], "composite": j["composite"],
        "cosine": m["cosine"], "rouge_l": m["rouge_l"],
        "bertscore_f1": m["bertscore_f1"], "latency_s": latency,
        "cost_usd": cost, "training_overlap_free": key[0] not in training_overlap_clusters,
    })

for row in teacher:
    source = {**row, "tickets": input_by_key[pair_key(row)]["tickets"]}
    append_result(source, "teacher", None, 1, row["latency_s"], row["latency_s"], teacher_row=row)
for row in student_rows:
    append_result(row, row["variant"], row["nominal_batch_size"],
                  row["actual_batch_size"], row["full_batch_seconds"], row["latency_s"])

assert len(csv_rows) == 500 + 500 * len(complete_configs)
assert len({(r["cluster_id"], r["prompt_id"], r["variant"], r["nominal_batch_size"])
            for r in row_provenance}) == len(csv_rows)

csv_path = RESULT_DIR / "qwen3_results.csv"
csv_temp = csv_path.with_name(csv_path.name + ".tmp")
with csv_temp.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(csv_rows)
os.replace(csv_temp, csv_path)
save_json_atomic(RESULT_DIR / "raw_records_matrix.json", csv_rows)
save_json_atomic(RESULT_DIR / "row_provenance.json", row_provenance)

summary_df = pd.DataFrame(summary_source)
summary = summary_df.groupby(["variant", "nominal_batch_size"], dropna=False).agg(
    rows=("composite", "size"), judge_composite=("composite", "mean"),
    cosine=("cosine", "mean"), rouge_l=("rouge_l", "mean"),
    bertscore_f1=("bertscore_f1", "mean"),
    latency_s_per_label=("latency_s", "mean"),
    estimated_cost_usd=("cost_usd", "sum"),
).reset_index()
summary.to_csv(RESULT_DIR / "summary_full.csv", index=False)
unseen = summary_df[summary_df["training_overlap_free"]]
unseen_summary = unseen.groupby(["variant", "nominal_batch_size"], dropna=False).agg(
    rows=("composite", "size"), judge_composite=("composite", "mean"),
    cosine=("cosine", "mean"), rouge_l=("rouge_l", "mean"),
    bertscore_f1=("bertscore_f1", "mean"),
).reset_index()
unseen_summary.to_csv(RESULT_DIR / "summary_training_overlap_free.csv", index=False)

print("CSV:", csv_path, "| rows:", len(csv_rows), "| complete configs:", len(complete_configs), "/10")
print("Training-overlap-free clusters:", 100 - len(training_overlap_clusters), "/100")
print("Source-sample-overlap-free clusters:", 100 - len(source_overlap_clusters), "/100")
print(summary.to_string(index=False))
if len(complete_configs) < 10:
    print("Some configurations did not complete. See oom_status.json; do not claim a full 5,500-row sweep.")
else:
    assert len(csv_rows) == 5500
    print("Full 5,500-row Qwen sweep verified.")

# %% [markdown]
# ## Interpretation notes
# - Baseline means pretrained Qwen3-8B with the adapter disabled; it has zero
#   fine-tuning epochs. Fine-tuned means checkpoint-40 from a run planned for
#   five epochs at 1e-4; its trainer state records 2.893 epochs at this step.
# - The 500 saved Haiku labels are references, not newly generated outputs.
# - Teacher cosine/ROUGE/BERTScore against itself are identity checks, not
#   independent teacher-quality measurements; its judge score is reference-free.
# - Latency excludes model loading, validation, warmup, API judging, and disk
#   writes. Student cost is a historical full-utilization estimate.
# - Judge calls are cached, but a resolved Haiku model change must be reported.
# - Summaries use 500 related cluster-prompt pairs per complete configuration;
#   five prompts from one cluster are not independent observations.
# - The training-overlap-free summary excludes every cluster containing a
#   ticket seen in this checkpoint's training prompts. The separate source
#   overlap count also includes tickets from the run's validation/test source.
# - No Git push or training is performed by this notebook.
