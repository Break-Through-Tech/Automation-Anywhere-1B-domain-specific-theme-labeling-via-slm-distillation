"""
phase1/finetuning/trainer.py

Hardware-aware LoRA / QLoRA fine-tuning of the student SLM.

Device modes
------------
colab      : CUDA + QLoRA (bitsandbytes 4-bit NF4) + Unsloth if available.
local_mps  : Apple MPS + LoRA in bfloat16 (no quantisation).
local_cpu  : CPU + LoRA in float32 (smoke test only — very slow).

Version compatibility handled internally
----------------------------------------
- SFTConfig: _safe_sft_config() removes any param rejected by the installed
  trl version (e.g. completion_only_loss on older trl).
- SFTTrainer: _build_sft_trainer() tries 'processing_class' then 'tokenizer'
  to handle the trl >= 0.15 rename.
- LoRA target_modules: _resolve_target_modules() converts "all-linear" to an
  explicit layer list, avoiding PEFT versions that iterate the string as chars.

Resume safety
-------------
_find_last_checkpoint() checks output_dir for an existing checkpoint-N/ before
training starts. If found, trainer.train(resume_from_checkpoint=...) picks up
from there instead of restarting — protects against Colab disconnects.
"""

import logging
import os
import re
import time
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


# ── Public: load model ────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """
    Load the student SLM and tokenizer with hardware-appropriate settings.
    Returns (model, tokenizer).
    """
    device_mode = cfg["device_mode"]
    model_id    = cfg["student_slm"]["model_id"].strip()
    cfg["student_slm"]["model_id"] = model_id          # persist the stripped value
    hf_cache    = cfg["paths"].get("hf_cache", None)

    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        logger.info(f"[trainer] HF model cache → {hf_cache}")

    logger.info(f"[trainer] Loading {model_id} in mode={device_mode} ...")

    if device_mode == "colab":
        model, tokenizer = _load_colab(model_id, cfg)
    elif device_mode == "local_mps":
        model, tokenizer = _load_mps(model_id, cfg)
    elif device_mode == "local_cpu":
        model, tokenizer = _load_cpu(model_id, cfg)
    else:
        raise ValueError(f"Unknown device_mode '{device_mode}'.")

    _configure_tokenizer(tokenizer, model)
    logger.info(f"[trainer] Model loaded. Trainable params: {_count_trainable(model):,}")
    return model, tokenizer


# ── Checkpoints ────────────────────────────────────────────────────────────────

def _find_last_checkpoint(out_dir: Path):
    """Return the path to the latest checkpoint-N dir if one exists, else None."""
    if out_dir.exists() and any(out_dir.glob("checkpoint-*")):
        try:
            ckpt = get_last_checkpoint(str(out_dir))
            if ckpt:
                logger.info(f"[trainer] Found existing checkpoint: {ckpt}")
            return ckpt
        except Exception as e:
            logger.warning(f"[trainer] Could not resolve last checkpoint: {e}")
    return None


# ── Public: train ─────────────────────────────────────────────────────────────

def run_finetuning(
    cfg: dict,
    model,
    tokenizer,
    train_path: str,
    val_path: str,
    business_eval=None,
):
    """
    Fine-tune the model and save the LoRA adapter.
    Returns (peft_model, tokenizer, adapter_dir_str).
    """
    from datasets import load_dataset
    from peft import get_peft_model, LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig

    device_mode = cfg["device_mode"]
    train_cfg   = cfg["training"]
    lora_cfg    = cfg["lora"]
    model_id    = cfg["student_slm"]["model_id"].strip()
    out_dir     = Path(cfg["paths"]["models_out"]) / "lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model internally if pipeline passes model=None
    if model is None:
        logger.info("[trainer] Loading model for fine-tuning ...")
        model, tokenizer = load_model_and_tokenizer(cfg)

    # Apply LoRA — skip if Unsloth already applied it during model load
    if not _is_unsloth_model(model):
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=_resolve_target_modules(model_id, lora_cfg["target_modules"]),
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds   = load_dataset("json", data_files=val_path,   split="train")
    logger.info(f"[trainer] Train: {len(train_ds)} | Val: {len(val_ds)} examples")

    # Build SFTConfig via the self-healing builder (handles version diffs)
    use_bf16 = train_cfg["bf16"] and _supports_bf16(device_mode)
    use_fp16 = train_cfg["fp16"] and not use_bf16

    steps_per_epoch = max(
        1,
        len(train_ds) // (
            train_cfg["per_device_train_batch_size"]
            * train_cfg["gradient_accumulation_steps"]
        ),
    )

    # ── Warmup resolution ──────────────────────────────────────────────────────
    # Explicit warmup_steps (when set) always wins over warmup_ratio. This
    # matters for isolation sweeps at small data scale: a warmup_ratio delta
    # (e.g. 0.05 vs 0.10) can round to the SAME integer step count when
    # steps_per_epoch is small, silently collapsing the sweep into a no-op.
    # Setting warmup_steps explicitly guarantees separation between conditions.
    if train_cfg.get("warmup_steps") is not None:
        warmup_steps = int(train_cfg["warmup_steps"])
        logger.info(
            f"[trainer] Using explicit warmup_steps={warmup_steps} "
            f"(overriding warmup_ratio)."
        )
    else:
        warmup_steps = max(
            0,
            int(train_cfg["warmup_ratio"] * steps_per_epoch * train_cfg["num_train_epochs"]),
        )
        logger.info(
            f"[trainer] Computed warmup_steps={warmup_steps} "
            f"from warmup_ratio={train_cfg['warmup_ratio']} "
            f"({steps_per_epoch} steps/epoch × {train_cfg['num_train_epochs']} epochs)."
        )

    sft_config_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        warmup_steps=warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        report_to="none",
        max_seq_length=cfg["student_slm"]["max_seq_length"],
        # Config-driven (defaults True) rather than hardcoded, so a future
        # "with vs. without completion_only_loss" ablation is a one-line
        # config change instead of a code edit.
        completion_only_loss=train_cfg.get("completion_only_loss", True),
    )
    if device_mode == "local_cpu":
        sft_config_kwargs["use_cpu"] = True

    training_args = _safe_sft_config(sft_config_kwargs)
    trainer       = _build_sft_trainer(model, tokenizer, train_ds, val_ds, training_args, cfg)

    logger.info("[trainer] Starting training ...")
    t0 = time.time()
    resume_ckpt = _find_last_checkpoint(out_dir)
    if resume_ckpt:
        logger.info(f"[trainer] Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - t0

    if business_eval is not None:
        business_eval.record_finetuning_time(elapsed)

    logger.info(f"[trainer] Training finished in {elapsed / 60:.1f} min.")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"[trainer] LoRA adapter saved to {out_dir}")

    return trainer.model, tokenizer, str(out_dir)


# ── Public: inference ─────────────────────────────────────────────────────────

def generate_label(
    prompt_str: str,
    model,
    tokenizer,
    cfg: dict,
    device_mode: str,
) -> str:
    """Generate a single cluster label. Uses greedy decoding for reproducibility."""
    device = _get_device(device_mode)
    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["student_slm"]["max_seq_length"],
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len  = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Private: device-specific model loaders ────────────────────────────────────

def _load_colab(model_id: str, cfg: dict):
    """QLoRA with 4-bit NF4 quantisation (or bf16 if load_in_4bit=False). Tries Unsloth first."""
    from transformers import AutoTokenizer

    qlora_cfg      = cfg["qlora"]
    lora_cfg       = cfg["lora"]
    target_modules = _resolve_target_modules(model_id, lora_cfg["target_modules"])

    try:
        from unsloth import FastLanguageModel
        logger.info("[trainer] Unsloth detected — using accelerated loading.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=cfg["student_slm"]["max_seq_length"],
            load_in_4bit=qlora_cfg["load_in_4bit"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    except ImportError:
        logger.info("[trainer] Unsloth not found — using standard HuggingFace QLoRA.")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if qlora_cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=qlora_cfg["use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _load_mps(model_id: str, cfg: dict):
    """LoRA with bfloat16 on Apple MPS (no quantisation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.backends.mps.is_available():
        logger.warning("[trainer] MPS not available — falling back to CPU.")
        return _load_cpu(model_id, cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _load_cpu(model_id: str, cfg: dict):
    """LoRA with float32 on CPU (smoke test only — very slow)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.warning(
        "[trainer] Running on CPU. Training will be very slow. "
        "Use local_mps or colab for real experiments."
    )
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


# ── Private: SFTTrainer builder ───────────────────────────────────────────────

def _build_sft_trainer(model, tokenizer, train_ds, val_ds, training_args, cfg):
    """
    Build SFTTrainer handling both old and new trl / transformers APIs.

    trl >= 0.15 : 'tokenizer' renamed to 'processing_class'.
    dataset_text_field / max_seq_length live on SFTConfig (training_args) now,
    not here — SFTTrainer auto-detects the prompt/completion columns produced
    by dataset.py's _build_examples().

    Strategy: try 'processing_class' first, fall back to 'tokenizer'.
    Other unexpected kwargs are removed one at a time until the call succeeds.
    """
    from trl import SFTTrainer

    tokenizer_keys = ["processing_class", "tokenizer"]

    for tok_key in tokenizer_keys:
        kwargs = {
            "model":         model,
            tok_key:         tokenizer,
            "train_dataset": train_ds,
            "eval_dataset":  val_ds,
            "args":          training_args,
        }
        for _ in range(len(kwargs) + 1):
            try:
                trainer = SFTTrainer(**kwargs)
                logger.info(f"[trainer] SFTTrainer built (tokenizer param='{tok_key}').")
                return trainer
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if not match:
                    break   # non-param TypeError — try next tok_key
                bad = match.group(1)
                if bad == tok_key:
                    logger.info(f"[trainer] SFTTrainer rejected '{tok_key}' — trying alternative.")
                    break   # switch tokenizer key
                logger.warning(f"[trainer] SFTTrainer rejected param '{bad}' — removing.")
                kwargs.pop(bad, None)

    raise RuntimeError(
        "[trainer] Could not construct SFTTrainer.\n"
        f"  trl={_ver('trl')}  transformers={_ver('transformers')}  "
        f"Python={__import__('sys').version.split()[0]}"
    )


# ── Private: safe SFTConfig builder ───────────────────────────────────────────

def _safe_sft_config(sft_kwargs: dict):
    """
    Build SFTConfig, removing any param the installed trl version rejects.
    Warns loudly if 'completion_only_loss' itself is rejected, since that
    means the installed TRL is too old to mask the prompt out of the loss —
    training would silently fall back to full-sequence loss.
    """
    from trl import SFTConfig

    kwargs   = dict(sft_kwargs)
    max_iter = len(kwargs) + 1

    for _ in range(max_iter):
        try:
            return SFTConfig(**kwargs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match:
                raise
            bad = match.group(1)
            if bad == "completion_only_loss":
                logger.warning(
                    "[trainer] Installed TRL version does not support "
                    "'completion_only_loss' — upgrade trl (pip install -U trl) "
                    "or loss will be computed over the full sequence, not just "
                    "the label."
                )
            else:
                logger.warning(f"[trainer] SFTConfig rejected '{bad}' — removing.")
            kwargs.pop(bad, None)

    raise RuntimeError("[trainer] Could not build SFTConfig.")


# ── Private: target-modules resolver ─────────────────────────────────────────

def _resolve_target_modules(model_id: str, target_modules_cfg) -> list:
    """
    Convert 'all-linear' to an explicit layer list before passing to PEFT.

    Some PEFT versions treat 'all-linear' as a string iterable → {'a','l','l',...}.
    Using an explicit list is safe across all versions.

    LLaMA / SmolLM2 / Mistral / Qwen / Gemma: q/k/v/o_proj + gate/up/down_proj
    Phi-3 / Phi-3.5: qkv_proj + o_proj + gate_up_proj + down_proj
    """
    if target_modules_cfg != "all-linear":
        return target_modules_cfg   # already explicit — use as-is

    mid = model_id.lower()
    if any(x in mid for x in ("phi-3.5", "phi-3", "phi3")):
        modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

    logger.info(
        f"[trainer] target_modules 'all-linear' → {modules} "
        f"(model: {model_id.split('/')[-1]})"
    )
    return modules


# ── Private: utilities ────────────────────────────────────────────────────────

def _configure_tokenizer(tokenizer, model) -> None:
    """Ensure padding token and right-padding for training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token       = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"


def _count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _is_unsloth_model(model) -> bool:
    return "unsloth" in type(model).__module__.lower()


def _supports_bf16(device_mode: str) -> bool:
    if device_mode == "colab":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if device_mode == "local_mps":
        return True   # MPS supports bfloat16 in PyTorch >= 2.2
    return False


def _get_device(device_mode: str) -> str:
    if device_mode == "colab":
        return "cuda"
    if device_mode == "local_mps":
        return "mps"
    return "cpu"


def _ver(pkg: str) -> str:
    try:
        return __import__(pkg).__version__
    except Exception:
        return "?"
