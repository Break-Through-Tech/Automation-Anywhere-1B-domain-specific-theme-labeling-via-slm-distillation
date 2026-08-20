"""
phase1/finetuning/trainer.py

Hardware-aware LoRA / QLoRA fine-tuning of the student SLM.

Device modes
------------
colab      : CUDA + QLoRA (bitsandbytes 4-bit NF4) + Unsloth if available.
local_mps  : Apple MPS + LoRA in bfloat16 (no quantisation).
local_cpu  : CPU + LoRA in float32 (smoke test only — very slow).

The adapter weights are saved to cfg['paths']['models_out']/lora_adapter/.
The full base model is NOT saved (too large); only the adapter (~50–200 MB).
"""

import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ── Public: load model ────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """
    Load the student SLM and tokenizer with hardware-appropriate settings.

    Returns
    -------
    model, tokenizer
    """
    device_mode = cfg["device_mode"]
    # .strip() guards against accidental whitespace/newlines from YAML editing
    model_id    = cfg["student_slm"]["model_id"].strip()
    hf_cache    = cfg["paths"].get("hf_cache", None)

    # Write the clean value back so the rest of the pipeline sees it too
    cfg["student_slm"]["model_id"] = model_id

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


# ── Public: train ─────────────────────────────────────────────────────────────

def run_finetuning(
    cfg: dict,
    model,
    tokenizer,
    train_path: str,
    val_path: str,
    business_eval=None,
) -> str:
    """
    Fine-tune the model and save the LoRA adapter.

    Parameters
    ----------
    cfg : dict
        Loaded phase1_config.yaml.
    model, tokenizer : from load_model_and_tokenizer()
    train_path, val_path : str
        Paths to train.jsonl and val.jsonl.
    business_eval : BusinessEvaluator | None
        If provided, total training time is recorded.

    Returns
    -------
    str
        Path to the saved LoRA adapter directory.
    """
    from datasets import load_dataset
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import TrainingArguments
    from trl import SFTTrainer

    device_mode = cfg["device_mode"]
    train_cfg   = cfg["training"]
    lora_cfg    = cfg["lora"]
    out_dir     = Path(cfg["paths"]["models_out"]) / "lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model if not provided (pipeline calls with model=None after restructure)
    if model is None:
        logger.info("[trainer] Loading model for fine-tuning ...")
        model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Apply LoRA config ─────────────────────────────────────────────────────
    # If Unsloth is available (Colab), model already has LoRA applied.
    # Otherwise apply via PEFT.
    if not _is_unsloth_model(model):
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds   = load_dataset("json", data_files=val_path,   split="train")
    logger.info(
        f"[trainer] Train: {len(train_ds)} examples | Val: {len(val_ds)} examples"
    )

    # ── Training arguments ────────────────────────────────────────────────────
    # Use transformers.TrainingArguments directly — more stable across trl
    # versions than SFTConfig, which changed its base class in trl >= 0.15.
    #
    # _safe_training_args() handles Python 3.14 / library-version surprises by
    # progressively removing any param that TrainingArguments rejects, so the
    # code never needs to be patched just because a param was renamed upstream.
    use_bf16 = train_cfg["bf16"] and _supports_bf16(device_mode)
    use_fp16 = train_cfg["fp16"] and not use_bf16

    # Compute warmup_steps as a manual fallback in case warmup_ratio is rejected
    # (70 examples / batch_size 4 / grad_accum 4 = ~4 steps per epoch; use 5%
    #  of total steps, but floor at 0 so it's always safe)
    _steps_per_epoch = max(
        1,
        len(train_ds) // (
            train_cfg["per_device_train_batch_size"]
            * train_cfg["gradient_accumulation_steps"]
        ),
    )
    _total_steps  = _steps_per_epoch * train_cfg["num_train_epochs"]
    _warmup_steps = max(0, int(train_cfg["warmup_ratio"] * _total_steps))

    training_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],   # dropped and replaced below if rejected
        warmup_steps=_warmup_steps,               # fallback; overridden by warmup_ratio if accepted
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
    )

    # Force CPU for local_cpu mode
    if device_mode == "local_cpu":
        training_kwargs["use_cpu"] = True

    training_args = _safe_training_args(training_kwargs)

    trainer = _build_sft_trainer(model, tokenizer, train_ds, val_ds, training_args, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("[trainer] Starting training ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    if business_eval is not None:
        business_eval.record_finetuning_time(elapsed)

    logger.info(f"[trainer] Training finished in {elapsed / 60:.1f} min.")

    # ── Save adapter only (not full model) ────────────────────────────────────
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"[trainer] LoRA adapter saved to {out_dir}")

    # Return the trained PEFT model so the pipeline can use
    # model.disable_adapters() / model.enable_adapters() for the
    # baseline vs fine-tuned inference toggle without reloading.
    return trainer.model, tokenizer, str(out_dir)


# ── Public: inference ─────────────────────────────────────────────────────────

def generate_label(
    prompt_str: str,
    model,
    tokenizer,
    cfg: dict,
    device_mode: str,
) -> str:
    """
    Generate a single cluster label from a formatted prompt string.

    Parameters
    ----------
    prompt_str : str
        Output of build_inference_prompt() — the full formatted prompt.
    model, tokenizer : loaded model and tokenizer.
    cfg : dict
    device_mode : str

    Returns
    -------
    str
        Generated label text (stripped, without special tokens).
    """
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
            do_sample=False,         # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (after the prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    label = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return label.strip()


# ── Private: device-specific model loaders ────────────────────────────────────

def _load_colab(model_id: str, cfg: dict):
    """QLoRA with 4-bit NF4 quantisation. Tries Unsloth first, falls back to PEFT."""
    from transformers import AutoTokenizer

    qlora_cfg = cfg["qlora"]
    lora_cfg  = cfg["lora"]

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
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    except ImportError:
        logger.info("[trainer] Unsloth not found — using standard HuggingFace QLoRA.")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qlora_cfg["use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
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
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _load_cpu(model_id: str, cfg: dict):
    """LoRA with float32 on CPU (smoke test)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.warning(
        "[trainer] Running on CPU. Training will be very slow. "
        "Use local_mps or colab for real experiments."
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _build_sft_trainer(model, tokenizer, train_ds, val_ds, training_args, cfg):
    """
    Build SFTTrainer handling both old and new trl / transformers APIs.

    Breaking changes tracked
    ------------------------
    transformers 4.x / trl < 0.15:
        SFTTrainer(model=..., tokenizer=..., max_seq_length=..., dataset_text_field=...)
    transformers 5.x / trl >= 0.15:
        - 'tokenizer' renamed to 'processing_class'
        - 'max_seq_length' and 'dataset_text_field' may have moved elsewhere

    Strategy
    --------
    1. Try with 'processing_class' (modern name) and all SFT-specific params.
    2. On TypeError, extract the rejected param name:
       a. If the rejected param IS the current tokenizer key → switch to the
          other key ('tokenizer' ↔ 'processing_class') and restart.
       b. Otherwise → remove that param and retry (same as _safe_training_args).
    3. If both tokenizer key names are exhausted → raise a clear RuntimeError.
    """
    import re
    from trl import SFTTrainer

    max_seq_len = cfg["student_slm"]["max_seq_length"]

    # Try modern API name first, then legacy
    tokenizer_keys = ["processing_class", "tokenizer"]

    for tok_key in tokenizer_keys:
        kwargs = {
            "model":              model,
            tok_key:              tokenizer,
            "train_dataset":      train_ds,
            "eval_dataset":       val_ds,
            "args":               training_args,
            "max_seq_length":     max_seq_len,
            "dataset_text_field": "text",
        }
        max_iter = len(kwargs) + 1
        succeeded = False

        for _ in range(max_iter):
            try:
                trainer = SFTTrainer(**kwargs)
                logger.info(
                    f"[trainer] SFTTrainer built successfully "
                    f"(tokenizer param='{tok_key}')."
                )
                return trainer
            except TypeError as exc:
                msg   = str(exc)
                match = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not match:
                    # Different TypeError (e.g. wrong type) — stop trying this key
                    logger.debug(f"[trainer] SFTTrainer non-param TypeError: {exc}")
                    break

                bad_param = match.group(1)

                if bad_param == tok_key:
                    # This tokenizer key name is not supported → try the other one
                    logger.info(
                        f"[trainer] SFTTrainer rejected '{tok_key}' as tokenizer param — "
                        f"trying alternative."
                    )
                    break   # break inner loop, outer loop tries next tok_key

                logger.warning(
                    f"[trainer] SFTTrainer rejected param '{bad_param}'. "
                    "Removing and retrying."
                )
                kwargs.pop(bad_param, None)

    raise RuntimeError(
        "[trainer] Could not construct SFTTrainer with any known API combination.\n"
        f"  trl version:              {_get_pkg_version('trl')}\n"
        f"  transformers version:     {_get_pkg_version('transformers')}\n"
        f"  Python version:           {__import__('sys').version.split()[0]}\n"
        "Please check compatibility between your trl, transformers, and Python versions."
    )


def _get_pkg_version(pkg: str) -> str:
    try:
        return __import__(pkg).__version__
    except Exception:
        return "unknown"


# ── Private: safe TrainingArguments builder ───────────────────────────────────

def _safe_training_args(training_kwargs: dict):
    """
    Build a TrainingArguments instance, automatically removing any keyword
    argument that the installed transformers version does not support.

    This makes the training loop resilient to:
      - Python 3.14 / dataclass __init__ generation changes
      - Params renamed between transformers versions (e.g. warmup_ratio →
        warmup_steps, evaluation_strategy → eval_strategy, etc.)
      - New trl restructurings that change which args TrainingArguments accepts

    Each removed param is logged with a WARNING so nothing is silently lost.

    Strategy
    --------
    1. Try building with all kwargs.
    2. On TypeError, parse the unsupported param name from the error message.
    3. Remove that param and retry.
    4. Repeat until success or until a non-TypeError is raised.

    Notes
    -----
    - warmup_ratio and warmup_steps are both included in the initial kwargs;
      if warmup_ratio is rejected, warmup_steps (a concrete integer) remains
      as the fallback and achieves the same effect.
    - The loop is bounded by the number of kwargs, so it cannot infinite-loop.
    """
    import re
    from transformers import TrainingArguments

    kwargs   = dict(training_kwargs)
    max_iter = len(kwargs) + 1   # safety bound

    for _ in range(max_iter):
        try:
            return TrainingArguments(**kwargs)
        except TypeError as exc:
            msg   = str(exc)
            match = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if not match:
                # Different TypeError (e.g. wrong value type) — re-raise
                raise
            bad_param = match.group(1)
            logger.warning(
                f"[trainer] TrainingArguments rejected param '{bad_param}' "
                f"(Python {__import__('sys').version.split()[0]} / "
                f"transformers {__import__('transformers').__version__}). "
                "Removing and retrying."
            )
            kwargs.pop(bad_param, None)

    # Should never reach here
    raise RuntimeError(
        "[trainer] Could not build TrainingArguments after removing all "
        "unsupported kwargs. Check your transformers installation."
    )


# ── Private: utilities ────────────────────────────────────────────────────────

def _configure_tokenizer(tokenizer, model) -> None:
    """Ensure padding token and left-padding for generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # right-pad during training
    # Note: switch to padding_side="left" during generation if batching inference


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
