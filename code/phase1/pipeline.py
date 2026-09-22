"""
phase1/pipeline.py

Phase 1 pipeline — revised flow:

  Steps 1–4 : Data (clustering → preprocessing → label generation → dataset)
  Step  4b  : Copy labeled CSV into run dir (self-contained)
  Step  5   : Fine-tuning  (if enabled)
  Steps 6–7 : Inference    (baseline then fine-tuned, adapter toggled, AFTER fine-tuning)
  Steps 8–9 : Evaluations  (non-LLM metrics + LLM judge for all three models)
  Step  10  : Business eval + latency files
  Step  11  : Combine (pivot + summaries)

Key design points
-----------------
- Evaluations are fully decoupled from training; set run_finetuning: false and
  point evaluation.existing_run_dir to a previous run to re-evaluate without retraining.
- Baseline inference always happens AFTER fine-tuning, using the trained PEFT
  model with adapters toggled off (model.disable_adapters / enable_adapters).
- Per-model evaluation files (nonllm_*.csv, llm_*.csv, latency_*.csv) are written
  independently so any single model's eval can be re-run without affecting others.
- The combine step reads whatever per-model files exist and builds the merged outputs.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def run_phase1(cfg: dict) -> None:
    from phase1.data.schema import (
        CLUSTER_ID, TICKET_RANK, TICKET_DETAILS,
        PROMPT_IDS, cluster_name_col,
        FILE_CLUSTERED_CSV, FILE_GROUPED_CSV, FILE_LABELED_CSV, FILE_LABELED_COPY,
        FILE_TRAIN_JSONL, FILE_VAL_JSONL, FILE_TEST_JSONL,
        FILE_TEACHER_PREDS, FILE_BASELINE_PREDS, FILE_FINETUNED_PREDS,
        FILE_TEACHER_LATENCY, FILE_BUSINESS_EVAL,
        nonllm_file, llm_file, latency_file,
        top_k_details_col,
    )
    from phase1.data.clustering    import run_clustering
    from phase1.data.preprocessing import run_preprocessing
    from phase1.labeling.frontier_llm import run_label_generation
    from phase1.finetuning.dataset import build_dataset, load_split_clusters
    from phase1.finetuning.trainer import (
        load_model_and_tokenizer, run_finetuning, generate_label
    )
    from phase1.evaluation.metrics   import run_evaluation
    from phase1.evaluation.llm_judge import run_llm_judge
    from phase1.evaluation.business_eval import BusinessEvaluator
    from phase1.evaluation.combine   import run_combine
    from phase1.prompts.templates    import build_inference_prompt

    pipe_cfg = cfg["pipeline"]
    paths    = cfg["paths"]
    k        = cfg["top_k"]
    teacher_model_id = cfg["teacher_llm"]["model"]
    eval_cfg = cfg.get("evaluation", {})

    # ─────────────────────────────────────────────────────────────────────────
    # Run-directory setup
    # ─────────────────────────────────────────────────────────────────────────
    existing_run = eval_cfg.get("existing_run_dir")

    if existing_run:
        # Eval-only mode: use an existing run directory, skip all training
        existing_run = Path(existing_run)
        if not existing_run.exists():
            raise FileNotFoundError(
                f"evaluation.existing_run_dir not found: {existing_run}"
            )
        eval_dir    = existing_run / "evaluation"
        adapter_dir = existing_run / "models" / "lora_adapter"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Load labeled CSV from the existing run (self-contained copy)
        labeled_copy = existing_run / FILE_LABELED_COPY
        if labeled_copy.exists():
            labeled_df = pd.read_csv(labeled_copy)
        else:
            labeled_df = pd.read_csv(Path(paths["data_processed"]) / FILE_LABELED_CSV)
        logger.info(
            f"[pipeline] Eval-only mode → existing run: {existing_run}\n"
            f"  Adapter: {adapter_dir}\n"
            f"  Labeled data: {labeled_copy if labeled_copy.exists() else 'data/processed/'}"
        )
    else:
        # Normal mode: generate a timestamped run directory
        slm_short = cfg["student_slm"]["model_id"].strip().split("/")[-1]
        epochs    = cfg["training"]["num_train_epochs"]
        run_id    = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{slm_short}_ep{epochs}"
        run_dir   = Path(paths["outputs"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        paths["labels_out"]     = str(run_dir / "labels")
        paths["models_out"]     = str(run_dir / "models")
        paths["evaluation_out"] = str(run_dir / "evaluation")
        for sub in ("labels", "models", "evaluation"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        eval_dir    = run_dir / "evaluation"
        adapter_dir = run_dir / "models" / "lora_adapter"
        labeled_df  = None   # built during step 3

        logger.info(f"[pipeline] Run ID : {run_id}")
        logger.info(f"[pipeline] Outputs → {run_dir}")

    processed_dir = Path(paths["data_processed"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "_cache").mkdir(exist_ok=True)  # hides intermediate files
    business_eval = BusinessEvaluator(cfg)

    # ═════════════════════════════════════════════════════════════════════════
    # TRAINING STEPS  (skipped if existing_run_dir is set)
    # ═════════════════════════════════════════════════════════════════════════

    if not existing_run:

        # ── STEP 1: Clustering ────────────────────────────────────────────────
        clustered_csv = processed_dir / FILE_CLUSTERED_CSV
        if pipe_cfg["run_clustering"]:
            logger.info("\n" + "━" * 60 + "\n  STEP 1: Clustering\n" + "━" * 60)
            clustered_df = run_clustering(cfg)
        elif clustered_csv.exists():
            logger.info("[pipeline] Skipping clustering — loading checkpoint.")
            clustered_df = pd.read_csv(clustered_csv)
        else:
            raise FileNotFoundError(
                f"run_clustering=false but {clustered_csv} not found."
            )

        # ── STEP 2: Preprocessing ─────────────────────────────────────────────
        grouped_csv = processed_dir / FILE_GROUPED_CSV
        if pipe_cfg["run_preprocessing"]:
            logger.info("\n" + "━" * 60 + "\n  STEP 2: Preprocessing\n" + "━" * 60)
            grouped_df = run_preprocessing(cfg, clustered_df)
        elif grouped_csv.exists():
            logger.info("[pipeline] Skipping preprocessing — loading checkpoint.")
            grouped_df = pd.read_csv(grouped_csv)
        else:
            raise FileNotFoundError(f"run_preprocessing=false but {grouped_csv} not found.")

        # ── STEP 3: Label generation ──────────────────────────────────────────
        labeled_csv = processed_dir / FILE_LABELED_CSV
        if pipe_cfg["run_label_generation"]:
            logger.info("\n" + "━" * 60 + "\n  STEP 3: Label generation\n" + "━" * 60)
            labeled_df = run_label_generation(cfg, clustered_df, grouped_df, business_eval)
            # Persist teacher latency so future eval-only runs can report it
            business_eval.save_teacher_latency_records(
                str(processed_dir / FILE_TEACHER_LATENCY)
            )
        elif labeled_csv.exists():
            logger.info("[pipeline] Skipping label generation — loading checkpoint.")
            labeled_df = pd.read_csv(labeled_csv)
        else:
            raise FileNotFoundError(f"run_label_generation=false but {labeled_csv} not found.")

        # ── STEP 3b: Copy labeled CSV to run dir (self-contained) ─────────────
        run_labeled_copy = run_dir / FILE_LABELED_COPY
        shutil.copy2(str(labeled_csv if labeled_csv.exists() else processed_dir / FILE_LABELED_CSV),
                     str(run_labeled_copy))
        logger.info(f"[pipeline] Labeled data copied → {run_labeled_copy}")

        # ── STEP 4: Build train/val/test JSONL ───────────────────────────────
        train_path = processed_dir / FILE_TRAIN_JSONL
        val_path   = processed_dir / FILE_VAL_JSONL

        # We need a tokenizer for dataset construction; load a temp one
        _, tokenizer_tmp = load_model_and_tokenizer(cfg)
        if pipe_cfg["run_finetuning"] or not train_path.exists():
            logger.info("\n" + "━" * 60 + "\n  STEP 4: Building dataset\n" + "━" * 60)
            split_paths = build_dataset(cfg, labeled_df, tokenizer_tmp)
        else:
            split_paths = {
                "train": str(processed_dir / FILE_TRAIN_JSONL),
                "val":   str(processed_dir / FILE_VAL_JSONL),
                "test":  str(processed_dir / FILE_TEST_JSONL),
            }
            logger.info("[pipeline] Using existing JSONL splits.")
        del tokenizer_tmp   # free memory before loading model properly

        # ── STEP 5: Fine-tuning ───────────────────────────────────────────────
        if pipe_cfg["run_finetuning"]:
            logger.info("\n" + "━" * 60 + "\n  STEP 5: Fine-tuning\n" + "━" * 60)
            model, tokenizer, _ = run_finetuning(
                cfg=cfg,
                model=None,
                tokenizer=None,
                train_path=split_paths["train"],
                val_path=split_paths["val"],
                business_eval=business_eval,
            )
            # Free the fine-tuned model from VRAM immediately.
            # Inference reloads the base model fresh — if the fine-tuned model
            # is still in memory when the fresh load starts, both copies compete
            # for VRAM and OOM on large models (8B+) even with 4-bit quantisation.
            logger.info("[pipeline] Releasing fine-tuned model from VRAM before inference ...")
            del model, tokenizer
            _clear_device_cache()
        else:
            logger.info("[pipeline] Skipping fine-tuning.")

    # ═════════════════════════════════════════════════════════════════════════
    # INFERENCE STEPS  (always after fine-tuning in the same pipeline run)
    # ═════════════════════════════════════════════════════════════════════════

    # Load split cluster IDs from labeled_df
    split_clusters   = load_split_clusters(cfg, labeled_df)
    test_cluster_ids = split_clusters["test"]

    # eval_all_splits: run SLM inference on ALL clusters (train+val+test)
    # This reveals overfitting when you compare train vs test in the summary.
    eval_all_splits = cfg.get("evaluation", {}).get("eval_all_splits", False)
    if eval_all_splits:
        from phase1.data.schema import CLUSTER_ID
        inference_cluster_ids = set(labeled_df[CLUSTER_ID].astype(int).unique().tolist())
        logger.info(
            f"[pipeline] eval_all_splits=true → running SLM inference on "
            f"ALL {len(inference_cluster_ids)} clusters (train+val+test)."
        )
    else:
        inference_cluster_ids = test_cluster_ids
        logger.info(
            f"[pipeline] eval_all_splits=false → SLM inference on "
            f"{len(test_cluster_ids)} test clusters only."
        )

    # Build split_map for combine step ({cluster_id: "train"/"val"/"test"})
    split_map: dict[int, str] = {}
    for split_name, cluster_set in split_clusters.items():
        for cid in cluster_set:
            split_map[int(cid)] = split_name

    run_any_inference = (
        pipe_cfg["run_baseline_eval"] or pipe_cfg["run_finetuned_eval"]
    )

    if run_any_inference:
        logger.info("\n" + "━" * 60 + "\n  STEP 6: Inference\n" + "━" * 60)

        # We load the model fresh for each inference pass rather than toggling
        # adapters. The toggle API (disable_adapters/enable_adapters) changed
        # between PEFT and transformers 5.x and is unreliable across versions.
        # Fresh loads are always correct; from cache on Drive they take ~10-30s.

        baseline_preds_path  = eval_dir / FILE_BASELINE_PREDS
        finetuned_preds_path = eval_dir / FILE_FINETUNED_PREDS

        # ── 6a: Baseline inference (clean base model, no adapter) ─────────────
        if pipe_cfg["run_baseline_eval"]:
            logger.info("[pipeline] Loading fresh base model for baseline inference ...")
            t_load = time.time()
            base_model, base_tok = load_model_and_tokenizer(cfg)
            business_eval.record_model_load_time("baseline", time.time() - t_load)
            _run_inference(
                model=base_model, tokenizer=base_tok, cfg=cfg,
                labeled_df=labeled_df, cluster_ids=inference_cluster_ids,
                fine_tuned=False, output_path=str(baseline_preds_path),
                business_eval=business_eval,
            )
            # Free memory before loading fine-tuned model
            del base_model
            _clear_device_cache()

        # ── 6b: Fine-tuned inference (base model + LoRA adapter) ──────────────
        if pipe_cfg["run_finetuned_eval"]:
            if adapter_dir.exists():
                logger.info(
                    f"[pipeline] Loading base model + LoRA adapter "
                    f"for fine-tuned inference ..."
                )
                from peft import PeftModel
                t_load = time.time()
                ft_base, ft_tok = load_model_and_tokenizer(cfg)
                ft_model        = PeftModel.from_pretrained(ft_base, str(adapter_dir))
                ft_model.eval()
                business_eval.record_model_load_time("finetuned", time.time() - t_load)
                _run_inference(
                    model=ft_model, tokenizer=ft_tok, cfg=cfg,
                    labeled_df=labeled_df, cluster_ids=inference_cluster_ids,
                    fine_tuned=True, output_path=str(finetuned_preds_path),
                    business_eval=business_eval,
                )
                del ft_model
                _clear_device_cache()
            else:
                logger.warning(
                    f"[pipeline] Adapter not found at {adapter_dir}. "
                    "Skipping fine-tuned inference."
                )

        # ── 6c: Teacher predictions (no model needed — from labeled CSV) ───────
        teacher_preds_path = eval_dir / FILE_TEACHER_PREDS
        _make_teacher_predictions(labeled_df, inference_cluster_ids, cfg, str(teacher_preds_path))

    else:
        logger.info("[pipeline] Skipping inference steps.")
        teacher_preds_path   = eval_dir / FILE_TEACHER_PREDS
        baseline_preds_path  = eval_dir / FILE_BASELINE_PREDS
        finetuned_preds_path = eval_dir / FILE_FINETUNED_PREDS

    # ═════════════════════════════════════════════════════════════════════════
    # EVALUATION STEPS  (fully decoupled — read prediction files from disk)
    # ═════════════════════════════════════════════════════════════════════════

    # ── STEP 7: Non-LLM metrics ───────────────────────────────────────────────
    logger.info("\n" + "━" * 60 + "\n  STEP 7: Non-LLM evaluation\n" + "━" * 60)
    for preds_path, fine_tuned, tag, label in [
        (teacher_preds_path,   False, "teacher",   f"Teacher Ceiling ({teacher_model_id})"),
        (baseline_preds_path,  False, "baseline",  "SLM Baseline (pre-distillation)"),
        (finetuned_preds_path, True,  "finetuned", f"SLM Fine-tuned ({cfg['student_slm']['model_id'].strip().split('/')[-1]})"),
    ]:
        if preds_path.exists():
            run_evaluation(
                cfg=cfg, labeled_df=labeled_df,
                predictions_path=str(preds_path),
                fine_tuned=fine_tuned,
                output_path=str(eval_dir / f"{tag}_metrics.csv"),
                eval_label=label, tag=tag,
            )

    # ── STEP 8: LLM judge ────────────────────────────────────────────────────
    if pipe_cfg["run_llm_judge"]:
        logger.info("\n" + "━" * 60 + "\n  STEP 8: LLM-as-judge\n" + "━" * 60)
        for preds_path, fine_tuned, tag, label in [
            (teacher_preds_path,   False, "teacher",   f"Teacher Ceiling ({teacher_model_id})"),
            (baseline_preds_path,  False, "baseline",  "SLM Baseline (pre-distillation)"),
            (finetuned_preds_path, True,  "finetuned", f"SLM Fine-tuned ({cfg['student_slm']['model_id'].strip().split('/')[-1]})"),
        ]:
            if preds_path.exists():
                run_llm_judge(
                    cfg=cfg, labeled_df=labeled_df,
                    predictions_path=str(preds_path),
                    fine_tuned=fine_tuned,
                    output_path=str(eval_dir / f"judge_scores_{tag}.csv"),
                    eval_label=label, tag=tag,
                )
    else:
        logger.info("[pipeline] Skipping LLM judge.")

    # ── STEP 9: Business eval ─────────────────────────────────────────────────
    if pipe_cfg["run_business_eval"]:
        logger.info("\n" + "━" * 60 + "\n  STEP 9: Business evaluation\n" + "━" * 60)

        # Load teacher latency from file if not computed at runtime
        teacher_latency_file = str(processed_dir / FILE_TEACHER_LATENCY)
        business_eval.load_teacher_latency_from_file(teacher_latency_file)

        # Save per-model latency files (read by combine step)
        for tag in ("teacher", "baseline", "finetuned"):
            business_eval.save_latency_file(
                tag, str(eval_dir / latency_file(tag))
            )

        business_eval.save(str(eval_dir / FILE_BUSINESS_EVAL))
        business_eval.log_summary()
    else:
        logger.info("[pipeline] Skipping business eval.")

    # ── STEP 10: Combine ──────────────────────────────────────────────────────
    logger.info("\n" + "━" * 60 + "\n  STEP 10: Combining results\n" + "━" * 60)
    run_combine(eval_dir, split_map=split_map, labeled_df=labeled_df, cfg=cfg)

    logger.info(
        f"\n{'=' * 60}\n"
        f"  Phase 1 pipeline complete.\n"
        f"  Results: {eval_dir}\n"
        f"{'=' * 60}"
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_inference(
    model, tokenizer, cfg, labeled_df, cluster_ids, fine_tuned, output_path, business_eval
) -> None:
    """Run SLM on test clusters for all 5 prompts and write predictions JSONL."""
    from phase1.data.schema import (
        CLUSTER_ID, TICKET_RANK, TICKET_DETAILS, PROMPT_IDS,
        PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_MODEL, PRED_FINE_TUNED, PRED_GENERATED_LABEL,
    )
    from phase1.finetuning.trainer import generate_label
    from phase1.prompts.templates  import build_inference_prompt

    device_mode = cfg["device_mode"]
    slm_id      = cfg["student_slm"]["model_id"].strip()
    domain      = cfg["dataset"]["domain"]
    k           = cfg["top_k"]
    n_warmup    = cfg["evaluation"]["business"]["n_inference_warmup_runs"]

    top_k_df = labeled_df[
        (labeled_df[TICKET_RANK] <= k) & (labeled_df[CLUSTER_ID].isin(cluster_ids))
    ]
    cluster_tickets = {
        int(cid): grp.sort_values(TICKET_RANK)[TICKET_DETAILS].tolist()
        for cid, grp in top_k_df.groupby(CLUSTER_ID)
    }

    tag   = "fine-tuned" if fine_tuned else "baseline"
    total = len(cluster_ids) * len(PROMPT_IDS)
    logger.info(f"[inference:{tag}] {len(cluster_ids)} clusters × {len(PROMPT_IDS)} prompts = {total} predictions ...")

    warmed = False
    with open(output_path, "w", encoding="utf-8") as f_out:
        for cid in sorted(cluster_ids):
            ticket_texts = cluster_tickets.get(int(cid), [])
            if not ticket_texts:
                continue
            for prompt_id in PROMPT_IDS:
                prompt_str = build_inference_prompt(prompt_id, ticket_texts, cfg, tokenizer, domain)
                if not warmed:
                    for _ in range(n_warmup):
                        generate_label(prompt_str, model, tokenizer, cfg, device_mode)
                    warmed = True

                t0    = time.time()
                label = generate_label(prompt_str, model, tokenizer, cfg, device_mode)
                business_eval.record_inference_latency(cid, prompt_id, time.time() - t0, fine_tuned)

                f_out.write(json.dumps({
                    PRED_CLUSTER_ID:      int(cid),
                    PRED_PROMPT_ID:       prompt_id,
                    PRED_MODEL:           slm_id,
                    PRED_FINE_TUNED:      fine_tuned,
                    PRED_GENERATED_LABEL: label,
                }, ensure_ascii=False) + "\n")

    logger.info(f"[inference:{tag}] → {output_path}")


def _clear_device_cache() -> None:
    """Release GPU/MPS memory after deleting a model reference."""
    import gc
    import torch
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _make_teacher_predictions(labeled_df, inference_cluster_ids, cfg, output_path) -> None:
    """Write teacher labels for test clusters in prediction JSONL format."""
    from phase1.data.schema import (
        CLUSTER_ID, PROMPT_IDS, cluster_name_col,
        PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_MODEL, PRED_FINE_TUNED, PRED_GENERATED_LABEL,
    )
    teacher_model = cfg["teacher_llm"]["model"]
    label_cols    = {pid: cluster_name_col(teacher_model, pid) for pid in PROMPT_IDS}

    cluster_rows = (
        labeled_df[labeled_df[CLUSTER_ID].isin(inference_cluster_ids)]
        .groupby(CLUSTER_ID).first().reset_index()
    )
    records = []
    for _, row in cluster_rows.iterrows():
        cid = int(row[CLUSTER_ID])
        for pid in PROMPT_IDS:
            col   = label_cols[pid]
            label = str(row[col]) if col in row.index and pd.notna(row[col]) else ""
            if label:
                records.append({
                    PRED_CLUSTER_ID:      cid,
                    PRED_PROMPT_ID:       pid,
                    PRED_MODEL:           teacher_model,
                    PRED_FINE_TUNED:      False,
                    PRED_GENERATED_LABEL: label,
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"[pipeline] Teacher predictions → {output_path} ({len(records)} records)")
