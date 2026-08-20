"""
main.py — Entry point for all pipeline phases.

Usage
-----
# Phase 1 (default device_mode from config):
python main.py --phase 1 --config configs/phase1_config.yaml

# Phase 1 explicitly on Colab:
python main.py --phase 1 --config configs/phase1_config.yaml --device_mode colab

# Phase 1 on local M1 Mac:
python main.py --phase 1 --config configs/phase1_config.yaml --device_mode local_mps

# Skip checkpoints and recompute everything:
python main.py --phase 1 --config configs/phase1_config.yaml --no_checkpoints
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config file and return as a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict) -> dict:
    """
    Resolve {drive_root} placeholders in all path values.
    Also applies the Colab drive override if device_mode == 'colab'.
    """
    # If running on Colab, override drive_root with the Colab path
    if cfg.get("device_mode") == "colab":
        colab_cfg = cfg.get("colab", {})
        if colab_cfg.get("mount_drive", False):
            override = colab_cfg.get("drive_root_override", "")
            if override:
                cfg["paths"]["drive_root"] = override
                logging.getLogger(__name__).info(
                    f"[main] Colab mode: drive_root → {override}"
                )

    drive_root = cfg["paths"]["drive_root"]

    # Resolve {drive_root} in all path values
    resolved = {}
    for key, val in cfg["paths"].items():
        if isinstance(val, str):
            resolved[key] = val.replace("{drive_root}", drive_root)
        else:
            resolved[key] = val
    cfg["paths"] = resolved

    # Resolve {paths.models_out} in training output_dir
    if "training" in cfg and "output_dir" in cfg["training"]:
        cfg["training"]["output_dir"] = cfg["training"]["output_dir"].replace(
            "{paths.models_out}", cfg["paths"]["models_out"]
        )

    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to the loaded config."""
    if args.device_mode:
        logging.getLogger(__name__).info(
            f"[main] Overriding device_mode: {cfg['device_mode']} → {args.device_mode}"
        )
        cfg["device_mode"] = args.device_mode

    if args.no_checkpoints:
        cfg["checkpoints"]["use_checkpoints"] = False
        logging.getLogger(__name__).info("[main] Checkpoints disabled via CLI flag.")

    if args.model:
        cfg["student_slm"]["model_id"] = args.model.strip()
        logging.getLogger(__name__).info(f"[main] SLM model override: {args.model.strip()}")

    # Sanitize all model ID strings loaded from YAML —
    # strips accidental whitespace / newlines from copy-paste or editor artefacts.
    _sanitize_model_ids(cfg)

    return cfg


def _sanitize_model_ids(cfg: dict) -> None:
    """Strip whitespace from all model ID fields to prevent URL encoding errors."""
    for section in ("student_slm", "teacher_llm"):
        if section in cfg and "model_id" in cfg[section]:
            cfg[section]["model_id"] = cfg[section]["model_id"].strip()
        if section in cfg and "model" in cfg[section]:
            cfg[section]["model"] = cfg[section]["model"].strip()
    for section in ("evaluation",):
        if section in cfg:
            for key in ("embedding_model", "bertscore_model"):
                if key in cfg[section]:
                    cfg[section][key] = cfg[section][key].strip()
            if "judge_llm" in cfg[section] and "model" in cfg[section]["judge_llm"]:
                cfg[section]["judge_llm"]["model"] = cfg[section]["judge_llm"]["model"].strip()


def setup_colab_env(cfg: dict) -> None:
    """
    If running on Google Colab:
    - Mount Google Drive (if configured)
    - Set HF_HOME to Drive cache
    - Load API keys from Colab Secrets
    """
    try:
        import google.colab  # only available in Colab
        _in_colab = True
    except ImportError:
        _in_colab = False

    if not _in_colab:
        # Load API keys from .env file (local development)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logging.getLogger(__name__).info("[main] Loaded .env file.")
        except ImportError:
            pass
        return

    # ── Running in Colab ──────────────────────────────────────────────────────
    colab_cfg = cfg.get("colab", {})

    if colab_cfg.get("mount_drive", True):
        from google.colab import drive
        drive.mount("/content/drive")
        logging.getLogger(__name__).info("[main] Google Drive mounted.")

    # Set HF cache
    hf_cache = cfg["paths"].get("hf_cache", "")
    if hf_cache:
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache

    # Load secrets
    try:
        from google.colab import userdata
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
            try:
                val = userdata.get(key)
                if val:
                    os.environ[key] = val
            except Exception:
                pass
        logging.getLogger(__name__).info("[main] Colab Secrets loaded.")
    except Exception:
        pass


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLM Distillation Pipeline — Break Through Tech AI Studio 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Pipeline phase to run (1 or 2).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--device_mode",
        type=str,
        choices=["colab", "local_mps", "local_cpu"],
        default=None,
        help="Override the device_mode in the config. "
             "colab=CUDA+QLoRA, local_mps=Apple MPS+LoRA, local_cpu=CPU+LoRA.",
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpointing — recompute everything from scratch.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override student SLM model ID (e.g. 'HuggingFaceTB/SmolLM2-1.7B-Instruct').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "demo"],
        default="train",
        help="'train' runs the full training pipeline. "
             "'demo' runs the inference demo on a provided cluster.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="[demo mode] Path to a text file with one support ticket per line.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="[demo mode] Path to the saved LoRA adapter directory "
             "(e.g. outputs/20260819_2347_SmolLM2-1.7B-Instruct_ep6/models/lora_adapter).",
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default="P1",
        choices=["P1", "P2", "P3", "P4", "P5"],
        help="[demo mode] Which prompt template to use for label generation (default: P1).",
    )
    parser.add_argument(
        "--demo_k",
        type=int,
        default=None,
        help="[demo mode] Number of tickets to use (overrides top_k in config).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Load and resolve config ───────────────────────────────────────────────
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg = resolve_paths(cfg)

    # ── Colab environment setup (Drive, secrets, HF cache) ───────────────────
    setup_colab_env(cfg)

    # ── Create output directories ─────────────────────────────────────────────
    for key in ("data_raw", "data_processed", "checkpoints",
                "labels_out", "models_out", "evaluation_out"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # ── Log run summary ───────────────────────────────────────────────────────
    logger.info(
        f"\n{'=' * 60}\n"
        f"  SLM Distillation — Phase {args.phase} | Mode: {args.mode}\n"
        f"  Config:      {args.config}\n"
        f"  Device mode: {cfg['device_mode']}\n"
        f"  Student SLM: {cfg['student_slm']['model_id']}\n"
        f"  Teacher LLM: {cfg['teacher_llm']['model']}\n"
        f"  Drive root:  {cfg['paths']['drive_root']}\n"
        f"{'=' * 60}"
    )

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if args.mode == "demo":
        from phase1.inference_pipeline import run_live_demo

        inf_cfg     = cfg.get("inference", {})
        adapter_dir = args.adapter_dir or inf_cfg.get("adapter_dir")
        prompt_id   = args.prompt_id   or inf_cfg.get("prompt_id", "P1")

        if not adapter_dir:
            logger.error("--adapter_dir (or inference.adapter_dir in config) is required in demo mode.")
            sys.exit(1)
        if not Path(str(adapter_dir)).exists():
            logger.error(f"Adapter directory not found: {adapter_dir}")
            sys.exit(1)

        # demo_k from CLI or config overrides top_k for the demo
        demo_k = args.demo_k or cfg.get("inference", {}).get("demo_k")
        if demo_k:
            cfg["top_k"] = int(demo_k)

        run_live_demo(cfg=cfg, adapter_dir=adapter_dir, prompt_id=prompt_id)
        return

    # ── Train mode ─────────────────────────────────────────────────────────────
    if args.phase == 1:
        from phase1.pipeline import run_phase1
        run_phase1(cfg)

    elif args.phase == 2:
        try:
            from phase2.pipeline import run_phase2
            run_phase2(cfg)
        except ImportError:
            logger.error(
                "Phase 2 pipeline not yet implemented. "
                "This will be added before Phase 2 kickoff."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
