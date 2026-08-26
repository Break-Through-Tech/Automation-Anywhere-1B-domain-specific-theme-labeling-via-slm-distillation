"""
phase1/inference_pipeline.py

Live inference demo: given a cluster of support ticket texts, generates and
compares labels from three models side-by-side.

Live mode (recommended for presentations)
------------------------------------------
python main.py --phase 1 --config configs/phase1_config.yaml --mode demo

Models are loaded ONCE at startup.  Then the program waits for you to type a
file path (one ticket per line).  It runs all three models and prints the
comparison table.  Type 'exit' to stop.

Single-shot mode (Jupyter / scripting)
---------------------------------------
from phase1.inference_pipeline import run_demo
results = run_demo(cfg, ticket_texts=[...], adapter_dir="...", prompt_id="P1")
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 68


# ── Live interactive demo ─────────────────────────────────────────────────────

def run_live_demo(cfg: dict, adapter_dir: str, prompt_id: str = "P1") -> None:
    """
    Interactive live demo loop for presentations.

    Models and metric tools are loaded ONCE at startup.  The loop then accepts
    ticket file paths one at a time and prints a comparison table for each.

    Controls
    --------
    - Type a file path and press Enter to run inference.
    - Type 'exit', 'quit', or press Ctrl-C to stop.
    - Press Enter with an empty line to repeat the last file (useful during demos).
    """
    from phase1.finetuning.trainer   import load_model_and_tokenizer, generate_label
    from phase1.labeling.frontier_llm import _get_llm_caller
    from phase1.evaluation.metrics   import _load_embed_model, _load_rouge
    from phase1.evaluation.llm_judge import _judge_anthropic, _judge_openai
    from phase1.prompts.templates     import build_inference_prompt, build_messages
    from peft import PeftModel

    teacher_cfg = cfg["teacher_llm"]
    judge_cfg   = cfg["evaluation"]["judge_llm"]
    domain      = cfg["dataset"]["domain"]
    device_mode = cfg["device_mode"]
    k           = cfg["top_k"]

    print(f"\n{SEPARATOR}")
    print(f"  SLM DISTILLATION — LIVE DEMO")
    print(f"  Default prompt: {prompt_id}  |  Top-k: {k}  |  Device: {device_mode}")
    judge_mode = cfg.get("evaluation", {}).get("judge_mode", "reference")
    print(f"  Judge mode: {judge_mode}")
    print(SEPARATOR)

    # ── Load metric tools ──────────────────────────────────────────────────────
    print("\nLoading metric tools ...", flush=True)
    t0           = time.time()
    embed_model  = _load_embed_model(cfg["evaluation"]["embedding_model"])
    rouge_scorer = _load_rouge()
    print(f"  Metric tools ready ({time.time()-t0:.1f}s)")

    # ── Load teacher (API — no local model) ────────────────────────────────────
    llm_caller = _get_llm_caller(teacher_cfg["provider"], teacher_cfg)

    # ── Load base SLM ─────────────────────────────────────────────────────────
    print(f"Loading base SLM ({cfg['student_slm']['model_id'].split('/')[-1]}) ...", flush=True)
    t0 = time.time()
    base_model, tokenizer = load_model_and_tokenizer(cfg)
    base_load_s = time.time() - t0
    print(f"  Base SLM ready  ({base_load_s:.1f}s)")

    # ── Load fine-tuned SLM ───────────────────────────────────────────────────
    print(f"Loading fine-tuned SLM from {adapter_dir} ...", flush=True)
    t0 = time.time()
    ft_base, _ = load_model_and_tokenizer(cfg)
    ft_model   = PeftModel.from_pretrained(ft_base, str(adapter_dir))
    ft_model.eval()
    ft_load_s  = time.time() - t0
    print(f"  Fine-tuned SLM ready  ({ft_load_s:.1f}s)")

    print(f"\n{'─'*68}")
    print(f"  Model loading times (one-time, NOT included in per-label latency):")
    print(f"    Base SLM:        {base_load_s:.1f}s")
    print(f"    Fine-tuned SLM:  {ft_load_s:.1f}s")
    print(f"{'─'*68}")
    print("\nReady.  Type a ticket file path (one ticket per line) and press Enter.")
    print("Type 'exit' to stop.\n")

    PROMPT_DESCRIPTIONS = {
        "P1": "concise label (5-15 words)",
        "P2": "primary issue in one phrase",
        "P3": "knowledge base category name",
        "P4": "common theme description",
        "P5": "chatbot routing label",
    }
    judge_mode  = cfg.get("evaluation", {}).get("judge_mode", "reference")
    current_pid = prompt_id   # mutable — user can change per request

    print(f"\n{'─'*68}")
    print(f"  Judge mode: {judge_mode}")
    print(f"  Default prompt: {current_pid} — {PROMPT_DESCRIPTIONS[current_pid]}")
    print(f"\n  Each request:")
    print(f"    1. Enter a ticket file path (or press Enter to repeat last file)")
    print(f"    2. Select a prompt (or press Enter to keep current)")
    print(f"  Type 'exit' to stop.\n")

    session_count = 0
    last_path     = None

    while True:
        # ── Step 1: File path ─────────────────────────────────────────────────
        try:
            raw = input("Ticket file >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted. Exiting demo.")
            break

        if raw.lower() in ("exit", "quit", "q"):
            print("Exiting demo.")
            break

        if not raw:
            if last_path:
                print(f"[Repeating last file: {last_path}]")
                raw = last_path
            else:
                continue

        target = Path(raw)
        if not target.exists():
            print(f"  File not found: {raw}")
            continue

        ticket_lines = [l.strip() for l in target.open(encoding="utf-8") if l.strip()]
        if not ticket_lines:
            print(f"  File is empty: {raw}")
            continue

        last_path    = raw
        ticket_texts = ticket_lines[:k]
        print(f"  {len(ticket_texts)} tickets loaded from {target.name}")

        # ── Step 2: Prompt selection ──────────────────────────────────────────
        print("\n  Prompts:")
        for opt, desc in PROMPT_DESCRIPTIONS.items():
            marker = " ◄ current" if opt == current_pid else ""
            print(f"    {opt}: {desc}{marker}")
        try:
            pid_input = input(f"  Select prompt [{current_pid}]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            pid_input = ""
        if pid_input in PROMPT_DESCRIPTIONS:
            current_pid = pid_input
            print(f"  Using {current_pid}: {PROMPT_DESCRIPTIONS[current_pid]}")
        elif pid_input:
            print(f"  '{pid_input}' not recognised — keeping {current_pid}.")

        session_count += 1
        print(f"\n[Request #{session_count} | Prompt {current_pid}: {PROMPT_DESCRIPTIONS[current_pid]}]")

        # ── Teacher label ─────────────────────────────────────────────────────
        messages      = build_messages(current_pid, ticket_texts, cfg, domain)
        t0            = time.time()
        teacher_label = llm_caller(messages, teacher_cfg).strip()
        teacher_lat   = time.time() - t0

        # ── Baseline label ─────────────────────────────────────────────────────
        prompt_str     = build_inference_prompt(current_pid, ticket_texts, cfg, tokenizer, domain)
        t0            = time.time()
        baseline_label = generate_label(prompt_str, base_model, tokenizer, cfg, device_mode)
        baseline_lat   = time.time() - t0

        # ── Fine-tuned label ──────────────────────────────────────────────────
        t0             = time.time()
        finetuned_label = generate_label(prompt_str, ft_model, tokenizer, cfg, device_mode)
        finetuned_lat   = time.time() - t0

        # ── Scores (vs teacher label) ─────────────────────────────────────────
        from phase1.evaluation.metrics import cosine_similarity, rouge_l

        print("  Scoring ... (calls judge LLM)")
        from phase1.evaluation.llm_judge import score_with_reference, score_reference_free
        score_fn = score_reference_free if judge_mode == "reference_free" else score_with_reference
        base_scores = {"cos": cosine_similarity(baseline_label, teacher_label, embed_model),
                       "rouge": rouge_l(baseline_label, teacher_label, rouge_scorer),
                       "judge": score_fn(ticket_texts, teacher_label, baseline_label, judge_cfg)}
        ft_scores   = {"cos": cosine_similarity(finetuned_label, teacher_label, embed_model),
                       "rouge": rouge_l(finetuned_label, teacher_label, rouge_scorer),
                       "judge": score_fn(ticket_texts, teacher_label, finetuned_label, judge_cfg)}

        # ── Print comparison table ────────────────────────────────────────────
        _print_live_comparison(
            ticket_texts  = ticket_texts,
            prompt_id     = prompt_id,
            teacher_label = teacher_label,  teacher_lat = teacher_lat,
            baseline_label = baseline_label, baseline_lat = baseline_lat,
            finetuned_label = finetuned_label, finetuned_lat = finetuned_lat,
            base_scores   = base_scores,
            ft_scores     = ft_scores,
        )

    print(f"\nSession ended.  {session_count} inference request(s) processed.")


# ── Single-shot mode (Jupyter / scripting) ────────────────────────────────────

def run_demo(cfg: dict, ticket_texts: list[str], adapter_dir: str,
             prompt_id: str = "P1") -> dict:
    """
    Single-cluster inference demo (programmatic / Jupyter use).
    Loads models fresh for each call — use run_live_demo() for presentations.
    """
    from phase1.finetuning.trainer    import load_model_and_tokenizer, generate_label
    from phase1.labeling.frontier_llm import _get_llm_caller
    from phase1.evaluation.metrics    import _load_embed_model, _load_rouge, cosine_similarity, rouge_l
    from phase1.prompts.templates     import build_inference_prompt, build_messages
    from peft import PeftModel

    teacher_cfg = cfg["teacher_llm"]
    judge_cfg   = cfg["evaluation"]["judge_llm"]
    domain      = cfg["dataset"]["domain"]
    device_mode = cfg["device_mode"]
    k           = cfg["top_k"]
    tickets     = ticket_texts[:k]

    llm_caller   = _get_llm_caller(teacher_cfg["provider"], teacher_cfg)
    embed_model  = _load_embed_model(cfg["evaluation"]["embedding_model"])
    rouge_scorer = _load_rouge()

    t0 = time.time()
    teacher_label = llm_caller(build_messages(prompt_id, tickets, cfg, domain), teacher_cfg).strip()
    teacher_lat   = time.time() - t0

    model, tokenizer = load_model_and_tokenizer(cfg)
    prompt_str = build_inference_prompt(prompt_id, tickets, cfg, tokenizer, domain)
    t0 = time.time()
    baseline_label = generate_label(prompt_str, model, tokenizer, cfg, device_mode)
    baseline_lat   = time.time() - t0
    del model

    ft_base, ft_tok = load_model_and_tokenizer(cfg)
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_dir))
    ft_model.eval()
    t0 = time.time()
    finetuned_label = generate_label(prompt_str, ft_model, ft_tok, cfg, device_mode)
    finetuned_lat   = time.time() - t0

    def scores(gen):
        return {
            "cos":   cosine_similarity(gen, teacher_label, embed_model),
            "rouge": rouge_l(gen, teacher_label, rouge_scorer),
            "judge": _run_judge(gen, teacher_label, tickets, judge_cfg),
        }

    base_scores = scores(baseline_label)
    ft_scores   = scores(finetuned_label)

    _print_live_comparison(
        ticket_texts, prompt_id,
        teacher_label, teacher_lat,
        baseline_label, baseline_lat,
        finetuned_label, finetuned_lat,
        base_scores, ft_scores,
    )
    return {
        "teacher":   {"label": teacher_label, "latency_s": teacher_lat},
        "baseline":  {"label": baseline_label, "latency_s": baseline_lat, **base_scores},
        "finetuned": {"label": finetuned_label, "latency_s": finetuned_lat, **ft_scores},
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_judge(generated: str, reference: str, ticket_texts: list[str],
               judge_cfg: dict) -> dict:
    """Call the LLM judge and return parsed scores."""
    from phase1.evaluation.llm_judge import _judge_anthropic, _judge_openai
    fn = _judge_anthropic if judge_cfg["provider"] == "anthropic" else _judge_openai
    try:
        return fn(ticket_texts, reference, generated, judge_cfg)
    except Exception as e:
        logger.warning(f"[demo] Judge call failed: {e}")
        return {"faithfulness": 0, "specificity": 0, "equivalence": 0,
                "composite": 0, "reasoning": "error"}


def _print_live_comparison(
    ticket_texts: list[str], prompt_id: str,
    teacher_label: str, teacher_lat: float,
    baseline_label: str, baseline_lat: float,
    finetuned_label: str, finetuned_lat: float,
    base_scores: dict, ft_scores: dict,
) -> None:
    """Print a clean side-by-side comparison table."""
    try:
        from tabulate import tabulate
        _tab = True
    except ImportError:
        _tab = False

    print(f"\n{SEPARATOR}")
    print(f"  RESULTS — Prompt {prompt_id}")
    print(f"{'─'*68}")
    print(f"  Input tickets ({len(ticket_texts)}):")
    for i, t in enumerate(ticket_texts):
        print(f"    {i+1}. {t[:85]}{'...' if len(t)>85 else ''}")

    # Label + latency table
    label_rows = [
        ["Teacher LLM (ceiling)",      teacher_label,   f"{teacher_lat:.2f}s"],
        ["SLM Baseline (pre-distil)",   baseline_label,  f"{baseline_lat:.2f}s  ← generation only"],
        ["SLM Fine-tuned (post-distil)", finetuned_label, f"{finetuned_lat:.2f}s  ← generation only"],
    ]
    print(f"\n{'─'*68}  GENERATED LABELS")
    if _tab:
        print(tabulate(label_rows, headers=["Model", "Label", "Latency"],
                       tablefmt="rounded_outline", maxcolwidths=[26, 32, 24]))
    else:
        for r in label_rows:
            print(f"  [{r[0]}]\n    {r[1]}  ({r[2]})")

    # Quality scores table
    def _j(s, key, default="—"):
        v = s.get("judge", {}).get(key)
        return str(v) if v is not None else default

    score_rows = [
        ["Teacher LLM", "—", "—", "5 (self-eval)", "5", "5", "5.00"],
        ["SLM Baseline",
         f"{base_scores['cos']:.3f}", f"{base_scores['rouge']:.3f}",
         _j(base_scores,"faithfulness"), _j(base_scores,"specificity"),
         _j(base_scores,"equivalence"),
         f"{base_scores['judge'].get('composite',0):.2f}" if isinstance(base_scores.get('judge'),dict) else "—"],
        ["SLM Fine-tuned",
         f"{ft_scores['cos']:.3f}", f"{ft_scores['rouge']:.3f}",
         _j(ft_scores,"faithfulness"), _j(ft_scores,"specificity"),
         _j(ft_scores,"equivalence"),
         f"{ft_scores['judge'].get('composite',0):.2f}" if isinstance(ft_scores.get('judge'),dict) else "—"],
    ]
    print(f"\n{'─'*68}  QUALITY SCORES (vs Teacher label)")
    headers = ["Model","Cosine","ROUGE-L","Faithful.","Specific.","Equiv.","Comp./5"]
    if _tab:
        print(tabulate(score_rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  " + " | ".join(f"{h:12}" for h in headers))
        for r in score_rows:
            print("  " + " | ".join(f"{str(v):12}" for v in r))

    # Judge reasoning
    print(f"\n{'─'*68}  JUDGE REASONING")
    for label, s in [("Baseline", base_scores), ("Fine-tuned", ft_scores)]:
        r = s.get("judge", {}).get("reasoning", "")
        if r and r != "error":
            print(f"  {label}: {r}")

    print(f"{SEPARATOR}\n")
