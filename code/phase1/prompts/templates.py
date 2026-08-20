"""
phase1/prompts/templates.py

Loads P1–P5 prompt templates from config and formats them with ticket text.
All prompt strings come from the YAML config — nothing is hardcoded here.
"""

import json
import logging

logger = logging.getLogger(__name__)

PROMPT_IDS = ["P1", "P2", "P3", "P4", "P5"]


def format_tickets_block(ticket_texts: list[str]) -> str:
    """
    Format a list of ticket texts into the numbered block passed to the LLM.

    Example output:
        Ticket 1: User cannot log in after password reset...
        Ticket 2: Getting error 0x80070032 when resetting credentials...
    """
    return "\n".join(
        f"Ticket {i + 1}: {text.strip()}"
        for i, text in enumerate(ticket_texts)
    )


def build_messages(
    prompt_id: str,
    ticket_texts: list[str],
    cfg: dict,
    domain: str | None = None,
) -> list[dict]:
    """
    Build the messages list (system + user) for a given prompt ID and cluster.

    Parameters
    ----------
    prompt_id : str
        One of "P1", "P2", "P3", "P4", "P5".
    ticket_texts : list[str]
        Top-k ticket texts for this cluster.
    cfg : dict
        Full phase1_config.yaml (prompts section is read from here).
    domain : str | None
        If include_domain_in_prompt is True in cfg, this is inserted into
        the system prompt (e.g. "IT", "HR", "CX").

    Returns
    -------
    list[dict]
        OpenAI / Anthropic messages format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    prompts_cfg   = cfg["prompts"]
    system_prompt = prompts_cfg["system"].strip()

    if prompts_cfg.get("include_domain_in_prompt", False) and domain:
        system_prompt = (
            f"You are analysing {domain} support tickets. " + system_prompt
        )

    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Must be one of {PROMPT_IDS}.")

    user_template = prompts_cfg[prompt_id]
    tickets_block = format_tickets_block(ticket_texts)
    user_content  = user_template.replace("{tickets}", tickets_block)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


def build_inference_prompt(
    prompt_id: str,
    ticket_texts: list[str],
    cfg: dict,
    tokenizer,
    domain: str | None = None,
) -> str:
    """
    Build a tokenizer-formatted prompt string for SLM inference.

    Uses the tokenizer's apply_chat_template so the format matches
    what the model was trained on.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The SLM's tokenizer.
    (others) : see build_messages()

    Returns
    -------
    str
        Ready-to-tokenise prompt string (no assistant turn appended).
    """
    messages = build_messages(prompt_id, ticket_texts, cfg, domain)
    prompt   = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # adds the opening of the assistant turn
    )
    return prompt
