from __future__ import annotations

"""Thin Phase 7 wrapper around the budget-aware rater for the LoRA adapter.

CODE READY — UNTESTED
"""

from pathlib import Path

from ..data.qwen_rater import rate_case_budgeted, write_trace_budgeted
from ..qpipeline.contracts import load_q_track_config


def adapter_model_name() -> str:
    return str(load_q_track_config().get("models", {}).get("lora_model", "qwen3-8b-lora-r1"))


def rate_with_adapter(
    symbol: str,
    sector: str,
    pre_t_context: str,
    article_text: str,
    thinking_budget: int = 512,
    abort_event=None,
):
    """Rate one article through the Phase 6 adapter model."""
    return rate_case_budgeted(
        symbol=symbol,
        sector=sector,
        pre_t_context=pre_t_context,
        article_text=article_text,
        thinking_budget=thinking_budget,
        model=adapter_model_name(),
        abort_event=abort_event,
    )


def write_adapter_trace(
    trace_dir: Path,
    case_id: str,
    prompt: str,
    result,
    thinking_budget: int,
    phase: str = "phase7",
    extra=None,
):
    return write_trace_budgeted(
        trace_dir,
        case_id,
        prompt,
        result,
        thinking_budget,
        model=adapter_model_name(),
        phase=phase,
        extra=extra,
    )
