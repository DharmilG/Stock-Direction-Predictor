from __future__ import annotations

"""Budget-aware Qwen rating wrapper (Track Q phases 3/4/7).

This module is intentionally separate from ``qwen_harness.py`` so the live Q0
run is untouched. It reuses the same prompt skeleton, validator and trace
format, but forwards ``thinking_budget`` into ``build_prompt`` and lets the
caller choose the model (base or LoRA adapter).

CODE READY — UNTESTED
"""

import json
import re
import time
from pathlib import Path
from typing import Any

from ..data.qwen_harness import (
    CALL_TIMEOUT_SECONDS,
    MAX_RETRIES,
    MODEL,
    NUM_CTX,
    NUM_PREDICT,
    TEMPERATURE,
    TOP_P,
    PLAYBOOK_VERSION,
    HarnessResult,
    build_prompt,
    validate_output,
)


def rate_case_budgeted(
    symbol: str,
    sector: str,
    pre_t_context: str,
    article_text: str,
    thinking_budget: int = 512,
    model: str = MODEL,
    num_ctx: int = NUM_CTX,
    num_predict: int = NUM_PREDICT,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    abort_event=None,
    max_retries: int = MAX_RETRIES,
) -> tuple[HarnessResult, str]:
    """Rate one article with an explicit thinking cap.

    Returns ``(HarnessResult, prompt)`` so the caller can write its own trace.
    """
    import ollama

    def _aborted() -> bool:
        return abort_event is not None and abort_event.is_set()

    client = ollama.Client(timeout=CALL_TIMEOUT_SECONDS + 60)
    prompt, truncated = build_prompt(
        symbol=symbol,
        sector=sector,
        pre_t_context=pre_t_context,
        article_text=article_text,
        thinking_budget=thinking_budget,
    )
    result = HarnessResult(no_news=not (article_text or "").strip())
    result.truncated = truncated
    last_reason = ""
    for attempt in range(1, max_retries + 1):
        if _aborted():
            result.harness_fail = True
            result.thermal_abort = True
            result.fail_reason = "thermal-abort"
            return result, prompt
        result.attempts = attempt
        start = time.time()
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
                think=True,
                keep_alive="5m",
            )
            message = response["message"] if isinstance(response, dict) else response.message
            thinking = (message.get("thinking") or "") if isinstance(message, dict) else (message.thinking or "")
            content = (message.get("content") or "") if isinstance(message, dict) else (message.content or "")
        except Exception as exc:  # noqa: BLE001 — rating must never raise
            result.seconds += time.time() - start
            if _aborted():
                result.harness_fail = True
                result.thermal_abort = True
                result.fail_reason = "thermal-abort"
                return result, prompt
            last_reason = f"ollama-error:{type(exc).__name__}"
            continue
        result.seconds += time.time() - start
        if _aborted():
            result.harness_fail = True
            result.thermal_abort = True
            result.fail_reason = "thermal-abort"
            return result, prompt
        parsed, reason = validate_output(content, thinking)
        if parsed is not None:
            result.vote = parsed["vote"]
            result.sentiment = parsed["sentiment"]
            result.confidence = parsed["confidence"]
            result.trace_steps = parsed["trace_steps"]
            result.thinking = thinking
            result.content = content
            result.think_len = len(thinking)
            return result, prompt
        last_reason = reason
    result.harness_fail = True
    result.fail_reason = last_reason or "unknown"
    return result, prompt


def write_trace_budgeted(
    trace_dir: Path,
    case_id: str,
    prompt: str,
    result: HarnessResult,
    thinking_budget: int,
    model: str = MODEL,
    phase: str = "unspecified",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a per-call trace with phase/budget metadata. Pure file write."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_case = re.sub(r"[^A-Za-z0-9_.-]", "_", str(case_id))
    path = trace_dir / f"{safe_case}.json"
    payload: dict[str, Any] = {
        "case_id": case_id,
        "phase": phase,
        "model": model,
        "prompt_version": PLAYBOOK_VERSION,
        "thinking_budget": thinking_budget,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "prompt": prompt,
        "vote": result.vote,
        "sentiment": result.sentiment,
        "confidence": result.confidence,
        "thinking": result.thinking,
        "content": result.content,
        "trace_steps": result.trace_steps,
        "truncated": result.truncated,
        "no_news": result.no_news,
        "harness_fail": result.harness_fail,
        "thermal_abort": result.thermal_abort,
        "thermal_temp_c": result.thermal_temp_c,
        "fail_reason": result.fail_reason,
        "seconds": round(result.seconds, 3),
        "attempts": result.attempts,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
