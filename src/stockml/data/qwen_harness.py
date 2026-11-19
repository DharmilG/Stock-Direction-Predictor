from __future__ import annotations

"""Qwen3 reasoning-rater harness (Track Q, Phase 2).

Frozen-base Ollama client with the analyst playbook (see
``qwen_playbook_v1.md``, the versioned prompt source — this module only
wires it). Pinned operating point from Phase 1 (2026-10-01):

* model ``qwen3:8b`` (Q4_K_M, hash ``500a1f067a9f``), Ollama server
* ``num_ctx=2048`` (4096 OOMs/crashes on RTX 3050 6GB + Ollama 0.34.2)
* ``num_predict>=1024`` mandatory with ``think=True`` (256 truncates
  content to empty)
* reasoning is read from ``message.thinking`` (Ollama chat API), NOT from
  ``<think>`` tags inside ``content``
* ``temperature=0.6``, ``top_p=0.95`` (Qwen3 thinking-mode spec)

Policy: strict JSON schema + validator, 3 retries then ``HARNESS_FAIL``
(logged, never silently accepted), 120s per-call timeout,
concurrency <= 2 (single GPU server — higher values do not help),
TRUNCATED (>1500 tokens) / NO_NEWS flagging, per-call trace logging
to ``track_q/phase_0/traces/``.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

HOLDOUT_START = date(2023, 8, 29)  # §5.0 pinned — never 2023-12-20
EMBARGO_DAYS = 5
STAR_CUTOFF = date(2023, 8, 24)

MODEL = "qwen3:8b"
NUM_CTX = 2048
NUM_PREDICT = 1024  # skeleton prompt (~150 tok) + thinking ~350 tok + JSON — verified 2026-10-02
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_RETRIES = 3
CALL_TIMEOUT_SECONDS = 300  # cold-server worst case ~280s; single GPU, no queue
MAX_TOKENS_ARTICLE = 1500
DEFAULT_THINKING_BUDGET = 512  # Phase 3 winner replaces this; prompt-level cap
PLAYBOOK_VERSION = "v1-skeleton"  # full text + 5 few-shots live versioned in qwen_playbook_v1.md

# Runtime skeleton (measured 2026-10-02): the full md (574 words with
# few-shots) makes qwen3:8b overthink ~6700 chars, starving content even at
# npred 1600. Skeleton keeps think=True traces real at ~1400 chars / ~40s.
# Few-shots return with playbook v2 (kept-STaR pairs) on Phase 8 PASS.
_RUNTIME_SKELETON = (
    "You are an NSE equity analyst rating ONE news item for a 5-day directional signal. "
    "You have NEVER seen prices after decision_date. Stating otherwise invalidates the answer. "
    "Checklist (execute in order, cite step numbers in trace): "
    "1. CLASSIFY the event: earnings_result | earnings_guidance | order_win | mna | "
    "regulatory | rating_change | management | capital_action | debt_credit | "
    "litigation | expansion | macro_policy | short_seller | noise. "
    "If noise, vote FLAT with confidence at most 0.3 and stop. "
    "2. BASE RATE: state the historical precedent for this event type on NSE "
    "(usually-up / usually-down / mixed). Do NOT invent statistics. "
    "3. POSITIONING: using ONLY the PRE-T CONTEXT block, judge whether the move "
    "looks priced in. Overbought plus good news is a fade candidate; oversold plus "
    "bad news is a bounce candidate. "
    "4. COMMIT: vote plus confidence, citing which steps carried weight. "
    "Confidence must be proportionate to evidence: single-source rumour at most 0.4, "
    "confirmed filing plus supporting positioning at most 0.8, never 1.0. "
    'Return STRICT JSON only: {"sentiment": <float -1..1>, "vote": "UP"|"DOWN"|"FLAT", '
    '"confidence": <float 0..1>, "trace_steps": [1,2,3,4]}. '
    "Sentiment sign must agree with vote (positive goes UP, negative goes DOWN, "
    "absolute value below 0.2 goes FLAT)."
)

VALID_VOTES = ("UP", "DOWN", "FLAT")

PLAYBOOK_PATH = Path(__file__).with_name("qwen_playbook_v1.md")


def load_playbook() -> str:
    """Analyst playbook text. Prompt edits are versioned commits to the md."""
    if PLAYBOOK_PATH.exists():
        return PLAYBOOK_PATH.read_text(encoding="utf-8")
    return _FALLBACK_PLAYBOOK


_FALLBACK_PLAYBOOK = """You are an NSE equity analyst rating ONE news item for a 5-day directional signal.
You have NEVER seen prices after decision_date. Stating otherwise invalidates the answer.
Checklist (execute in order, cite step numbers in trace):
1. CLASSIFY the event.
2. BASE RATE: historical precedent (usually-up / usually-down / mixed).
3. POSITIONING: using ONLY the PRE-T CONTEXT block, judge priced-in or not.
4. COMMIT: vote + confidence, citing which steps carried weight.
Return STRICT JSON only."""


def build_prompt(symbol: str, sector: str, pre_t_context: str,
                 article_text: str, playbook: str | None = None,
                 thinking_budget: int = DEFAULT_THINKING_BUDGET) -> tuple[str, bool]:
    """Assemble the per-case prompt. Returns (prompt, truncated_flag)."""
    truncated = False
    text = (article_text or "").strip()
    if len(text.split()) > MAX_TOKENS_ARTICLE:
        text = " ".join(text.split()[:MAX_TOKENS_ARTICLE]) + " [TRUNCATED]"
        truncated = True
    prompt = (
        f"SYMBOL: {symbol}  SECTOR: {sector}\n"
        f"PRE-T CONTEXT:\n  {pre_t_context}\n"
        f"ARTICLE TEXT (full, un-summarized):\n  {text}\n"
        f"INSTRUCTION: {_RUNTIME_SKELETON}\n"
        f"Limit internal reasoning to about {thinking_budget} tokens, then commit."
    )
    return prompt, truncated


@dataclass
class HarnessResult:
    vote: str | None = None
    sentiment: float | None = None
    confidence: float | None = None
    trace_steps: list[int] = field(default_factory=list)
    thinking: str = ""
    content: str = ""
    think_len: int = 0
    truncated: bool = False
    no_news: bool = False
    harness_fail: bool = False
    fail_reason: str = ""
    thermal_abort: bool = False
    thermal_temp_c: float | None = None
    seconds: float = 0.0
    attempts: int = 0


_JSON_RE = re.compile(r"\{[^{}]*\"vote\"[^{}]*\}", re.S)


def validate_output(content: str, thinking: str) -> tuple[dict[str, Any] | None, str]:
    """Strict schema validator. Returns (parsed, reason); reason == '' iff OK."""
    if not thinking or len(thinking.strip()) < 50:
        return None, "empty-thinking"
    match = _JSON_RE.search(content or "")
    if not match:
        return None, "no-json-vote"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "json-parse"
    vote = str(data.get("vote", "")).upper()
    if vote not in VALID_VOTES:
        return None, f"bad-vote:{vote}"
    try:
        sentiment = float(data.get("sentiment"))
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None, "bad-numbers"
    if not (-1.0 <= sentiment <= 1.0 and 0.0 <= confidence <= 1.0):
        return None, "out-of-range"
    steps = data.get("trace_steps", [1, 2, 3, 4])
    if list(steps) != [1, 2, 3, 4]:
        return None, "bad-trace-steps"
    # sign agreement: positive→UP, negative→DOWN, |sent|<0.2→FLAT
    if vote == "UP" and sentiment < 0:
        return None, "sign-mismatch"
    if vote == "DOWN" and sentiment > 0:
        return None, "sign-mismatch"
    if vote == "FLAT" and abs(sentiment) >= 0.5:
        return None, "sign-mismatch"
    return {"vote": vote, "sentiment": sentiment, "confidence": confidence,
            "trace_steps": [1, 2, 3, 4]}, ""


def rate_case(symbol: str, sector: str, pre_t_context: str,
              article_text: str, playbook: str | None = None,
              thinking_budget: int = 512,
              abort_event=None) -> HarnessResult:
    """Rate one article through the frozen base model (no LoRA).

    ``abort_event`` is set by the 5-second thermal monitor at >=87C.  When it
    fires, the in-flight Ollama request is abandoned and never retried: the
    case is left out of the ledger and retried after the cooldown.
    """
    import ollama

    def _aborted() -> bool:
        return abort_event is not None and abort_event.is_set()

    client = ollama.Client(timeout=CALL_TIMEOUT_SECONDS + 60)
    result = HarnessResult(no_news=not (article_text or "").strip())
    prompt, truncated = build_prompt(symbol, sector, pre_t_context,
                                     article_text, playbook)
    result.truncated = truncated
    last_reason = ""
    for attempt in range(1, MAX_RETRIES + 1):
        if _aborted():
            result.harness_fail = True
            result.thermal_abort = True
            result.fail_reason = "thermal-abort"
            return result
        result.attempts = attempt
        start = time.time()
        try:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": TEMPERATURE, "top_p": TOP_P,
                         "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
                think=True,
                keep_alive="5m",
            )
            message = response["message"] if isinstance(response, dict) else response.message
            thinking = (message.get("thinking") or "") if isinstance(message, dict) else (message.thinking or "")
            content = (message.get("content") or "") if isinstance(message, dict) else (message.content or "")
        except Exception as exc:  # noqa: BLE001 — harness must never raise
            result.seconds += time.time() - start
            if _aborted():
                result.harness_fail = True
                result.thermal_abort = True
                result.fail_reason = "thermal-abort"
                return result
            last_reason = f"ollama-error:{type(exc).__name__}"
            continue
        result.seconds += time.time() - start
        if _aborted():
            result.harness_fail = True
            result.thermal_abort = True
            result.fail_reason = "thermal-abort"
            return result
        parsed, reason = validate_output(content, thinking)
        if parsed is not None:
            result.vote = parsed["vote"]
            result.sentiment = parsed["sentiment"]
            result.confidence = parsed["confidence"]
            result.trace_steps = parsed["trace_steps"]
            result.thinking = thinking
            result.content = content
            result.think_len = len(thinking)
            return result
        last_reason = reason
    result.harness_fail = True
    result.fail_reason = last_reason or "unknown"
    return result


def write_trace(trace_dir: Path, case_id: str, prompt: str,
                result: HarnessResult) -> Path:
    """Append-only per-call trace log. Returns the trace path."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{case_id}.json"
    payload = {
        "case_id": case_id,
        "model": MODEL,
        "prompt_version": PLAYBOOK_VERSION,
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
        "truncated": result.truncated,
        "no_news": result.no_news,
        "harness_fail": result.harness_fail,
        "thermal_abort": result.thermal_abort,
        "thermal_temp_c": result.thermal_temp_c,
        "fail_reason": result.fail_reason,
        "seconds": round(result.seconds, 1),
        "attempts": result.attempts,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path
