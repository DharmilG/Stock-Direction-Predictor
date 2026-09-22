# Track Q — LLM Reasoning-Rater (Rung 1)

**Status:** design locked, execution pending phase-by-phase sign-off.
**Scope:** Qwen3-8B-Q4_K_M as a news-rating stage. LightGBM remains the sole decision model (§9.1 holds). Direction-only (UP/DOWN/FLAT triple-barrier, h=5 primary, h=3 secondary). No summarizer — full article text. One global LoRA adapter with symbol/sector identity tokens. No-news days flagged, never filled. No RF baseline. No stacking.

---

## 1. Why This Track Exists

Chart-only models cannot beat constant-price by more than a margin (Radfar 2025, myth-busting result). The only hope is external information — news. FinBERT is the current rater (fast, CPU, shallow). Track Q adds a reasoning rater: a model that walks an analyst checklist step by step, commits to a direction vote, and saves its reasoning trace so wrong predictions can be autopsied by reading *why*, not just *how much*.

Success is judged downstream (LightGBM UP-AUC delta), never by the rater's own accuracy in isolation.

---

## 2. Locked Decisions

| Decision | Value | Rationale |
|---|---|---|
| Model | Qwen3-8B-Q4_K_M | Biggest reasoner fitting 6GB VRAM (~4.7GB weights + ~1GB KV at 4k ctx). Bigger-Q4 out-reasons smaller-Q5. Native `<think>` reasoning mode. |
| Context window | 4096 | Worst-case call ~2,700 tokens; 4k covers it with spare. 8k would double KV cache toward OOM for zero benefit. |
| Horizon | h=5 primary, h=3 secondary | Measured lock-in (see GLOSSARY_AND_METHOD_NOTES.md §2). h=1 rejected: costs eat 1/3 of move, autocorrelation −0.007. |
| Text policy | Full text, no summarizer | Announcements median 278 chars (~70 tokens), p95 616 chars. Summarization would reason over someone else's judgment for zero token savings. TRUNCATED flag past 1,500 tokens. |
| Adapter scope | One global LoRA, symbol/sector tokens in context | Per-stock adapters would starve (~1,860 articles/symbol, skewed), cost ×50, and memorize. Identity tokens give conditioning without fragmentation. Sector adapters gated behind future heterogeneity evidence. |
| Thinking cap | By micro-ablation (default 512) | 256 vs 512 vs 1024 on 200 cases, agreement-with-label decides. Overflow = low confidence, never silent cut. |
| Null policy | Flag, never fill | Counts = TRUE zeros. Means/shocks = NaN (LightGBM native missing). NO_NEWS token + days_since on silent days. Filling fabricates sentiment. |
| Final decision | LightGBM | LLM output is uncalibrated decoration; LightGBM converts votes into probabilities the thresholds, costs, and ledger are calibrated against. One accountable decision point. |
| Training shape | Single full QLoRA run | Curriculum (easy-first) deferred to rung 2; rung 1 needs one config, one ledger entry. |
| Stacking | Forbidden (§9.1) | LLM is a feature source, not a level-0 model in an ensemble. |

---

## 3. Architecture

```
 NSE announcements / GDELT / RSS (existing fetch, unchanged)
        |
        v
 +------------------+     +---------------------+
 | FinBERT (stays,  |     | Qwen3-8B-Q4_K_M     |
 | CPU, baseline    |     | + global LoRA       |
 | rater)           |     | (nightly rater)     |
 +--------+---------+     +----------+----------+
          |                          |
          v                          v
   finbert_score               qwen_sent (-1..+1)
   finbert_label               qwen_vote_up / qwen_vote_down
                               qwen_conf (0..1)
                               qwen_trace_len
                               finbert_qwen_disagree
                               qwen_available + days_since_qwen
                               reasoning trace -> results/traces/
                               TRUNCATED / NO_NEWS flags
          |                          |
          +------------+-------------+
                       v
          news_context.py (same groupby code,
            new score_col, NaN convention intact)
                       |
                       v
          join.py (unchanged) --> LightGBM (frozen config)
                       |
                       v
          calibrated P(UP/DOWN/FLAT)
                       |
                       v
          tau_flat=0.40 + confidence gate --> signal or abstain
                       |
                       v
          backtest / monitor / ledger (all unchanged)
```

### 3.1 Feature specification (6 core + 2 optional)

| # | Column | Definition |
|---|---|---|
| 1 | `qwen_sent` | Mean LoRA sentiment per decision_date (−1…+1) |
| 2 | `qwen_vote_up` / `qwen_vote_down` | Daily directional vote counts |
| 3 | `qwen_conf` | Mean confidence (conviction weight) |
| 4 | `finbert_qwen_disagree` | abs(FinBERT − Qwen) — uncertainty flag, abstention input |
| 5 | `qwen_trace_len` | Mean `<think>` length — rambling correlates with confusion (testable) |
| 6 | `qwen_available` + `days_since_qwen` | Availability convention, same as all N6 families |
| 7 | *(optional)* `qwen_sector_adj` | Vote minus sector-day mean — idiosyncratic vs herd |
| 8 | *(optional)* `qwen_vote_entropy` | Cross-article disagreement — contested-news detector |

Core 6 ship in rung 1. Columns 7–8 are one-line groupbys, added only if core shows signal (keeps the ablation to one hypothesis: "LLM rating adds edge").

---

## 4. Phase 0 — Environment

* `nvidia-smi` check: 6GB VRAM visible. Disk: 25GB+ free (HF weights ~16GB + GGUF ~5GB + adapters + traces).
* Install Ollama (Windows app) for inference. Training uses unsloth + trl + peft (Ollama cannot train — inference only; QLoRA is the only stack fitting LoRA on 6GB).
* New file `requirements-llm.txt`, pinned separately so the core env never destabilizes.
* Record all versions in `artifacts/state/qwen_setup.json`.
* Ledger entry: `qwen_setup`.

## 5. Phase 1 — Model Acquisition

* **Inference artifact:** `ollama pull qwen3:8b` (Q4_K_M). Health check: one prompt with `enable_thinking=True`, parse `<think>` block + JSON vote. Fail = reinstall, never proceed on a sick model.
* **Training artifact:** HF weights `Qwen/Qwen3-8B` (~16GB disk), loaded 4-bit via unsloth. Full fp16 finetune needs ~60GB+ — never attempted on 6GB.
* Record model hashes + quant + versions in `artifacts/state/qwen_setup.json`.

## 6. Phase 2 — Harness (`src/stockml/data/qwen_harness.py`)

The harness is versioned code. Prompt changes are commits. Contents:

1. **Analyst playbook system prompt** — deterministic 4-step checklist, executed in order:
   1. Classify the event (earnings / leadership / regulatory / corporate action / macro / noise).
   2. State the historical precedent and base rate ("similar NSE events typically resolve X within 5 days") — forces base-rate reasoning, fights narrative fallacy.
   3. Check positioning (pre-T trend: is the move priced in? overbought into good news = fade candidate).
   4. Commit to vote + confidence, citing which steps carried weight.
2. **Strict JSON schema** — `{sentiment, vote, confidence, trace}`. Validator rejects malformed output → automatic retry (max 3) → after 3 fails the case is logged as `HARNESS_FAIL`, never silently accepted.
3. **Leakage self-check instruction** — "you have never seen prices after decision_date; stating otherwise invalidates the answer."
4. **Few-shot slots** — bootstrapped with 5 hand-written examples; replaced by kept STaR pairs once they exist (playbook v1 → v2 as a commit).
5. **Ollama client** — `enable_thinking=True`, temperature 0.6, top-p 0.95 (Qwen thinking-mode spec; greedy decoding degrades reasoning), thinking-budget parameter, `<think>` parser, TRUNCATED / NO_NEWS flagging, per-call trace logging to `results/traces/`.

### 6.1 Forcing compliance (selection, not generation)

The model is free to think anything. Only compliant, correct reasoning survives into weights, via three layers (no trust):
1. Schema-enforced checklist — validator rejects traces missing steps.
2. Temperature 0.6 + thinking-budget cap bound creativity.
3. Outcome + process filters (Phase 5) — non-compliant reasoning never becomes training data.

## 7. Phase 3 — Thinking-Cap Micro-Ablation

* 200 articles × budgets {256, 512, 1024} → agreement-with-label picks the global cap.
* One evening of GPU. Result committed + ledger entry (`thinking_cap`).
* Overflow rule (permanent): unfinished think = low-confidence vote, never silently cut.

## 8. Phase 4 — STaR Dataset Builder (`src/stockml/data/star_builder.py`)

* Input: 92,960 announcements + GDELT/RSS rows, each with a known triple-barrier label (history is the answer key).
* Per case prompt: full article text (TRUNCATED flag past 1,500 tokens) + pre-T context block — 5-day trend, volume shock, sector move, FinBERT prior, `SYMBOL:` / `SECTOR:` identity tokens, `NO_NEWS` token where applicable. B-05 aligned: zero post-T information by construction.
* Generation: harness drives Ollama (frozen base model), saves (prompt, `<think>` trace, vote) per case to `data/llm/star_candidates.jsonl`.
* Resumable: per-case status log; reruns pick up unfinished cases only.
* Throughput: single Ollama server, concurrent requests, incremental saves. Days, not weeks.

## 9. Phase 5 — Filter, Scanner, Audit

* **Outcome filter (automatic, 100%):** keep iff vote == historical label. The program "knows" correctness the way an exam grader does — the answer key was written by history before the test.
* **Leakage scanner (automatic, 100%):** regex + phrase list for post-T dates, outcome mentions, future price levels → dropped even if the vote was correct. Correct for cheating reasons is still wrong.
* **Human audit (~200 kept cases):** rubric — uses the checklist? cites provided context (not invented facts)? confidence proportionate to evidence? Fail-rate >30% → playbook revision, no training.
* Outputs: `data/llm/star_kept.jsonl` (trains) + `data/llm/star_dropped.jsonl` (analysis only, never trained).

STaR is supervised finetuning on self-generated correct rationales — not reinforcement learning. RL (GRPO on direction reward, rung 4) comes only after the scanner is battle-tested, because RL finds reward cheats faster than STaR does.

## 10. Phase 6 — QLoRA Training (unsloth)

* Base: 4-bit Qwen3-8B (frozen). Trainable: LoRA adapters, rank 16, alpha 32, targets q/k/v/o + mlp (~5% of params).
* **Checkpoints every ~15 minutes wall-clock:** timed trial run measures steps-per-15-min → `save_steps` set accordingly, `save_total_limit` rotation, `load_best_model_at_end`. Longer runs survive any interruption.
* **Resume:** `resume_from_checkpoint=True` reads the latest checkpoint in `checkpoints/qwen_lora/` — power loss, OOM, or manual stop restarts exactly where training ended. No repeated work, no silent restarts from zero.
* Validation: 5% held-out kept-cases; track loss + agreement; early-stop patience 3 evals.
* Output: `artifacts/models/qwen_lora_r1/` + committed training log + ledger entry (`qwen_lora_r1`).

## 11. Phase 7 — Nightly Rater + Feature Integration

* Rater script scores new articles through Ollama + LoRA (adapter-loaded), writes sentiment/vote/confidence + traces.
* `news_context.py`: same groupby path, `score_col="qwen_score"`, 6 core columns, NaN convention intact (counts = TRUE zeros, means = NaN, availability flags).
* Join → LightGBM retrain (frozen config) → predictions → existing thresholds/gates.

## 12. Phase 8 — Ablation Gate (pre-registered kill bar)

+qwen columns vs FinBERT-only, identical folds: **Diebold-Mariano p<0.005 AND 2×SE UP-AUC delta.**

* **Pass:** ship rater, lock playbook v2, ledger entry.
* **Fail:** adapters deleted, builder + harness + scanner kept for rung 2, ledger entry written. Failure is a committed result, not a setback.

## 13. Rung Ladder (gated, not scheduled)

| Rung | Upgrade | Unlock condition |
|---|---|---|
| 1 (this doc) | STaR + LoRA, 8B-Q4, local | — (execute now) |
| 2 | Bigger reasoner (Qwen3-32B, cloud) | Rung 1 shows signal worth scaling |
| 3 | Full-parameter finetune | LoRA plateaus below target |
| 4 | Verifier-RL (GRPO on direction reward) | Scanner battle-tested + rungs 1–3 prove reward isn't noise |

Never climb a rung until the one below shows signal. Budget funds honest climbing, not expensive coin flips.

## 14. Files This Track Adds

| File | Phase |
|---|---|
| `requirements-llm.txt` | 0 |
| `artifacts/state/qwen_setup.json` | 0–1 |
| `src/stockml/data/qwen_harness.py` | 2 |
| `src/stockml/data/star_builder.py` | 4 |
| `data/llm/star_candidates.jsonl` | 4 |
| `data/llm/star_kept.jsonl` / `star_dropped.jsonl` | 5 |
| `checkpoints/qwen_lora/` | 6 |
| `artifacts/models/qwen_lora_r1/` | 6 |
| `results/traces/` | 7 |
| `logs/qwen_*.log` | all |
