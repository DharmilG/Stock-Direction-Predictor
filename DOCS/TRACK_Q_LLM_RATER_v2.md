# Track Q — LLM Reasoning-Rater (Rung 1), Revised

**Status:** design locked, execution phase-gated — see Status Ledger below. Nothing beyond the marked DONE rows has been executed.
**Supersedes:** the original Track Q draft. This version exists specifically to close the gaps found in review — see §1 Change Log.
**Scope:** Qwen3-8B-Q4_K_M as a news-rating stage feeding LightGBM. LightGBM remains the sole decision model. Direction-only (UP/DOWN/FLAT triple-barrier, h=5 primary, h=3 secondary). No summarizer — full article text. One global LoRA adapter with symbol/sector identity tokens. No-news days flagged, never filled. No RF baseline. No stacking.

### Status Ledger (updated 2026-10-20 - DONE vs PENDING/TODO)

| Item | Status | Evidence |
|---|---|---|
| Phase 0 — Environment | **DONE** | `artifacts/state/qwen_setup.json` (`status: complete`, Ollama 0.34.2, WSL torch cu130) |
| Phase 1 — Model Acquisition + throughput probe | **DONE (sampled path)** | `qwen_setup.json: phase1` — health pass (corrected), `ctx 2048` pinned, 14-day gate FAILED for full Pool A → sampled STaR |
| Doc fixes: §5.0 holdout pin, Pool C `n=1500`, cap `n=1000`, §14A–D appendix | **DONE** | This file §§5, 7.4, 8, 14 |
| News-plumbing audit (coverage 0.52% news / 96.6% ann) | **DONE (diagnostic only, no data changed)** | Subagent audit 2026-10-01; fix itself is TODO below |
| Phase 2 - Harness (`qwen_harness.py` + playbook + guards test) | **DONE (2026-10-02/03)** | `qwen_harness.py`, `qwen_playbook_v1.md`, `tests/test_qwen_guards.py` (15/15 green 2026-10-20) |
| Phase 2.5 - Q0-D1 zero-shot gate (Pool C `n=1500`) | **IN PROGRESS** | 122/1500 rated (2026-10-20 01:54); ledger `data/llm/qwen_rating_ledger.parquet`; pilot n=26: Qwen 30.8% vs FinBERT 19.2% (inconclusive) |
| Thermal guard v3 + overnight supervisor (2026-10-20) | **DONE** | `run_q0_overnight.ps1` + `q0_d1.py` ladder (80/84/85/87C, 20-min rest, 4/hour 30-min cooldown, 5s abort, 1400 MHz cap set/reset); smoke test 1 case @72C max |
| Zero-touch scheduled task - Option A (2026-10-20) | **CODE READY - user runs setup once (single UAC)** | `setup_q0_task.ps1` registers `StockML_Q0_Supervisor` (Highest, at-logon, no time limit, auto-restart); `q0_status.ps1` read-only view; supervisor mutex + stale cleanup + Ollama health/restart + crash backoff 60s-30min + heartbeats; actual-work 15-min break timer in `q0_d1.py`; parser-checked, 70/70 tests green; not started by the agent |
| Phase 3 — Thinking-cap (`n=1000` paired) | **CODE READY — UNTESTED (gated on Q0-D1 PASS)** | `src/stockml/evaluation/q3_thinking_cap.py`, `src/stockml/data/qwen_rater.py`, `config/q_track.yaml` |
| Phase 4 — STaR builder (sampled, assertion-enforced) | **CODE READY — UNTESTED (gated on Phase 3 + Phase 6 probe)** | `src/stockml/data/star_builder.py`, `config/q_track.yaml` |
| Phase 5 — Filter/scanner/audit + keep-rate | **CODE READY — UNTESTED (gated on Phase 4)** | `src/stockml/data/leakage_scanner.py`, `src/stockml/data/star_builder.py` |
| Phase 6 — QLoRA training | **CODE READY — UNTESTED (gated on Phase 5)** | `src/stockml/training/qwen_qlora.py`, `src/stockml/training/qwen_export.py`, `scripts/qwen_10step_probe.py` |
| Phase 6 probe — 10-step pre-probe | **CODE READY — UNTESTED (gated on Phase 3)** | `src/stockml/qpipeline/runner.py`, `scripts/qwen_10step_probe.py` |
| Phase 7 — Nightly rater + integration | **CODE READY — UNTESTED (gated on Phase 6)** | `src/stockml/data/qwen_nightly.py`, `src/stockml/data/qwen_lora_harness.py`, `src/stockml/features/qwen_features.py` |
| Phase 8 — Ablation gate (Pool D, once, logged) | **CODE READY — UNTESTED (gated on Phase 7)** | `src/stockml/evaluation/q8_ablation.py`, `src/stockml/qpipeline/*` |
| Orchestrator + wrappers | **CODE READY — UNTESTED** | `src/stockml/qpipeline/*`, `run_q_pipeline.ps1`, `q_pipeline_status.ps1`, `config/q_track.yaml` |
| Rungs 2–4 (32B / full-finetune / RL) | **PENDING / TODO (gated on rung-1 signal)** | — |
| News-plumbing fix A1 (66k ann scoring + NaN bug) | **DONE (2026-10-19/20)** | 92,959/92,960 announcements FinBERT-scored; `M&M.NS.parquet` empty; NaN convention fixed in `news_context.py` |
| Observability §7.5 (blame taxonomy, funnel graphs) + §14E/F (coverage, metrics) | **SPEC DONE 2026-10-02, implementation TODO** | §§7.5, 14E, 14F drafted; `joined_diagnostic.py`, coverage-audit command, news-flag carry-through still TODO |
| G3 PAUSE verdict | **UNCHANGED** | Phase 8 pass earns a retest only, never auto-unpauses (§8) |

---

## 1. Change Log — what changed from v1, and why

| # | Change | Why |
|---|---|---|
| 1 | Removed the "Radfar 2025" citation as the track's justification | Searched for it; could not locate the paper. An unverifiable citation should not be the sole stated reason for a multi-day engineering investment. Replaced with the project's own verified evidence (§2). |
| 2 | Added **Q0-D1**, a zero-shot gate, between harness-build and thinking-cap ablation | v1 spent five phases (environment, acquisition, harness, thinking-cap ablation, full 92,960-row STaR generation) before the first reality check at Phase 8. The harness is usable zero-shot the moment it's built — there is no reason to wait. This mirrors the exact discipline that made U2-D1 and U3-D1 cheap instead of expensive. |
| 3 | Made holdout exclusion from the STaR corpus an explicit, tested, code-enforced rule | v1 described the STaR corpus as "92,960 announcements... each with a known triple-barrier label" with no stated boundary. If the sealed test window's rows were included, the LoRA adapter would be fine-tuned on reasoning traces whose correctness was judged using test-window outcomes — a leakage class worse than anything in the touch ledger so far. §5 makes the boundary explicit and enforced. |
| 4 | Added a keep-rate-by-class diagnostic to Phase 5 | STaR's outcome filter (keep iff vote == historical label) can't distinguish genuine signal from lucky guessing in a domain where UP's true AUC is ~0.51–0.52. A 200-case human audit doesn't statistically cover this. Tracking keep-rate by class against the known AUCs is a cheap, automatic sanity check that the filter is tracking something real. |
| 5 | Reconnected the success criterion to economics | A prior decision in this campaign explicitly separated economics into its own thread so it would "not ride along silently in UP-AUC reports." Track Q's gate is still UP-AUC-based (correctly — that's the right first question), but §8 now states explicitly that a pass **earns a new economics retest**, not an automatic reversal of the standing PAUSE verdict. |
| 6 | Extended touch-ledger logging to every sample that reads realized outcomes, including the thinking-cap tuning set | Consistency with how smaller reads have been logged throughout the campaign. |
| 7 | Added a per-row rating ledger (§7.1), separate from and feeding into the existing prediction ledger | Addresses "for each row... we need its benchmark to see how the model performed" — every LLM-rated row now has its own trackable, queryable record, not just aggregate metrics. |
| 8 | Added an explicit scope statement (§3) | Addresses "mention clearly what we are doing and what not." |
| 9 | Added cold-start, model-drift, and multi-checkpoint edge cases (§11) | Not covered in v1; realistic operational gaps for a locally-served LoRA adapter. |

---

## 2. Why This Track Exists — revised, evidence-based justification

This track is **not** justified by an external paper. It is justified by this project's own completed, verified diagnostics:

- **U3-D1** (FinBERT sentiment content, company scope): `P(UP | positive sentiment) = 43%` vs. `P(UP | no news) = 45%`. Sentiment *tone* carries no marginal information for this label — the base rate moved the wrong way.
- **U2-D1** (announcement presence/scope — group, sector, market): all three scope slices came back at or below the pre-registered bar; market-high days ranked at coin-flip.

Both of those nulls are about **what a tone classifier and a presence flag can extract from text.** Neither tested a structurally different capability: an analyst-style reasoning process that classifies the event type, compares it against a historical base rate, and checks whether the move looks already priced in. That "is this priced in" check in particular is mechanistically distinct from anything a sentiment score can produce — tone doesn't know whether the market already moved on anticipation. This is the actual, falsifiable hypothesis Track Q tests: **reasoning depth, not sentiment polarity, is the missing ingredient** — and it is tested cheaply (§4, Q0-D1) before any expensive commitment, exactly because it might also turn out to be false.

---

## 3. Scope — explicit in/out

### In scope
- Rating already-ingested text (NSE announcements + GDELT/RSS — no new data source, no new ingestion pipeline)
- Producing a fixed set of numeric/categorical features consumed by the existing, frozen LightGBM pipeline
- A cheap zero-shot pre-check before any training is spent
- A single global LoRA adapter, trained via STaR (supervised fine-tuning on self-generated, outcome-and-leakage-filtered, human-audited reasoning traces)
- Full, code-enforced exclusion of the sealed test window from every stage of training and tuning
- One pre-registered ablation gate, judged the same way every prior track has been judged (DM test, PBO-aware, tripwire)

### Explicitly out of scope for this rung
- **Replacing LightGBM as the decision model.** The LLM is a feature source. LightGBM converts its output into calibrated probabilities. This is non-negotiable per the standing architecture decision.
- **Stacking.** No ensemble of LLM-vote-as-model alongside LightGBM. One accountable decision point.
- **Reinforcement learning (GRPO).** Rung 4 only, after the leakage scanner is battle-tested across multiple rungs — RL finds reward-hacking exploits faster than supervised fine-tuning does, and this domain's near-random UP label is exactly the kind of noisy reward signal RL would latch onto destructively.
- **Per-stock or per-sector LoRA adapters.** Would fragment an already-thin dataset (~1,860 articles/symbol on average, skewed) and invite memorization. Gated behind future evidence of real cross-symbol heterogeneity, not assumed up front.
- **New data sources.** This track uses exactly the text already ingested for Tracks U/U2/U3. It is a different *reader* of the same text, not a different corpus.
- **Revisiting DOWN or FLAT.** Those are closed (DOWN threshold-inert, FLAT fixed via τ=0.40). Track Q targets UP exclusively, because UP is the only side with a live, unresolved information question — DOWN and FLAT's problems were already shown to be ranking-absent and boundary-only respectively, neither of which a better *text reader* would fix.
- **Relabeling the target.** That's Track I0 (horizon/threshold sweep), a separate, already-running thread. Track Q holds the label fixed and only changes how text becomes a feature.
- **Automatically reversing the G3 PAUSE verdict.** Covered explicitly in §8.

---

## 4. Data Feeding — exactly what goes in, at each stage

### 4.1 Source (unchanged from existing pipeline)
NSE announcements (currently 92,960 records, 50 symbols, cursor-resumable) + GDELT/RSS rows, exactly as ingested for Tracks U/U2/U3. No new fetch code.

### 4.2 Point-in-time alignment (unchanged, reused)
Every case uses the existing `decision_date()` function (the B-05 fix: UTC→IST conversion, 15:30 cutoff, NSE trading-calendar-aware deferral). No new timing logic — reusing proven, already-tested code is deliberate, since timing bugs are the single most dangerous class of error in this whole project.

### 4.3 Per-case context block (what the model actually sees)
For a given `(symbol, decision_date)`:
```
SYMBOL: <ticker>        SECTOR: <sector>
PRE-T CONTEXT:
  5-day trend:        <value>
  Volume shock (z):   <value>
  Sector move (5d):   <value>
  FinBERT prior:       <score, or NO_NEWS>
ARTICLE TEXT (full, un-summarized):
  <text, TRUNCATED flag appended if > 1,500 tokens>
INSTRUCTION: <analyst playbook, §6.2>
```
Zero information dated after `decision_date` enters this block, by construction — same B-05-aligned guarantee the rest of the pipeline relies on.

### 4.4 Two distinct feeding modes — do not conflate them
| Mode | When | Data window | Purpose |
|---|---|---|---|
| **Training-time feeding** | Phase 4 (STaR generation), Phase 6 (QLoRA fit) | **Strictly pre-holdout only** — see §5 | Builds the reasoning corpus and trains the adapter |
| **Inference-time feeding** | Phase 7 (nightly) | Any date, including post-holdout-start, including live future dates | Produces the features the frozen LightGBM actually consumes |

These must never share code paths that could blur the boundary — the STaR builder and the nightly rater are separate entry points, both calling the same harness, but gated by different date filters (§5.2).

---

## 5. Train / Validation / Holdout Split — the critical section

This section exists because v1 didn't specify it, and that gap is the single most serious risk in the original design.

### 5.0 Pinned constants (live values, verified 2026-10-01)

| Constant | Value | Source |
|---|---|---|
| `HOLDOUT_START` | `2023-08-29` | `results/metrics.json:2`, `artifacts/model_metadata.json:208` (`test_start`, `n_test=37292`) |
| `EMBARGO_DAYS` | `5` | `config/config.yaml:99-102` (`validation.embargo_days`, `purge_days: 5`) |
| `STAR_CUTOFF` | `2023-08-24` | `HOLDOUT_START − EMBARGO_DAYS` — no STaR row may have `decision_date >= 2023-08-24` |
| `N_TEST` | `37292` | `src/stockml/evaluation/touch_ledger.py:18` |

The `2023-12-20 / n=670` date appearing in older docs (`MASTER_PLAN.md:251`) is the stale single-symbol run and must never be used as a boundary in this track. `star_builder.py` and every Pool filter import these constants from one place; hardcoded dates elsewhere are a defect.

### 5.1 The four distinct data pools

| Pool | Date range | Used for | Touches sealed holdout? |
|---|---|---|---|
| **A — STaR training corpus** | `decision_date < holdout_start − embargo_days` | Phase 4 generation, Phase 6 LoRA fit (train split) | **No** |
| **B — STaR validation slice** | 5% of Pool A, held out, same pre-holdout range | Phase 6 early-stopping | **No** |
| **C — Q0-D1 / thinking-cap tuning sample** | `n=1500` stratified cases (500 DOWN / 250 FLAT / 750 UP approx, matching Pool A base rates) drawn from Pool A's date range, **not** from Pool D, with case_ids excluded from Pool A-train and Pool B | Phase 2.5 zero-shot gate (§7.4), Phase 3 thinking-cap ablation (§8) | **No sealed-holdout touch** (Pool C reads still get a rating-ledger row per §5.4, but cost no holdout touch) |
| **D — Sealed test window** | `decision_date >= 2023-08-29` (`HOLDOUT_START`, `n=37292`, the window every prior track since G1-attempt-2 has used) | **Only** Phase 8's final ablation gate | **Yes — exactly once, logged** |

`holdout_start` and `embargo_days` are the pinned values from §5.0 (`2023-08-29`, `5`) — the same values LightGBM's purge/embargo scheme already uses — this track does not define its own boundary, it inherits the existing one, so there is never a question of the LLM and the GBDT disagreeing about what "test" means.

### 5.2 Code-level enforcement, not a comment

The STaR builder (`star_builder.py`) must assert this boundary, not just document it:

```python
from datetime import date
HOLDOUT_START = date(2023, 8, 29)  # §5.0 pinned — never 2023-12-20 (stale single-symbol run)
EMBARGO_DAYS = 5                   # config/config.yaml validation.embargo_days
STAR_CUTOFF = date(2023, 8, 24)    # HOLDOUT_START − EMBARGO_DAYS

assert df["decision_date"].max().date() < STAR_CUTOFF, \
    "STaR corpus must not include holdout-adjacent dates — leakage guard tripped"
```

This assertion runs **before** any generation starts, and a failed assertion halts Phase 4 entirely — no partial corpus, no manual override. The same guard is unit-tested with a deliberately contaminated synthetic input to confirm it actually fires, the same way the delivery-CLV NaN edge case was unit-tested in H1b.

### 5.3 Why Pool C is not Pool D

Q0-D1 and the thinking-cap ablation need a sample to measure agreement-with-label against. It would be tempting to use the sealed test window for this, since that's the "real" target — **do not.** That would spend a precious holdout touch on a hyperparameter-tuning step, the exact thing the cumulative touch ledger exists to prevent. Pool C is drawn from the same date range as Pool A (pre-holdout), held out from the STaR training split the way Pool B is, but used for a different purpose (gating and tuning rather than fine-tuning).

### 5.4 Ledger logging for every pool touch

| Pool | Ledger entry required? |
|---|---|
| A | No — it's training data, not a holdout read |
| B | No — same reason |
| C | **Yes** — any read that compares model output against realized historical outcomes counts, per the standing rule, even outside the sealed window, because it still consumes a finite supply of "genuinely fresh" pre-holdout evaluation data |
| D | **Yes, exactly once**, at Phase 8, with the touch ledger's running total consulted *before* the pre-registered bar is finalized (§8) |

---

## 6. Architecture

### 6.1 Diagram

```
 NSE announcements / GDELT / RSS (existing fetch, unchanged)
        |
        v
 +------------------+     +---------------------------+
 | FinBERT (stays,  |     | Qwen3-8B-Q4_K_M            |
 | CPU, baseline    |     | + global LoRA (rung 1)     |
 | rater)           |     |                             |
 +--------+---------+     +----------+------------------+
          |                          |
          |              +-----------+-----------+
          |              |  Q0-D1 ZERO-SHOT GATE |  <-- NEW: runs on Pool C,
          |              |  (base model, no LoRA) |      before any training spend
          |              +-----------+-----------+
          |                          |
          |                    pass? -----no----> STOP. Infra kept for rung 2
          |                          |             review. No STaR, no LoRA.
          |                         yes
          |                          |
          |                          v
          |              thinking-cap ablation (Pool C)
          |                          |
          |                          v
          |              STaR generation (Pool A only, asserted)
          |                          |
          |                          v
          |              filter + scanner + audit + KEEP-RATE-BY-CLASS CHECK
          |                          |
          |                          v
          |              QLoRA training (Pool A/B)
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
    PHASE 8 GATE (Pool D, sealed test window, ONE TIME, LOGGED)
                       |
             pass -> new economics retest required (not auto-unpause)
             fail -> adapters deleted, infra kept for rung 2, logged
                       |
                       v
          backtest / monitor / ledger (all unchanged)
```

### 6.2 Component-by-component: what it is, why, pros, cons

| Component | What / why | Pros | Cons |
|---|---|---|---|
| **Qwen3-8B-Q4_K_M** | Largest reasoning-capable model fitting 6GB VRAM (~4.7GB weights + ~1GB KV at 4k context). Native `<think>` mode. | Genuine multi-step reasoning vs. FinBERT's single-pass classification; local, free, no API cost or rate limit; 4-bit quantization keeps it within hardware budget | Quantization degrades precision versus full-weight inference (bounded, but real); not finance-domain-pretrained, so general-purpose reasoning may not transfer cleanly to market-specific judgment; slower per-article than FinBERT by orders of magnitude — nightly rating of a large daily news volume could become a throughput bottleneck as the universe grows |
| **4096-token context** | Worst-case call ≈2,700 tokens (full article + context block); 4k covers it with margin | Keeps KV cache small, leaves VRAM headroom for the LoRA adapter during training | A small number of very long filings could still exceed this — handled via the TRUNCATED flag, but truncation itself discards information, which is a real (if rare) cost |
| **Full article text, no summarizer** | Announcements are short (median 278 chars); summarizing a 278-char announcement to save tokens would mean reasoning over someone else's compressed judgment for no real savings | Simpler pipeline, one fewer model in the loop, no summarizer-introduced distortion | For the rarer long-form articles (p95 616 chars, plus occasional longer filings), the model does more work per case than a summarized version would — acceptable given it's still far under the context budget |
| **One global LoRA + identity tokens, not per-symbol adapters** | ~1,860 articles/symbol on average, skewed — too thin to train isolated adapters without memorizing | Shares learning across the whole universe, avoids the ×50 training cost of per-symbol adapters, avoids per-symbol overfitting | A genuinely idiosyncratic company (unusual disclosure style, unusual sector dynamics) gets less tailored treatment than a dedicated adapter could give — explicitly deferred, not solved, pending future evidence of real cross-symbol heterogeneity |
| **Q0-D1 zero-shot gate (new)** | Runs the harness with the frozen base model, no LoRA, on Pool C, before any training spend | Converts the whole track's first real test from "after 6 phases of building" to "after 1 phase of building" — the single highest-leverage change in this revision | Zero-shot performance is a lower bound, not a true measure of what the fine-tuned adapter could achieve — a borderline zero-shot result is a judgment call, not an automatic kill (see §8.1) |
| **STaR (outcome + leakage + human filter), not raw SFT on all generations** | Only keeps reasoning traces that reached the correct label *and* passed a leakage scan *and* survived human rubric review | Filters out both wrong answers and right-answers-for-wrong-reasons (leakage); the human audit catches process failures an automatic filter can't | In a near-random-AUC domain, the outcome filter alone can't distinguish genuine signal from lucky guesses (§5 keep-rate-by-class diagnostic exists specifically to surface this risk, not eliminate it) |
| **QLoRA (4-bit base, LoRA rank 16/alpha 32 on q/k/v/o+mlp) via unsloth** | The only finetuning approach that fits a trainable adapter on 6GB VRAM; full fp16 finetune needs ~60GB+ | Checkpoint-resumable, small trainable parameter count (~5% of the model) limits overfitting surface, proven technique | Adapter capacity is inherently limited — if the needed signal requires deeper representational change than a rank-16 LoRA can express, this approach structurally can't find it (this is explicitly why rung 3, full-parameter finetune, exists as an escalation path, gated behind rung-1 showing signal first) |
| **STaR over RL at rung 1** | Supervised fine-tuning on filtered self-generated traces, not reinforcement learning on a direction reward | Far more stable, far less prone to reward hacking, and the leakage scanner gets real-world testing before RL (which exploits scanner gaps faster than SFT does) would ever be exposed to it | Learns only from traces that happened to be generated and happened to pass filtering — doesn't actively explore alternative reasoning strategies the way RL eventually could (deferred to rung 4, deliberately) |
| **FinBERT kept running in parallel, not replaced** | `finbert_qwen_disagree` becomes a feature in its own right | Cheap fallback if Qwen's rater ever fails health checks; disagreement itself may carry information (high-disagreement cases could be genuinely ambiguous news) | Running two raters costs more compute than running one — acceptable given FinBERT is CPU-cheap and already-amortized infrastructure |

---

## 7. Evaluation & Benchmarking Framework

### 7.1 Per-row rating ledger (new) — "for each row, what happened"

A persistent table, append-only, separate from but joinable to the existing prediction ledger:

```sql
-- qwen_rating_ledger
case_id            TEXT PRIMARY KEY
symbol             TEXT
decision_date      DATE
pool               TEXT        -- 'A' | 'B' | 'C' | 'D'
model_stage        TEXT        -- 'zero_shot' | 'lora_r1'
qwen_vote          TEXT        -- UP | DOWN | FLAT-equivalent abstain
qwen_sent          DOUBLE
qwen_conf          DOUBLE
qwen_trace_len     INT
finbert_score      DOUBLE      -- joined for direct comparison
finbert_qwen_disagree DOUBLE
historical_label   TEXT        -- the triple-barrier answer key
vote_matches_label BOOLEAN     -- the STaR outcome-filter criterion, logged explicitly
kept_for_training  BOOLEAN     -- post leakage-scan + human-audit status
drop_reason        TEXT        -- NULL if kept; 'wrong_vote' | 'leakage' | 'audit_fail' otherwise
truncated          BOOLEAN
no_news            BOOLEAN
harness_fail       BOOLEAN
trace_path         TEXT        -- pointer into results/traces/
```

Every case generated in Phases 2.5–7 gets a row here, regardless of whether it ends up in the kept STaR set. This is what makes "did this method actually work, and for which kinds of cases" answerable at the row level rather than only as an aggregate percentage — you can slice by sector, by event type, by pool, by confidence, exactly the way every prior D1 diagnostic has sliced the frozen LightGBM predictions.

### 7.2 Keep-rate-by-class diagnostic (new, from §1 item 4)

Computed automatically at the end of Phase 5, before training starts:

```
keep_rate(class) = count(kept_for_training AND historical_label == class) / count(historical_label == class)
```

**Interpretation guide, stated before the numbers are seen (pre-registered, same discipline as every D1 bar):**
- If `keep_rate(DOWN)` exceeds `keep_rate(UP)` by more than `2×SE_diff` (two-proportion z on the kept/total counts per class), consistent with DOWN's known real AUC (~0.523–0.526, `results/up_autopsy.json:11-14`) vs. UP's known near-null AUC (~0.514), that's evidence the filter is tracking genuine signal, not just coin-flip luck.
- If the two keep-rates are statistically indistinguishable, that's a warning the UP-side kept set is dominated by lucky guesses — training on it risks reinforcing confident-sounding but spurious reasoning. This does not automatically halt the track, but it downgrades the prior on Phase 8 passing, and should be reported alongside the Phase 8 result, not buried.

### 7.3 Benchmark comparison table — this track's results against the full campaign history

Every reported result for Track Q must be shown **next to**, not instead of, the established campaign benchmarks. Current standing numbers, for reference and for the comparison table every future report should reuse:

| Metric | Established benchmark | Source |
|---|---|---|
| Headline test accuracy (F-adopted, current live model) | 47.14% | Track F |
| Majority-class baseline (G1 target, not yet met) | 47.85% | G1 |
| UP OvR AUC (structural ceiling, four independent attempts) | 0.5127–0.5181 (inside ~2 SE of ~0.515) | H1/H1b/U/baseline |
| DOWN OvR AUC | 0.523 | standing diagnostics |
| DOWN F1 / recall | ~0.60 / ~0.75 | standing diagnostics |
| FLAT OvR AUC (best of three classes) | 0.622 | standing diagnostics |
| FLAT F1 (post τ=0.40 adoption) | 0.2052 (from 0.2069, preserved) | Track F |
| Best economics across entire campaign (F&O-proxy, kill-confirming) | Sharpe −0.28, net −4.2% | G3c |
| Sizing viability | Falsified — negative mean net at every confidence bucket | G3-0 |
| Cost-to-edge ratio at best configuration | ~2:1 against | G3 |
| Campaign PBO threshold | < 0.3 | G3 kill rule |
| Campaign DM significance threshold (Bonferroni, as of G3) | p < 0.0038 over 13/15 eligible touches | G3 kill rule |

Track Q's Phase 8 report must reprint this table with Track Q's own result appended as a new row, so "did this help" is always answerable by comparison, never by Track Q's number in isolation.

### 7.4 Pre-registered bars (since none existed in v1)

No threshold existed in the original document — this version sets them explicitly, before any data is generated:

| Gate | Bar | Rationale |
|---|---|---|
| **Q0-D1 (zero-shot)** | `n=1500` Pool C cases (§5.1). Vote-agreement with label on Pool C exceeds FinBERT's equivalent agreement rate on the same pool by a margin exceeding 2×SE (two-proportion z, `SE_diff = sqrt(p1(1-p1)/n + p2(1-p2)/n)`) | Power: at `p≈0.47`, `SE_diff≈0.0182`, `2×SE≈3.6pp` — detects a ≥4pp lift. `n=200` would give `2×SE≈10pp` and could not separate anything; that is why v1's 200-case sizing is rejected here. Stratified sample (seed `42`), case_ids frozen and excluded from Pool A-train/B. Same statistical standard used to establish the structural-ceiling finding in §7.3 |
| **Phase 8 (final)** | UP OvR AUC on Pool D improves over the current baseline (0.518) by a margin exceeding 2×SE **AND** Diebold-Mariano p-value clears the Bonferroni-corrected threshold recomputed against the **full, current touch ledger** (snapshot 2026-09-20: `results/holdout_touches.jsonl` holds 22 rows, 21 eligible / 19 unique after DOWNFLAG triplicate — `0.05/21≈0.0024`, stricter than G3's `0.05/13≈0.0038`; the Phase 8 bar must be recomputed at run time, never copied verbatim) | Consistent with the "full campaign" PBO scope decision already made for G3 — a stricter, not looser, standard as more touches accumulate |

### 7.5 Observability — blame taxonomy, slices, and funnel graphs (was TODO, now specified)

Status: **SPEC DONE, implementation TODO** — the six subsections below are the build contract for `src/stockml/evaluation/joined_diagnostic.py` (new, TODO) plus small extensions to `evaluate.py`, `backtest.py`, and `economics.py`. No thresholds here are judged until Phase 8; slice diagnostics in §7.5.2–7.5.4 are report-only.

Join key for everything in this section: `(timestamp, symbol)` with timestamp normalized to date (no TZ) and symbol in `ADANIPORTS.NS` form. Verified lossless today: `results/predictions.parquet` (37292×14, `evaluation/evaluate.py:98-124`) ⨝ `results/backtest_timeseries.parquet` (37292×21, `backtest/backtest.py` → `backtest_metrics.json`) = 37292/37292. The rating ledger (§7.1) joins on `(symbol, decision_date)` where `decision_date` maps to the prediction `timestamp` via `features/decision_date.py` (B-05: UTC→IST, 15:30 cutoff, NSE calendar). Slice floors: `N ≥ 200` per bucket and `2×SE` bars throughout; era-stability via `cross_era` (≥3 eras `n≥50`, max−min < 0.25) exactly as `evaluation/up_autopsy.py:114-234` defines it.

#### 7.5.1 Qwen-SHAP rank (did LightGBM listen?)

Hook: `evaluation/evaluate.py:196-221` (global SHAP on `X_test.sample(min(1000,len), random_state=42)` via `shap.TreeExplainer`, `mean(|v|)`, writes `results/shap_feature_importance.csv` + `shap_summary.png` top-20). Baseline to beat: all `news_*` at `0.0` today (`shap_feature_importance.csv:129-137` — `news_finbert_score_mean`, `news_count`, `news_relevance_mean`, `news_sentiment_mean/std` all dead-rank bottom). Future `qwen_*` family from §6.1 (`qwen_sent`, `qwen_vote_up/down`, `qwen_conf`, `qwen_trace_len`, `finbert_qwen_disagree`, `qwen_available`) flows through `news_context.py` (`score_col="qwen_score"`) → `join.py` → frozen LightGBM — no new SHAP file, same CSV.

Spec: Phase 8 report prints rank + `mean_abs_shap` for every `qwen_*` next to `finbert_*`/`news_*`. Bar (pre-registered): `used = rank ≤ 40` per feature, matching prior U/H1b usage language; the family as a whole is `listened-to` iff any member clears it. Failing this with a passing UP-AUC means the lift came from elsewhere — investigate, do not ship. Generating script: existing `evaluate.py` SHAP block (extended with rank table, no new file).

#### 7.5.2 Calibration on news slice (is confidence honest on news rows?)

Hook: `evaluation/metrics.py:70-99` (`expected_calibration_error(y_true, proba, n_bins=10)`, pure, reuse as-is) called at `evaluate.py:47-54` (today global only → `results/metrics.json: ece=0.04244 + ece_bins[10]` + `calibration.png`; `confidence_accuracy.csv` likewise global-only). Gap: `predictions.parquet` has no news flag, so slicing is structurally impossible until the flag is carried (from `news_context.py`/`join.py` availability or the ledger `qwen_available` column).

Spec: report global ECE alongside news-available-subset ECE (`qwen_available=1`, equivalently `news_count>0`) with identical `n_bins=10` edges and per-bin `count/accuracy/confidence` table + reliability overlay. The campaign bar stays global (`MASTER_PLAN:57` `ECE < 0.05`); the slice is report-only — its job is catching "confident on news, wrong on news" (today's top-bin gap `0.27–0.40` at `0.8–1.0` in `metrics.json:130-156` is exactly the failure mode to watch). Generating script: `joined_diagnostic.py --slice news_available` (TODO).

#### 7.5.3 Cost on news slice (does the lift survive costs?)

Hook: `backtest/backtest.py:34-113` (`_event_backtest` emits per-trade `weight/gross_return/cost/net_return` via `round_trip_cost_rate()` — 31.5 bps delivery / 15 bps futures) + `evaluation/economics.py:121-141` (`run_variant(row_mask=...)`, mirroring the `fo_proxy_eligible` turnover-quartile pattern). Gap: `backtest_timeseries.parquet` carries no `news_*/qwen_*` column, and no economics variant passes a news mask.

Spec: report `net_return_sum`, `cost_paid`, trade count, and Sharpe separately for news-available vs. no-news rows under the identical cost model. Track Q deliberately earns (never auto-runs) the economics retest (§8: pass opens a new G-series); this slice is its required input — a UP-AUC win concentrated on rows whose `cost_paid` exceeds `gross_return_sum` (today's campaign ratio is ~2:1 against, `economics.json`, G3 PAUSE) must not unpause anything. Generating script: `economics.py` news-mask variant (TODO), output appended to `economics.json`.

#### 7.5.4 Trace quality vs. accuracy (does better reasoning predict better?)

Hook: TODO `data/star_builder.py` + TODO `tests/test_qwen_guards.py`; ledger already carries `qwen_trace_len/trace_path/kept_for_training/drop_reason` (§7.1) — add `quality_score` (audit rubric 0–3: checklist adherence, context-only citation, confidence proportionality) and `audit_verdict` (pass/fail) columns. Existing bars it builds on: 100% outcome filter + 100% regex scanner (Appendix B, unit-tested) + ~200-case human audit with `fail-rate > 30% → playbook revision` (Phase 5), plus §7.2 keep-rate `2×SE_diff`.

Spec: Phase 5 report shows vote-agreement-with-label split by audit-pass vs. audit-fail (fallback: `trace_len` quartile) with a pre-registered `2×SE` bar fixed before scoring. Passing means quality predicts correctness (reasoning signal); failing with a passing keep-rate means the filter keeps lucky guesses with confident prose (reinforcement risk — downgrade Phase 8 prior, report alongside, never bury). Generating script: `star_builder.py --quality-report` (TODO).

#### 7.5.5 Blame taxonomy + funnel graphs (which failure, where, action)

Taxonomy (every Phase 2.5–8 miss gets exactly one class from the joined row):

| Blame class | Rule (evaluated in order) | Slice evidence today (`file:key`) | Action |
|---|---|---|---|
| `coverage-gap` | `qwen_available=0` (or `NO_NEWS`) on the miss | `news_co 0.52% / mkt 1.44%` (§14E) | A1/A2 plumbing, not modeling |
| `LLM-wrong` | vote ≠ label, `harness_fail=0` | keep-rate §7.2; `positive AUC 0.486 < flip` (`u3_d1.json`) | playbook revision / STaR filter |
| `LLM-unreliable` | `HARNESS_FAIL` / `TRUNCATED=1` / leakage-hit | App.B scanner; §11 edge table | harness/scanner fix, never train on it |
| `ignored-by-GBDT` | vote = label but `P(up)` low / `qwen_*` SHAP ≈ 0 | §7.5.1 rank table | feature scaling / availability-flag check |
| `gated` | P correct but confidence/threshold abstained | `flat_threshold.json`, `confidence_accuracy.csv` | threshold review (own track, not Q) |
| `cost-killed` | gross > 0 but net ≤ 0 | `economics.json` (~2:1 against), §7.5.3 slice | G-series sizing, never Q retrain |
| `miscalibrated` | high-conf miss on news rows | §7.5.2 slice (top-bin gap 0.27–0.40) | calibration review |
| `drifted` | miss in high-PSI feature regime | `monitoring.json` (`atr 4.27`, `gold_vol 10.49`), §7.5.6 | regime report, no silent refit |

Graphs (each names x/y/slice, `N ≥ 200`, cross-era check, script + output path):

1. Funnel — `92,960 raw → 92,873 dated (87 undated dropped) → Pool A (<2023-08-24) → Pool C n=1500 frozen → kept (keep_rate by class) → Pool D once`. Script: `joined_diagnostic.py --funnel` → `results/qwen_funnel.json` + `qwen_funnel.png`. Answers "where did the N go".
2. PSI × accuracy — top-20 SHAP PSI (`monitoring.json`) vs per-bucket accuracy/miss on the same dates. Script: `joined_diagnostic.py --psi-accuracy` → `results/psi_accuracy.png`. Answers "when did drift hurt".
3. Cost waterfall — `gross_return_sum` vs `cost_paid` → net, overall + news slice, delivery 31.5bps / fo 15bps. Script: `economics.py` variant → `results/cost_waterfall.png`. Answers "did money survive".

Tier3 upgrades (explicit TODO, not claimed done): `worst_20.csv` today is UP→DOWN-only with `ctx_*` but no SHAP/text (`up_autopsy.py:114-122,192-234`) → add per-row SHAP + linked article; sliced 3×3 confusion missing (only UP-miss rates + `ovr_auc`) → add sector/regime/vol full matrices; missing `roc_pr.png / regime_performance.png / news_coverage.png` (`MASTER_PLAN §12`) → generate from the same join.

#### 7.5.6 Test-question bank (every test answers exactly one question)

| Question | Metric | Bar | Artifact | Action on fail |
|---|---|---|---|---|
| Does reasoning beat tone? (Q0) | vote-agreement Δ vs FinBERT, Pool C n=1500 | `> 2×SE ≈ 3.6pp` (§7.4) | rating ledger + §7.1 | stop, bank infra |
| What thinking budget? (cap) | paired agreement, n=1000 × {256,512,1024} | McNemar best cap | `thinking_cap` entry | use winner, log |
| Luck or signal? (keep-rate) | `keep_rate(DOWN)−keep_rate(UP)` | `> 2×SE_diff` (§7.2) | `star_kept/dropped.jsonl` | downgrade Phase 8 prior |
| Does +qwen lift UP? (Phase 8) | UP-AUC Δ + DM | `> 2×SE` AND recomputed Bonferroni | metrics + touch ledger | delete adapters, keep infra |
| Is confidence honest on news? | ECE news slice (§7.5.2) | report-only | calibration table | calibration review |
| Does lift survive costs? | net/Sharpe news slice (§7.5.3) | report-only → G-retest gate | `economics.json` | hold PAUSE |
| Did GBDT listen? | qwen SHAP rank (§7.5.1) | any member `rank ≤ 40` | `shap_feature_importance.csv` | investigate, don't ship |
| Does quality predict correctness? | audit-pass Δ (§7.5.4) | `> 2×SE` | quality report | playbook revision |

---

## 8. Phases

### Phase 0 — Environment — **DONE**
`nvidia-smi` check (6GB VRAM visible), 25GB+ disk free, Ollama install (Windows app, inference), unsloth + trl + peft (training stack — Ollama cannot train). `requirements-llm.txt` pinned separately from the core env. Versions recorded in `artifacts/state/qwen_setup.json`. Ledger entry: `qwen_setup`.

### Phase 1 — Model Acquisition — **DONE (sampled path, 2026-10-01)**
Inference artifact: `ollama pull qwen3:8b` (Q4_K_M — hash `500a1f067a9f`, 5.2GB, verified 2026-10-01, server 0.34.2, driver 617.14), health-checked with one `enable_thinking=True` prompt, parsed for a valid `<think>` block + JSON vote — fail means reinstall, never proceed on a sick model. Training artifact: HF weights `Qwen/Qwen3-8B` (~16GB disk), loaded 4-bit via unsloth — **TODO: WSL-side staging still unverified (disk check blocked), carried as TODO, not blocking Phase 2.** Model hashes + quant + versions recorded.
**Throughput probe — DONE, gate FAILED for full build:** 5-case corrected `@ctx2048` median 20.3s/article → `N=92,960 ≈ 524h (21.8d) > 14d` → **sampled STaR committed** (announcements-only, `relevance ≥ 0.5` first). Measured deviations pinned: `ctx 4096 → 2048` (4096 OOM/crashes on this stack), `num_predict ≥ 1024` required with `think=True`, reasoning read from `message.thinking` (not `<think>` tags — §14A patched). Full numbers in `artifacts/state/qwen_setup.json: phase1`.

### Phase 2 — Harness (`src/stockml/data/qwen_harness.py`) — **PENDING / TODO (next)**
Analyst playbook v1 (full text: Appendix A — event classification → historical base-rate comparison → positioning/priced-in check → vote+confidence commitment). Strict JSON schema (`{sentiment: float −1..1, vote: UP|DOWN|FLAT, confidence: 0..1, trace_steps: [1,2,3,4]}`) with validator, 3-retry-then-`HARNESS_FAIL` policy, 120s per-call timeout, concurrency ≤2 (GPU-bound — higher values do not help a single Ollama server). Explicit leakage self-check instruction in the prompt. Five hand-written few-shot examples (bootstrap; replaced by kept STaR pairs once they exist — playbook v1→v2 as a versioned commit). Ollama client at `enable_thinking=True`, temperature 0.6, top-p 0.95 (Qwen3 thinking-mode spec), thinking-budget parameter, `<think>` parser, TRUNCATED/NO_NEWS flagging, per-call trace logging.

### Phase 2.5 — Q0-D1 Zero-Shot Gate **(new, the key insertion)** — **PENDING / TODO (gated on Phase 2)**
Run the harness — base model, **no LoRA** — on Pool C (§5.1). Log every case to the rating ledger (§7.1). Compute vote-agreement-with-label, compare against FinBERT's agreement on the identical Pool C cases. Judge against the pre-registered bar (§7.4). **Pass → proceed to Phase 3. Fail → stop here.** Adapters were never trained, so there's nothing to delete; harness and rating-ledger code are kept as reusable infrastructure for a future rung-2 reconsideration (e.g., if a bigger reasoner becomes available), logged as a clean, documented stop — not a quiet abandonment.

### Phase 3 — Thinking-Cap Micro-Ablation (only reached if Q0-D1 passes) — **PENDING / TODO (gated on Q0-D1 PASS)**
`n=1000` Pool C cases (subset of the Q0-D1 1500, seed `43`) × budgets {256, 512, 1024} → paired agreement-with-label picks the global cap (McNemar on the same 1000 cases, not three independent samples). Power: paired `n=1000` detects a ~3pp systematic cap difference at ~80% power; `n=200` gives `2×SE≈10pp` and is rejected as noise. Overflow rule (permanent): an unfinished `<think>` block counts as a low-confidence vote, never a silently truncated one. Result committed, ledger entry `thinking_cap` (rating-ledger rows, no holdout touch).

### Phase 4 — STaR Dataset Builder (`src/stockml/data/star_builder.py`) — **PENDING / TODO (gated on Phase 3 + Phase 6 probe; sampled scope per Phase 1)**
Input: Pool A only, **with the §5.2 assertion enforced before generation starts.** Per-case prompt per §4.3. Generation via the harness (frozen base model), saved as (prompt, `<think>` trace, vote) to `data/llm/star_candidates.jsonl`. Resumable via per-case status log — reruns pick up unfinished cases only. Single Ollama server, concurrent requests, incremental saves.

### Phase 5 — Filter, Scanner, Audit — **PENDING / TODO (gated on Phase 4)**
Outcome filter (automatic, 100%): keep iff vote == historical label. Leakage scanner (automatic, 100% — pattern list: Appendix B): post-decision-date dates, outcome mentions, future price levels — dropped even if the vote happened to be correct. Human audit (~200 kept cases): checklist adherence, citation of provided context only (no invented facts), confidence proportionate to evidence — fail-rate >30% triggers a playbook revision, not training. **New: keep-rate-by-class computed and reported (§7.2) before Phase 6 begins.** Outputs: `star_kept.jsonl` (trains) / `star_dropped.jsonl` (analysis only, never trained).

### Phase 6 — QLoRA Training (unsloth) — **PENDING / TODO (gated on Phase 5; 10-step pre-probe must pass before Phase 4 generation)**
4-bit frozen base (NF4, double-quant), LoRA rank 16 / alpha 32 / dropout 0.05 on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (~5% of params trainable). Full hparams: Appendix C (LR `2e-4` cosine, effective batch 16 via micro-batch 1–2 × grad-accum, seq-len 4096 with 3072 fallback, grad-checkpointing ON, 8-bit AdamW, seed 42). Checkpoints roughly every 15 minutes wall-clock (timed trial run sets `save_steps`), `save_total_limit` rotation, `load_best_model_at_end`. `resume_from_checkpoint=True` reads the latest checkpoint — survives power loss, OOM, or manual stop with no repeated work. Validation on Pool B, early-stop patience 3 evals. Output: `artifacts/models/qwen_lora_r1/`, training log, ledger entry `qwen_lora_r1`.

### Phase 7 — Nightly Rater + Feature Integration — **PENDING / TODO (gated on Phase 6)**
Scores new articles through Ollama + the trained LoRA adapter (version-pinned, §11), writes sentiment/vote/confidence + traces + a rating-ledger row for every case. `news_context.py` reuses the existing groupby path with `score_col="qwen_score"`, the same NaN convention (counts = TRUE zeros, means = NaN, availability flags) as every other N6 feature family. Join → frozen-config LightGBM retrain → predictions → existing thresholds/gates.

### Phase 8 — Ablation Gate (pre-registered, §7.4) — **PENDING / TODO (gated on Phase 7; Pool D once, logged)**
Pool D, **exactly once**, logged. +qwen columns vs. FinBERT-only, identical folds. Judged against the recomputed, ledger-aware bar. **Pass:** ship the rater, lock playbook v2, ledger entry — **and open a new, separate economics retest (G-series) before considering any change to the standing PAUSE verdict.** A UP-AUC win alone does not unpause trading. **Fail:** adapters deleted, builder/harness/scanner kept for rung 2, ledger entry written, treated as a complete and honest result, not a setback.

---

## 9. Rung Ladder (unchanged from v1 — correctly designed)

| Rung | Upgrade | Unlock condition |
|---|---|---|
| 1 (this doc) | STaR + LoRA, 8B-Q4, local | — execute now, gated internally by Q0-D1 |
| 2 | Bigger reasoner (Qwen3-32B, cloud) | Rung 1 shows signal worth scaling |
| 3 | Full-parameter finetune | LoRA plateaus below target |
| 4 | Verifier-RL (GRPO on direction reward) | Scanner battle-tested + rungs 1–3 prove reward isn't noise |

Never climb a rung until the one below shows signal.

---

## 10. Governance / Ledger Integration Summary

1. Every pool-C, pool-D read (§5.4) gets a ledger row before its result is treated as final.
2. Before Phase 8's bar is finalized, the cumulative touch ledger is consulted and the Bonferroni correction recomputed against its **current** count — not the figure recorded at G3's time, since the ledger has grown since then (I0, I1, and now Q0-D1/thinking-cap itself add touches).
3. A Phase 8 "pass" is reported together with the full benchmark table (§7.3), never as an isolated number.
4. A Phase 8 "pass" opens a new economics retest; it does not, by itself, change the standing PAUSE verdict.

---

## 11. Edge Cases — covered explicitly

| Edge case | Handling |
|---|---|
| Malformed LLM output (schema violation) | Validator rejects, automatic retry (max 3), then `HARNESS_FAIL` logged — never silently accepted |
| Training crash (OOM, power loss, manual stop) | `resume_from_checkpoint=True` from the latest ~15-minute checkpoint — no repeated work |
| Article exceeds context budget | `TRUNCATED` flag, model reasons over the truncated text, flag carried into the rating ledger for later slicing |
| No news on a given decision_date | `NO_NEWS` token in the context block, `qwen_available=0`, `days_since_qwen` incremented — never filled with a neutral score |
| **Q0-D1 gate fails** | Track stops at Phase 2.5. No STaR generation, no training spent. Harness and rating-ledger code retained for future rung reconsideration. Logged as a clean, complete negative result. |
| **STaR keep-rate anomaly** (UP keep-rate ≈ DOWN keep-rate, both near 50%) | Reported alongside Phase 8's result, explicitly flagged as evidence the kept set may be dominated by lucky guesses — downgrades confidence in a Phase 8 pass even if the headline numbers look favorable |
| Disagreement between FinBERT and Qwen on the same article | Logged as `finbert_qwen_disagree`, a feature in its own right — not algorithmically resolved; LightGBM decides how much to weigh it |
| Thinking budget overflow | Counted as a low-confidence vote, never silently cut |
| **Base Ollama model updates between rung-1 training and later nightly inference** | Model hash and quant version pinned in `artifacts/state/qwen_setup.json`; nightly rater checks the pin before loading the adapter — a version mismatch halts rating for that night rather than silently mixing base-model versions |
| **Multiple trained LoRA checkpoints existing simultaneously** | Exactly one adapter is marked "shipped" at any time; every prediction row in the rating ledger records which `model_stage`/adapter version produced it — same discipline as `model_version` in the existing prediction ledger |
| **Cold-start symbol** (a ticker with few or no STaR training examples) | Identity token is still present in the prompt, but the adapter has little to no symbol-specific grounding for it. Explicitly flagged as a known limitation of the one-global-adapter design (§6.2), not silently papered over — `qwen_conf` is expected to run lower for these symbols, and that's treated as informative, not as a bug |
| **Holdout leakage discovered after the fact** (contingency, hopefully never triggered given §5.2's enforced assertion) | Nuclear option: delete the adapter, re-derive the STaR corpus from a verified-clean Pool A, retrain from scratch. The assertion in §5.2 exists specifically to make this contingency unnecessary, not to provide a safety net for skipping the check |

---

## 12. Files This Track Adds

| File | Phase | Status |
|---|---|---|
| `requirements-llm.txt` | 0 | DONE (split env; `ollama` client added to core venv 2026-10-01 for probing) |
| `artifacts/state/qwen_setup.json` | 0–1 | DONE (phase 0 + phase 1 blocks) |
| `src/stockml/data/qwen_harness.py` | 2 | TODO |
| `src/stockml/data/qwen_playbook_v1.md` | 2 (prompt text + 5 few-shots, versioned — Appendix A is its source) | TODO |
| `tests/test_qwen_guards.py` | 2–4 (holdout-assertion trip test + scanner true/false-positive test) | TODO |
| `data/llm/qwen_rating_ledger.parquet` | 2.5 onward (new) | TODO (`data/llm/` does not exist) |
| `src/stockml/data/star_builder.py` | 4 | TODO |
| `data/llm/star_candidates.jsonl` | 4 | TODO |
| `data/llm/star_kept.jsonl` / `star_dropped.jsonl` | 5 | TODO |
| `checkpoints/qwen_lora/` | 6 | TODO |
| `artifacts/models/qwen_lora_r1/` | 6 | TODO |
| `results/traces/` | 7 | TODO |
| `logs/qwen_*.log` | all | TODO |
| `src/stockml/evaluation/joined_diagnostic.py` | §7.5 (funnel/PSI×accuracy/slices) | TODO (spec §7.5.5 done) |
| `results/qwen_funnel.json/.png` | §7.5.5 graph 1 | TODO |
| `results/psi_accuracy.png` | §7.5.5 graph 2 | TODO |
| `results/cost_waterfall.png` | §7.5.5 graph 3 | TODO |
| `results/coverage_ledger.json` | §14E (byte-matches table) | TODO |

---

## 13. Suggested Execution Order

```
Phase 0 → Phase 1 (incl. Appendix D throughput probe — full vs. sampled STaR decided here)
    → Phase 2 (harness exists)
    → Phase 2.5 (Q0-D1 zero-shot gate — cheapest real checkpoint in the whole track)
         |
    FAIL → stop, infra banked, no further phases run
    PASS → Phase 3 → Phase 4 (assertion-enforced) → Phase 5 (incl. keep-rate check)
         → Phase 6 → Phase 7 → Phase 8 (Pool D, once, logged, bar recomputed against
           the current ledger) → if PASS, open a new economics retest as its own,
           separate decision — do not treat it as resolved by this track alone
```

---

## 14. Executable Appendix — prompt, scanner, training, throughput

### A. Analyst playbook v1 (system prompt skeleton — committed verbatim as `qwen_playbook_v1.md`)

```
You are an NSE equity analyst rating ONE news item for a 5-day directional signal.
You have NEVER seen prices after decision_date. Stating otherwise invalidates the answer.

Checklist (execute in order, cite step numbers in trace):
1. CLASSIFY the event: earnings_result | earnings_guidance | order_win | mna |
   regulatory | rating_change | management | capital_action | debt_credit |
   litigation | expansion | macro_policy | short_seller | noise.
   If noise → vote FLAT, confidence ≤ 0.3, stop.
2. BASE RATE: state the historical precedent for this event type on NSE
   ("similar events typically resolve X within 5 days"). Do NOT invent statistics —
   use qualitative base rates (usually-up / usually-down / mixed) unless the
   context block provides a number.
3. POSITIONING: using ONLY the PRE-T CONTEXT block (5-day trend, volume shock,
   sector move, FinBERT prior), judge whether the move looks priced in.
   Overbought + good news = fade candidate; oversold + bad news = bounce candidate.
4. COMMIT: vote + confidence, citing which steps carried weight.
   Confidence must be proportionate to evidence: single-source rumour ≤ 0.4,
   confirmed filing + supporting positioning ≤ 0.8, never 1.0.

Return STRICT JSON only: {"sentiment": <float -1..1>, "vote": "UP"|"DOWN"|"FLAT",
"confidence": <float 0..1>, "trace_steps": [1,2,3,4]}.
Sentiment sign must agree with vote (positive→UP, negative→DOWN, |sent|<0.2→FLAT).
```

Validator: JSON parse → enum check → `trace_steps == [1,2,3,4]` → sign-agreement check. Any failure → retry (max 3) → `HARNESS_FAIL`. The 5 bootstrap few-shots (one per `earnings_result / regulatory / order_win / management / noise`) live in `qwen_playbook_v1.md`, not here, so prompt edits are diffable commits.

> **Measured correction 2026-10-01 (Phase 1):** the Ollama chat API returns reasoning in `message.thinking`, NOT as `<think>` tags inside `content`. The harness must read the `thinking` field. `num_predict ≥ 1024` is mandatory with `think=True` (256 truncates content to empty). Pinned `num_ctx = 2048` (4096 OOMs/crashes on RTX 3050 6GB + Ollama 0.34.2 + driver 617.14).

### B. Leakage-scanner pattern list (v1 — extend, never shrink, without a commit)

| Class | Patterns (case-insensitive regex) |
|---|---|
| Post-decision dates | `\b(20\d\d)-(0[1-9]\|1[0-2])-(0[1-9]\|[12]\d\|3[01])\b` compared against `decision_date` (any date > decision_date = hit); `next week|following week|five days later|after the event` + a directional claim |
| Outcome mentions | `triple.?barrier|upper barrier (was )?hit|lower barrier (was )?hit|touched the ceiling|touched the floor|label (is\|was) (UP\|DOWN\|FLAT)` |
| Future prices/returns | `(rose\|fell\|gained\|lost\|closed at|settled at)\s+\d+(\.\d+)?\s*(%\|percent\|rs\|inr)` when the sentence refers to post-decision_date action; `5-day return|realized return|event_end` |
| Self-incrimination | `i have seen (future\|post\|subsequent)|as an ai.*knowledge cutoff|based on what happened (after\|next)` |

Unit test (`tests/test_qwen_guards.py`): one synthetic trace per class must trip the scanner; one clean trace with a same-day date (`decision_date` itself) must NOT trip. Scanner version is logged per rating-ledger row.

### C. QLoRA training hparams (6GB-fit baseline — change only with a commit)

| Param | Value | Why |
|---|---|---|
| Base load | 4-bit NF4, double-quant, `dtype=bfloat16` compute via unsloth | Only config fitting 8B + adapter + 4k ctx on 6GB |
| LoRA | rank 16, alpha 32, dropout 0.05, targets `q,k,v,o,gate,up,down` | ~5% trainable; `mlp` included because tone→reasoning shift lives there |
| LR / sched | `2e-4`, cosine, warmup 3% of steps | Unsloth QLoRA standard for 8B; lower LR underfits STaR's short traces |
| Batch | micro-batch 1 (2 only if OOM-free on probe), grad-accum to effective 16 | Effective 16 balances gradient noise vs. 6GB ceiling |
| Seq len | 4096, fallback 3072 (truncate + `TRUNCATED` flag) if the pre-probe OOMs | Prompt worst-case ~2700 tokens (§6.2) — 2048 would silently cut reasoning |
| Seq len — **SUPERSEDED 2026-10-01** | **Pinned 2048** for all Ollama inference (Phase 1: 4096 crashes — stack overrun, PTX JIT fail, 1.57GB alloc fail; 2048 stable ×6 calls) | Training-side 4096 fit still UNPROVEN — 10-step dummy pre-probe is TODO before Phase 4 |
| Optim / precision | 8-bit AdamW, grad-checkpointing ON, seed 42 | Checkpointing is non-negotiable on 6GB |
| Eval / stop | eval on Pool B every 10% of expected steps, early-stop patience 3 evals, `load_best_model_at_end` | Matches §8 Phase 6 |
| Pre-probe (mandatory) | 10-step dummy run at seq-len 4096 before Phase 4 generation is approved | Proves the fit; failure → seq-len 3072 re-probe, then rank-8 fallback, each logged |

### D. Throughput probe (Phase 1 exit criterion)

1. Rate 20 Pool C articles through the pinned `qwen3:8b` Ollama endpoint (thinking budget = Phase 3 winner or 512 default), record median `tok/s` and `s/article`.
2. Extrapolate: `hours = s/article × N / 3600` for `N = Pool A size`. Commit the numbers to `artifacts/state/qwen_setup.json`.
3. Gate: projected build `≤ 14 days` single-server → full Pool A STaR approved. Above that → sampled STaR (announcements-only, `relevance ≥ 0.5` first, remainder deferred to rung 2) — the sampling rule is committed before Phase 4 starts, never improvised mid-generation.
4. Concurrency is capped at 2 and incremental saves are mandatory (`star_candidates.jsonl` append per case) regardless of the outcome — power loss must never lose more than in-flight cases.

> **Phase 1 outcome 2026-10-01:** median 20.3s/article `@ctx2048` → full Pool A `≈524h (21.8d)` → gate FAILED → sampled STaR committed (see Phase 1 + `qwen_setup.json: phase1`).

---

> **Phase 1 outcome 2026-10-01:** median 20.3s/article `@ctx2048` → full Pool A `≈524h (21.8d)` → gate FAILED → sampled STaR committed (see Phase 1 + `qwen_setup.json: phase1`).

---

### E. Coverage ledger (spec DONE 2026-10-02, implementation TODO)

Status: **SPEC DONE** — table below is the contract for `python -m stockml.data.coverage_audit --out results/coverage_ledger.json` (TODO). Re-count command output must byte-match this table (modulo run date) or the table is stale and must be refreshed with a commit, never hand-edited.

Denominator rule (pre-registered): `92,960` = raw ingested rows (`data/raw/announcements/*.parquet`, 50 symbols); `92,873` = dated rows (`decision_date` non-null via `features/decision_date.py`); the 87-row gap is undated/dropped and named every time. Availability-flag rule (B-07): counts are TRUE zeros, means/z-scores/shocks stay NaN (LightGBM native missing) — any `0.0`-fill of a mean (as `news_context.py:21-26` `_empty_like` does today for empty symbols) is a defect and fails this ledger closed.

| Source | Raw N | Dated N | Scored N (`finbert_score` non-null) | Post-2018 avail% | Flag definition | Provenance |
|---|---|---|---|---|---|---|
| NSE announcements (company) | 92,960 (50 files + `_cursors/`) | 92,873 (`u2_d1.json:2`) | 26,325 (`u3_d1.json:3-4`; only `≥2023-08-29`) | `ann_co_available` 96.6% over 184,415 joined rows | `ann_co_*` (9 cols: count/5d/20d/`days_since`/4 event flags/`available`) | `data/announcements.py` (`run_ann_scoring(since=2023-08-29)`), `announcement_features.py:1-93`, `ann_score.json (scored 26325 / skipped 66635)` |
| GDELT DOC text (company) | 9,179 (38/50 files; 12 symbols + index empty) | same (90d window `2026-06-23..09-20`) | 9,179 (all scored) | `news_co_available` 0.52% | `news_co_*` (8 cols, `aggregate_scored_text`, `news_context.py:29-72`) | `data/news.py` (`collect_news_for_symbol`, `maxrecords=250`, `gdelt_days=90`), `news_fetch.py:116-130` |
| GDELT DOC text (market) | market file | 57d only | all scored | `news_mkt` 1.44% (max 799/day) | `news_mkt_*` | `market_features.parquet` (3951 rows) |
| Market announcements | market file | full | n/a (counts only) | `ann_mkt_count>0` 99.2% (max 9049 on 2010-09-15 backfill spike) | `ann_mkt_*` | same market file |
| GDELT-BQ events (tone) | 88,798 (11 files; 39 symbols zero; 4 symbols = 76%) | `1920..2026` incl. pre-1979 `SQLDATE` garbage | 100% `tone` (GDELT-native, never FinBERT) | `ev_co_available` 16.06% (0% on 39 symbols) | `ev_co_*` (4 cols) | `data/gdelt_bq.py` (`match_symbol` substring over-match — fix TODO), `gdelt_bq.json (11.7M scanned, 164GB)` |
| RSS | 103 (`ET 50 / Mint 35 / MC 18`) | stale `2016-10-05` cache | none | 0% (never merged — `symbol=''` filtered) | — | `data/rss.py`, `rss_2026-09-20.parquet` — BROKEN/TODO |
| AlphaVantage | 0 | — | — | 0% | — | `news.py:58` silent `try/except: pass`, `av_quota.json 25/25` spent — DEAD/TODO |

Worked example (post-2018 RELIANCE): announcements 61%, events 81%, GDELT text 0.1% (`MASTER_PLAN:1089` N6 BUILD) — B-12's ">60% met" holds only via announcements/events; text news is a 90-day sliver. Sentiment split where scored: `neutral 4460 / neg 2477 / pos 2242` (`u3_d1.json: rows_with_sentiment 15216 = 743 pos + 1964 neg + 12509 neutral`, `none 22076`). Reference gap blocking fan-out: `data/reference/` has `group_map.json`, `sector_map.json`, `trading_calendar.parquet` — **no `entity_map.json`** (B-06/F-05 TODO).

### F. Metrics catalog (spec DONE 2026-10-02, values live)

Status: **SPEC DONE** — every §7.3/§7.4 number traces to one code line + one artifact key. Recompute, never copy: the Bonferroni threshold below is a snapshot (`0.05/21≈0.0024`), the Phase 8 bar is recomputed against the then-current ledger.

| Metric | Definition / formula | Code (`file:line`) | Artifact (`file:key`) | N / threshold |
|---|---|---|---|---|
| Accuracy / bal-acc / MCC / macro P-R-F1 / confusion | `sklearn` standard on 3-class | `evaluation/metrics.py:17-27` (`classification_metrics`) | `metrics.json: accuracy 0.4713, bal 0.3803, mcc 0.037, f1_macro 0.3517` | `n_test 37292`, `test_start 2023-08-29` |
| OvR AUC macro / per-class | one-vs-rest binarized `roc_auc_score` | `metrics.py:48-58` | `metrics.json: ovr_macro 0.5543`; `up_autopsy.json: up 0.5140 / down 0.5261 / flat 0.6234` | ranking skill; UP ceiling 0.5127–0.5181 |
| Directional acc (nonflat / takes-direction) + coverage | `accuracy_score` on `actual∈{0,2}` / `actual&pred∈{0,2}` | `metrics.py:28-46`, `evaluate.py:50-57` | `metrics.json: 0.4938 (n=34689) / 0.5132 (coverage 0.9529, n=33382)` | exact 0-vs-2 match |
| Majority / logistic baselines, deltas | `max(class_counts)/n`, `LogisticRegression` anchor | `evaluate.py:42-47`, `models/baseline.py` | `majority 0.4785 (beats:false, Δ−0.0071)`, `logistic 0.4802 (Δ−0.0088)` | only honest accuracy comparison |
| ECE + bins / reliability | `Σ\|acc−conf\|·count/n`, 10 bins on max-prob | `metrics.py:70-99`, `evaluate.py:47-54` | `metrics.json: ece 0.04244 + ece_bins` (top-bin gap 0.27–0.40) | campaign bar `ECE < 0.05` |
| DM (UP-vs-rest, variants) | Newey-West `lags=5` on loss differential | `evaluation/dm.py` | `metrics.json: dm_up stat −4.82 p 1.3e-06`; `economics.json: dm_vs_V0` | Phase 8: recomputed Bonferroni (snapshot `0.05/21≈0.0024`, 22 rows/21 elig/19 unique) |
| Flat rule | `FLAT iff p_flat ≥ tau` else argmax(DOWN,UP) | `evaluation/flat_threshold.py:37-65` | `flat_threshold.json: tau 0.40, acc 0.4661→0.4713 (+0.52pp), DM 9.14 p 6e-20` | zero-retrain lift |
| Economics + PBO | event backtest, Indian costs, CPCV-PBO | `backtest/backtest.py:34-127`, `evaluation/economics.py` | `backtest_metrics.json` + `economics.json` (`V0 sharpe −2.48`, `thr70 −0.84`, `fo_proxy −0.28`, `_pbo 0.0`; 31.5bps delivery / 15bps futures) | G3 PAUSE holds; pass earns retest only |
| Drift PSI | quantile PSI, 10 buckets | `monitoring/drift.py:19`, `monitor.py:63-98` | `monitoring.json` (`atr 4.27`, `gold_vol 10.49`, alerts = PSI≥0.25 ∩ top-20 SHAP) | threshold 0.25; levels/calendars excluded |
| Governance | every sealed read logged | `evaluation/touch_ledger.py:17-87` (`N_TEST=37292`) | `holdout_touches.jsonl` (22 rows) | count feeds PBO |
| Training context | fingerprint manifest | `models/lightgbm_train.py` | `model_metadata.json:208` (`train 0.6394 / test 0.4662, gap 17.3pp`) | `best_params_hash + dataset_signature` |

---

## 15. Execution Log - what was DONE vs what is TODO (updated 2026-10-20)

### DONE (with evidence — do not redo)

1. **Phase 0 env** — Ollama 0.34.2 + `qwen3:8b` present, WSL `torch 2.12.1+cu130 / unsloth OK`. Evidence: `qwen_setup.json` (`status: complete` → now `phase1-complete-sampled`).
2. **Phase 1 pull + health (corrected)** — `ollama pull qwen3:8b` success (hash `500a1f067a9f`, 5.2GB); health PASS with `num_predict=1024` (`think_len 1163`, valid JSON `UP`, 21.9s). First 20-case run scored `0/20` only because the probe parsed `<think>` tags instead of the `message.thinking` field — parser fixed, finding recorded above.
3. **Throughput probe + gate decision** — `@ctx2048` 20.3s/article; `@ctx4096` 43.8s + 6 server crashes. Full-build gate FAILED → sampled STaR. Evidence: `qwen_setup.json: phase1`, temp scripts in `Temp/opencode` (`qwen_phase1_probe.py`, `qwen_phase1b.py`).
4. **Doc blockers fixed** — §5.0 (`HOLDOUT_START 2023-08-29`), Pool C `n=1500` / cap `n=1000` + power math, §14A–D appendix, ledger snapshot (`22 rows / 21 eligible / 19 unique → 0.05/21≈0.0024`). Evidence: this file §§5, 7.4, 8, 14.
5. **News audit (diagnostic only)** — company news `0.52%` / market `1.44%` / ann `96.6%`; AV 0 rows; RSS stale/unwired; BQ 11 symbols skewed; empty-symbol `0.0`-vs-NaN bug confirmed. No data files were changed.
6. **Observability spec (2026-10-02)** — §7.5.1–7.5.6 (SHAP rank, ECE slice, cost slice, trace-quality, blame taxonomy + 3 funnel graphs, question bank), §14E coverage ledger, §14F metrics catalog drafted with file:line provenance and pre-registered bars. Implementation (diagnostic script, coverage-audit command, news-flag carry-through) still TODO.
7. **Phase 2 built + A1 scored (2026-10-02/03)** — `qwen_harness.py` (skeleton prompt, `message.thinking` reader, 1024 floor, 120→300s timeout, `PLAYBOOK_VERSION=v1-skeleton`), `qwen_playbook_v1.md`, `tests/test_qwen_guards.py` (15/15 green on 2026-10-20), NaN-bug fix (`news_context.py` means/shocks/z→NaN), 92,959/92,960 announcements FinBERT-scored (M&M empty file only). Q0-D1 runner (`q0_d1.py`, resumable ledger + traces) with Pool C 1500 (500/250/750) built; pilot n=26: Qwen 30.8% vs FinBERT 19.2% (Δ+11.5pp, inconclusive — 2×SE=23.8pp at n=26).
8. **Thermal guard v1 (2026-10-03, user-requested)** — fixed 15-min rest at ≥85 °C, checked at each 5-case checkpoint AFTER the ledger flush (batch always completes first; no polling loop). Lives in `q0_d1.py` (`gpu_temp_c` / `thermal_rest_if_hot`, rests counted in `q0_d1.json: thermal_rests`). **Superseded by item 9** — kept here for history only.
9. **Thermal guard v2 + overnight supervisor (2026-10-20, user-specified)** — replaces v1 with: pre-case check; 80 °C watch (30 s breather only if rising), 84 °C prepare, 85–86 °C alarm (continue), **87 °C hard stop**; every rest starts with `ollama stop qwen3:8b`; **fixed 20-min rest**, no 60 °C hold; **4 hard stops in a rolling 60 min → 30-min cooldown + full counter/timer reset**, then resume (no thermal fault, no force quit); **15-min unloaded proactive break every 60 min**; **5-second monitor** during the case that aborts the in-flight case at 87 °C and discards it (re-rated later, never a partial vote); per-case parquet checkpoint so a stop loses nothing; `gpu_temp_start`/`gpu_temp_end` per ledger row; 5-second telemetry in `logs/gpu_5s.csv`. `run_q0_overnight.ps1` self-elevates, sets `nvidia-smi -lgc 0,1400` (falls back to the nearest supported clock ≤1400; this laptop's table lists 1395, not 1400), runs 24/7 (no night window), kills at 89 °C for two 5-s readings, rests 30 min, resets counters, restarts, and resets `nvidia-smi -rgc` in `finally`. Smoke test: `--pilot 1` completed, max 72 °C, ledger 111/1500, 15/15 guard tests green. Force-close caveat: if the supervisor window is killed, run `nvidia-smi -rgc` manually.
10. **Zero-touch Option A (2026-10-20, code only - not started by the agent)** - per user request, the supervisor was extended and a one-time setup script added, but nothing was registered or restarted; the user runs one elevated setup when ready. `run_q0_overnight.ps1` now has a single-instance `Global\StockML_Q0_Supervisor` mutex (with `Local\` fallback), stale-supervisor/stale-worker cleanup (matched by command line, never hardcoded PIDs), an Ollama `/api/tags` health check with `ollama serve` restart, exponential crash backoff (60s/120s/5m/10m/15m/30m, never gives up on code crashes), an atomic supervisor heartbeat `logs/q0_supervisor.json`, `logs/q0_supervisor_events.log`, `logs/q0_ALERT.txt`, `logs/q0_DONE.txt`, power settings (no AC sleep/hibernate, lid close does nothing), and `-Scheduled`/`-DryRun` modes. `q0_d1.py` now charges the 15-min-per-60-min proactive break against actual GPU seconds (`result.seconds`) instead of wall clock and writes an atomic `logs/q0_heartbeat.json` at start, before each case, after every checkpoint and thermal abort, and at the verdict. `setup_q0_task.ps1` self-elevates once, registers `StockML_Q0_Supervisor` (Interactive/RunLevel Highest, AtLogOn of RONIL\Ronny, `MultipleInstances IgnoreNew`, no execution time limit, restart on failure every 1 min), stops the current manual supervisor and stale workers, starts the task, waits up to 120s for a fresh heartbeat, and falls back to a manual elevated supervisor if verification fails. `q0_status.ps1` is a read-only status view (task state, heartbeats, ledger count, temperature/clocks, markers). `tests/test_qwen_guards.py` is 18/18 (added work-clock, pool-stats, and heartbeat tests); the full suite is 70 passed. On Q0 completion the supervisor stops, resets clocks, and writes the done marker - it does not auto-start Phase 3.

### Execution log (2026-10-26 — code only, no phase runs)

11. **Track Q phases 3–8 code written (untested)**: the full CPU-side and GPU-side phase modules plus the gate-enforcing orchestrator were added while Q0 was still running. New files: `src/stockml/qpipeline/{__init__,contracts,state,runner,__main__}.py`, `src/stockml/data/qwen_rater.py`, `src/stockml/data/leakage_scanner.py`, `src/stockml/data/star_builder.py`, `src/stockml/data/qwen_nightly.py`, `src/stockml/data/qwen_lora_harness.py`, `src/stockml/features/qwen_features.py`, `src/stockml/evaluation/q3_thinking_cap.py`, `src/stockml/evaluation/q8_ablation.py`, `src/stockml/training/{__init__,qwen_qlora,qwen_export}.py`, `scripts/qwen_10step_probe.py`, `config/q_track.yaml`, `run_q_pipeline.ps1`, `q_pipeline_status.ps1`. All phase statuses remain PENDING until actually run. Syntax checks only (`py_compile` + PowerShell parser) — no GPU, Ollama, WSL, or phase execution.

### TODO (explicitly gated — do not start early)

- [ ] **Q0-D1 full run (IN PROGRESS — 122/1500 as of 2026-10-20 01:54)**: currently on the temporary manual supervisor; run `setup_q0_task.ps1` **once** (single UAC) to hand over to the zero-touch scheduled task, then monitor with `q0_status.ps1` (read-only). The 15-min break timer counts actual GPU seconds. Per-case ledger means a thermal stop loses nothing. On completion read `results/q0_d1.json`: `pass:true` → Phase 3 thinking-cap; `pass:false` → stop, write the negative result, and do not start Phase 3/4.
- [ ] **A1 follow-up (optional)**: `M&M.NS.parquet` is empty; decide whether to backfill it. The other 92,959/92,960 announcements are already scored.
- [ ] **Phase 3** (`n=1000` paired, gated on Q0 PASS) → **Phase 6 probe** (10-step, before Phase 4) → **Phase 4** sampled STaR (gated on Phase 3 + probe) → **5/6/7/8** in order.
- [ ] **Track Q code written (2026-10-26, untested)**: Phase 3–8 modules + orchestrator + wrappers are in the repo and syntax-checked. No phase has been executed. Phase statuses remain PENDING until the real run passes.
- [ ] **Observability implementation**: `joined_diagnostic.py` (funnel/PSI×accuracy/slices), coverage-audit command (byte-match §14E), news-flag carry-through (`backtest.py` + `economics.py` mask), ledger `quality_score`/`audit_verdict` columns, worst-20 SHAP+text upgrade, sliced 3×3 confusion.
- [ ] **Training-side items**: HF weights staging (disk check blocked), 10-step 4096/2048 pre-probe, WSL fit proof.
- [ ] **Rungs 2–4**: gated on rung-1 signal; never started speculatively.
