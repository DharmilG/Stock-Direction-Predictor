# Track I — New Information Sources

**Status:** spec + build only. No training until G3 (economics thread) reaches a verdict.
**Governing rule, inherited from every prior track:** cheap read-only diagnosis before any retrain. One hypothesis in training at a time. Kill on the bar, not on vibes.
**Why this track exists:** U2-D1 (presence) and U3-D1 (FinBERT content) proved company-scope announcements carry no marginal information for 5-day UP at 1σ. That is a narrow, well-earned null — specific to those two information types, that horizon, and that threshold. It is not evidence that *no* untested information exists. Track I is the bounded, disciplined search for information that is mechanistically different from what has already failed.

---

## 0. Governance — read this before running anything

### 0.1 The cumulative test-touch ledger (new requirement, applies retroactively)

Every prior diagnostic (D1, U2-D1, U3-D1) joined candidate features against **already-computed predictions on the sealed test window** to compute slice statistics. That is a touch of the test set, even with zero training involved — it's a lighter touch than a full retrain-and-evaluate cycle, but it is not a zero touch, and the campaign has done this eight-plus times now.

**Before any Track I hypothesis's diagnostic result is treated as a green light to train:**

1. Create (or extend, if it exists) `checkpoints/test_touch_ledger.jsonl` — one line per event that reads the sealed test window's true labels in any form, whether via training, evaluation, or a read-only slice join. Fields: `date, track, phase (D1|ablation), description, rows_touched, artifact_ref`.
2. Backfill it now with every touch this campaign has already made (G1-attempt-1, G1-attempt-2, H1, H1b, U, U2-D1, U3-D1, F) — the reports already have the numbers needed to reconstruct this.
3. Before treating any Track I ablation's "PASS" as final, recompute PBO using the **full ledger count**, not just Track I's own touches. If PBO comes back elevated versus the last computed value, the bar for that ablation's pass criterion gets stricter (require the effect to clear a wider margin, not just cross the current threshold) rather than being accepted at face value.

This is bookkeeping, not a blocker — do it once, keep it current, and every future track benefits from it too.

### 0.2 Pre-registered effect-size bar for every D1 (do this before looking at results, not after)

The reason U3-D1's null was decisive rather than debatable is that it had an unambiguous quantitative bar: base-rate delta near zero, full spread of prior UP-AUC attempts inside ~2 Hanley-McNeil SE. Every Track I hypothesis below gets the same treatment — **the pass bar is written before the query runs**, not chosen after seeing the number. Where a hypothesis's D1 section below doesn't give you a hard number, compute the sample-size-appropriate SE first and set the bar at ≥2 SE before executing.

### 0.3 Sequencing

Only **one** hypothesis proceeds past its D1 gate into feature-build-and-ablation at a time. Cheapest and least-tested-adjacent first. Order below is deliberate — not alphabetical, not "most exciting first."

---

## 1. Ranked hypotheses

### I0 — Label redefinition: horizon × threshold sweep (do this first, zero new data)

**What it is.** Everything tested so far (H1, H1b, U, U2, U3) held the label fixed: 5-day horizon, 1σ vol-scaled threshold. Every negative result is a statement about *that specific target*, not about UP direction in general. This is the cheapest possible lever — no new ingestion, just relabeling data already on disk.

**Why it's mechanistically different from what already failed.** U2-D1 and U3-D1 proved *these features* don't predict *this label*. They say nothing about whether a different horizon (e.g. 10–20 days, where post-earnings drift and slower institutional accumulation plays out) or a different threshold (e.g. 0.5σ, producing more UP-labeled rows with less noise per row) would be predictable from the *same* features already in hand.

**D1 (read-only, no new data, no retrain):**
1. Re-derive labels at a small grid — horizon ∈ {1, 5, 10, 20} days × threshold ∈ {0.5σ, 1σ, 1.5σ} — using the existing triple-barrier code in dry-run mode (label generation only, no training).
2. For each grid cell, compute the same UP OvR AUC the current model *would* need to beat, using the **existing frozen model's predictions is not valid here** (different label = different target; skip straight to reporting class balance and label-noise proxies): realized-return spread per class (mirror-image check, same as the original D1), and row count per class.
3. Separately, slice UP-miss rate (using the *existing* 5-day/1σ model and its *existing* predictions) by "days since last earnings result" (bucketed 0-5, 6-10, 11-20, 21+). This tests the PEAD hypothesis directly against current predictions — zero new label generation needed for this part.

**Pre-registered bar:** the earnings-proximity slice must show a UP-miss delta between the 0-5-day bucket and the 21+ bucket exceeding 2×SE at that bucket's sample size (compute before running, same as §0.2). If the horizon/threshold grid shows a cell with dramatically better class balance *and* a materially different realized-return separation between classes, that cell becomes the candidate for a fresh triple-barrier-labeled ablation.

**Log to ledger:** yes — reads test-window realized outcomes for the earnings-proximity slice.

---

### I1 — Cross-sectional lead-lag (zero new data)

**What it is.** Does a sector or group leader's move on day *t−1* predict a laggard's move on day *t*? This is temporally different from the sector-relative features already built (same-day comparison against the sector average) — this is *yesterday's leader vs. today's laggard*, a genuinely untested relationship, and it requires no data beyond what's already ingested for the full universe.

**Why it's different from what already failed.** Sector-relative momentum (already in the 136-feature set, SHAP rank 43-110, "nearly unused" per the last report) measures *where a stock sits relative to its sector today*. Lead-lag measures *whether information travels from one stock to another with a one-day delay* — a distinct mechanism (information diffusion across correlated names) rather than a same-day relative-strength snapshot.

**D1 (read-only):** for each sector, identify the top-quintile-by-market-cap "leader" and bottom-quintile "laggards." Compute: does leader's day-*t−1* return sign correlate with laggard's day-*t* miss rate on the existing frozen model? Slice UP-miss specifically on (leader moved up strongly on *t−1*) vs. (leader flat/down on *t−1*).

**Pre-registered bar:** UP-miss delta between the two conditions must exceed 2×SE at the relevant sample size.

**Log to ledger:** yes.

---

### I2 — Bulk/block deal disclosures (new data, per-stock, free — highest priority new-data source)

**What it is.** NSE publishes daily bulk and block deal reports: large single trades tagged to the specific stock, with buyer/seller category where disclosed. This is a genuinely different information channel from anything tested — it's not sentiment about a stock, it's a record that unusually large capital actually moved into or out of it.

**Why it's different from what already failed.** U2-D1 tested announcement *presence* (did any announcement happen) and U3-D1 tested announcement *sentiment* (was the announcement positive/negative in tone). Neither tests "did a large, discrete trade happen." A bulk deal is closer to *revealed conviction* than either presence or sentiment — someone put real capital behind a view, as opposed to a company or journalist saying something.

**Data source:** NSE bulk/block deal daily reports (free, official, per-stock, exchange-mandated disclosure — same tier of confidence as the NSE/BSE announcements already ingested for Track U).

**Build required before D1 is possible:** unlike I0/I1, this needs an ingestion pass (per-symbol, resumable, same cursor pattern as `news_fetch.py`) before any diagnostic can run — there's no way to test a hypothesis against data that doesn't exist yet. Scope this ingestion narrowly: bulk/block deal date, symbol, buy/sell side, quantity, price — no NLP, no entity resolution ambiguity (same reason NSE/BSE announcements were high-confidence: pre-tagged scrip codes).

**D1 (read-only, after ingestion):** slice UP-miss rate on days with a same-day-or-prior-day bulk/block buy-side deal vs. days without, using the existing frozen model's predictions. Separately check: does the deal's direction (buy vs. sell side) correlate with the realized label's direction at all, independent of the model?

**Pre-registered bar:** UP-miss delta on buy-side-bulk-deal days vs. no-deal days must exceed 2×SE. Additionally, since this is the first "revealed conviction" test, also report raw `P(UP | buy-side bulk deal)` vs. `P(UP | no deal)` the same way U3-D1 reported the sentiment base rate — that comparison alone may be as decisive as it was for sentiment, in either direction.

**Log to ledger:** yes.

---

### I3 — FII/DII daily net flow (new data, market-wide, free — lower priority, likely weak)

**What it is.** NSE publishes daily aggregate FII (foreign) and DII (domestic institutional) net buy/sell figures for the whole market, confirmed free at `nseindia.com/api/fiidiiTradeData`, with historical archives via NSE/NSDL/CDSL.

**Why it's ranked below I2, honestly.** This is market-wide, not per-stock — the same *shape* of signal as the "market-high announcement" slice in U2-D1, which came back at coin-flip, directly contradicting the theory that market-wide information should be the strongest signal. There's no strong reason to expect a different outcome from a different market-wide series. Included because it's genuinely cheap to check and genuinely different in *content* (institutional flow, not news) even if similar in *scope* (market-wide) to something that already failed.

**D1 (read-only, after a lightweight ingestion — daily figures only, no per-symbol work):** slice UP-miss rate on strong-FII-inflow days vs. strong-outflow days vs. neutral days, using the existing frozen model's predictions, market-wide (same mechanism as the U2-D1 "market-high" slice).

**Pre-registered bar:** same 2×SE standard. Given the prior market-scope null, treat this as a fast, low-cost confirmation rather than a promising lead — one afternoon, not a sprint.

**Log to ledger:** yes.

---

### I4 — Options market data: put-call ratio and IV skew (new data, most complex ingestion, most differentiated information)

**What it is.** NSE F&O option-chain data — put-call ratio and implied-volatility skew per underlying. Options prices are forward-looking by construction: traders are pricing in expected future moves, not reacting to past ones. This is the most mechanistically different source on this list from anything already tested — everything else so far (price/volume, announcements, sentiment) is backward- or present-looking.

**Why it's ranked last.** Ingestion is genuinely more complex than the others (option-chain data has more structure, more symbols-within-a-symbol via strikes/expiries, and needs care to avoid point-in-time leaks around expiry rollovers). Given the standing discipline of cheapest-and-least-tested-adjacent first, this is the right one to hold until I0-I3 have been run and either yielded a working ablation or been exhausted.

**D1 (read-only, after ingestion of daily aggregate put-call ratio and near-the-money IV skew per underlying, F&O-eligible names only — this naturally restricts scope to the ~180-200 liquid names, which is also where any resulting signal would actually be tradeable):** slice UP-miss rate by put-call-ratio regime (bullish/neutral/bearish per standard thresholds) and by IV-skew direction, using the existing frozen model's predictions.

**Pre-registered bar:** same 2×SE standard.

**Log to ledger:** yes.

---

## 2. Feature build + ablation — only for a hypothesis that clears its D1 bar

If (and only if) a hypothesis's D1 result clears its pre-registered bar, it proceeds to a single, targeted feature (not a family of ten) and one training ablation, judged against the standing criteria used for every prior track:

| Check | Bar |
|---|---|
| UP OvR AUC | must move up from the current baseline by a real margin, not just cross it |
| DM test, UP-vs-rest | significant improvement, not just headline movement |
| Standing guardrails | AUC ≥0.55 overall, FLAT F1 not degraded, ECE <0.08 |
| Memorization tripwire | train/test gap ≤20pp, non-negotiable, no exceptions for "more fundamental" data sources |
| NaN/point-in-time discipline | new feature gets its own unit test for missing-data and cutoff-timing edge cases, same as `delivery_clv`'s test coverage in H1b |
| PBO | recomputed against the full ledger (§0.1) before the verdict is called final |

**Kill rule, same as every prior track:** if it doesn't clear the bar, it's recorded as a real, honest failure — not retried with a variant on the same session, not quietly dropped without a number attached.

---

## 3. What Track I does *not* do

- Does not touch DOWN or FLAT — those are closed (DOWN modestly working, FLAT fixed via threshold) and out of scope here
- Does not run in training mode until G3 concludes — everything above I0/I1's D1 steps and I2/I3/I4's ingestion steps can proceed now; actual ablation training waits
- Does not test more than one hypothesis in training at a time, regardless of how many D1s clear their bar simultaneously — queue them, don't parallelize the expensive step
- Does not treat a cleared D1 bar as a guaranteed ablation win — I0-I4's D1 checks are necessary, not sufficient; U2-D1 and U3-D1's clean nulls are the reminder of how often a plausible-sounding hypothesis turns out to carry nothing

---

## 4. Suggested order

```
Now, in parallel with G3 (no training, no confound with G3):
  I0-D1   — label grid + earnings-proximity slice     (zero new data, fastest)
  I1-D1   — lead-lag slice                              (zero new data)
  I2 build — bulk/block deal ingestion                  (new data, start now, slow)

After I0/I1 D1 results are in, still parallel with G3:
  I2-D1   — bulk/block deal slice (once ingested)
  I3 build + D1 — FII/DII                                (cheap, low priority)

Held until G3 concludes AND at least one D1 above clears its bar:
  First cleared hypothesis → single feature → one ablation, judged per §2

Held until the above resolves, or explicitly deprioritized:
  I4 — options data (most complex, most differentiated, last)
```

Report back per hypothesis in the same format the campaign has used throughout: numbers first, verdict stated plainly, ledger updated, tests green count, no framing that makes a null read as a maybe.
