# Model observability — the Behaviour Plate and live training telemetry

**Purpose:** answer *why this model behaves the way it does* — not only "why DOWN".
The unit of work is a **Behaviour Plate**: one card, one legend, five panels,
computed once per split from stored probabilities. The same code describes a
model that collapsed onto DOWN, one tilted toward UP, and one that is balanced
but uninformative. Nothing in the page assumes the answer in advance.

Everything here is **read-only with respect to the model**. No route, no CLI
step on this path retrains, tunes, rescales or changes `tau_flat`.

---

## 1. Where it lives

| Layer | File |
| --- | --- |
| Analysis (FastAPI-free) | `src/stockml/api/model_diagnostics.py` |
| Report stage / CLI | `src/stockml/evaluation/model_report.py` → `python -m stockml.main model-report` |
| Serving explanations | `src/stockml/inference/explain.py` → `python -m stockml.main explain` |
| API router | `src/stockml/api/dashboard_model.py` (`/dashboard/model/*`, `/dashboard/explain/{symbol}`) |
| Page | `stock-direction-dashboard/src/pages/ModelDiagnostics.jsx` (`#/model`) |

`model-report` is also part of `refresh` (after `monitor`, before `predict`,
`optional=True`, so a reporting failure never aborts a data refresh).

---

## 2. The Behaviour Plate

One bordered card, one legend, five panels, three panels' worth of payload per
split (`train`, `calibration`, `test`), rendered with identical axes so that
overfitting is *visible* rather than asserted.

| # | Panel | Chart | Question it answers |
| --- | --- | --- | --- |
| A | Distribution | paired bars: actual share vs predicted share per class, plus bias chips (±pp) | Is it balanced? What is over-produced? |
| B | Confusion | 3×3, row-normalised, raw counts in cell | What is it confusing with what? |
| C | Profile | 9 grouped bars: mean p(c) for each actual class | Does conviction track truth? |
| D | Conviction | stacked histogram of `max(p)` over `0 · 0.34 · 0.45 · 0.55 · 0.65 · 1` bins, split by predicted class | Is a call a decision or a plurality? |
| E | Drivers | horizontal bars: mean **signed** SHAP of the top features that push *toward* a class | Why that class, specifically? |

Under the plate: the generated `headline`, the `evidence[]` rows that produced
it (label / value / threshold), and chips for `starved:*` and `over_produced:*`.

The payload also carries a `margins` block (histogram of `max(p) − second(p)`),
which is what makes "plurality, not decision" checkable, and a `rolling[]` series
(30 sessions) of predicted shares and accuracy over time.

### `separation`

```text
separation = mean over class pairs (a, b) of ½ · Σ_c |p̄_a(c) − p̄_b(c)|
```

where `p̄_a` is the mean probability vector of the rows whose *actual* label is
`a` (panel C). Range 0–1. It is a single scalar that distinguishes "collapsed
prior" from "real tilt", and it is computed identically whatever the dominant
class is. `separation < 0.05` means the three classes are described by nearly the
same probability vector.

---

## 3. Mode taxonomy (global — never DOWN-specific)

| mode | condition | generated headline |
| --- | --- | --- |
| `collapsed_prior` | max bias > 15 pp **and** `separation < 0.05` | "Output collapsed onto **{C}**: produced {p}% of calls vs {a}% in the data, while the class profiles differ by only {s}." |
| `tilted` | max bias > 10 pp **and** `separation ≥ 0.05` | "Genuine tilt toward **{C}** (+{b} pp of output) with measurable class separation {s}." |
| `balanced_undiscriminating` | all \|bias\| ≤ 5 pp **and** accuracy ≤ prior + 2 pp | "Balanced but uninformative: produced shares match the data, yet accuracy {acc}% is level with the {pri}% prior." |
| `balanced_discriminating` | all \|bias\| ≤ 5 pp **and** accuracy > prior + 2 pp | "Balanced and informative: shares match the data and accuracy {acc}% beats the {pri}% prior by {edge} pp." |
| `mild_bias` | anything else | "Mild {C} bias (+{b} pp), separation {s}: between a real tilt and a collapsed prior." |
| `unknown` | no scored rows | "No scored rows." |

Flags, appended independently of the mode:

- `starved:{C}` — predicted share < 50% of actual share (and actual share > 2%).
- `over_produced:{C}` — bias > +15 pp.

Because the taxonomy is symmetric, "why does it predict UP for most cases" and
"why is it balanced but wrong" are the *same* function with a different `C`. The
mode is stored per split and in `artifacts/state/model_report.json`, so a future
retrain can be compared against today's mode.

---

## 4. Hypotheses — auditable reasons, not prose

Six candidate causes, each with a verdict and the exact numbers that decided it.
A **REJECTED** hypothesis is as valuable as a supported one: it stops the team
chasing the wrong cause.

| id | Verdict rule | What it means |
| --- | --- | --- |
| `class_overlap` | SUPPORTED when `separation < 0.05` **and** `share_below_half > 0.40` | The class probability vectors are nearly identical, so the decision is made by the tie-break, not by evidence. |
| `decision_rule_tiebreak` | SUPPORTED when rows below τ > 50% **and** fallback → DOWN ≥ 75% (or ≤ 25%) | Below τ the deployed rule discards `p_flat` and compares `p_down` vs `p_up` alone. |
| `flat_band_misset` | SUPPORTED when the best τ on the recomputed sweep differs from the configured `tau_flat` by ≥ 0.05 | Diagnostic only — it never changes the threshold. |
| `overfit` | SUPPORTED when train accuracy − test accuracy ≥ 10 pp | The same booster scored on the rows it was fitted on vs the sealed split. |
| `drift` | SUPPORTED when ≥ 30% of the top-15 SHAP features have PSI ≥ threshold; INCONCLUSIVE when no importance exists | PSI from `results/monitoring.json`. |
| `label_prior` | INFO always | Prior share, model accuracy, and the edge over a constant prediction. |

Verdict glyphs in the UI: `●` SUPPORTED, `○` REJECTED, `◐` INCONCLUSIVE, `◇` INFO.

**`decision_rule_tiebreak` is the deployed-rule hypothesis**, and it is why the
report can explain an always-DOWN model without blaming the features:
`apply_flat_threshold()` predicts FLAT only when `p_flat ≥ τ`; otherwise it
returns DOWN unless `p_up > p_down`, so *any* row below τ where DOWN leads even
slightly becomes a DOWN call. Panel D plus `rule.fallback_down_share` measure
exactly that.

---

## 5. The `model-report` stage

`python -m stockml.main model-report` — never trains, never writes to
`artifacts/models/`, never touches `results/predictions.parquet`.

It loads `artifacts/models/lightgbm_final.txt`, rebuilds `X, y` via
`load_training_frame` / `prepare_xy` (pinned to `model_metadata.feature_columns`
when present, so a growing feature store cannot silently change the input set),
splits with `three_way_date_split`, applies the stored temperature and
`apply_flat_threshold`, and writes four artifacts:

| Artifact | Contents |
| --- | --- |
| `results/predictions_splits.parquet` | `split, symbol, date, actual_label, predicted_label, p_down, p_flat, p_up, max_p, margin, correct, realized_return, event_start, event_end` for all three splits. `results/predictions.parquet` is left untouched. |
| `results/model_report.json` | `meta`, `consistency`, `splits{train,calibration,test}` (plate payloads), `comparison{accuracy_gap_pp, separation_gap, recall_gap}`, `thresholds[]`, `folds[]`, `importance[]`, `drivers[]`, `hypotheses[]`. |
| `results/explanations_splits.parquet` | Per-row top-12 contributions per class for a **stratified sample** (600 rows/split, stratified by predicted class so rare classes appear), with `base_value` per class. |
| `results/shap_by_class.csv` | `class, feature, mean_signed, mean_abs`. The signed-per-class matrix is what panel E needs; the older `shap_feature_importance.csv` only carries the all-class mean \|SHAP\|. |

Cost control: JSON rows are aggregated only (rolling per date), SHAP uses
1000-row samples per split via `TreeExplainer`. A full run is a minute-scale
read, not a training run.

### Consistency block (banner, not swallowed)

- `predictions_max_abs_diff` — the largest disagreement between the report's test
  probabilities and `results/predictions.parquet`; `predicted_label_agreement`
  and `rows_differing` alongside it.
- `dataset_signature` vs `metadata_signature` (`signature_matches`) — whether the
  feature matrix is the one the booster was fitted on.
- `feature_alignment` — `model_features`, `scored_features`, `store_features`,
  `missing_from_store`, `pinned_to_metadata`.
- `same_window` — whether the report's test window equals the evaluated one.

A mismatch means the features or the booster moved since evaluation. The page
then *says so* instead of presenting stale numbers as current.

---

## 6. Live training telemetry

Deliberately **not** scoring in-progress boosters: a half-written checkpoint would
report a quality number nobody can trust. Three observation-only writes instead:

| File | Written by | Contents |
| --- | --- | --- |
| `logs/train_heartbeat.json` | `TrainingHeartbeat` (atomic, throttled 2 s) | `status` (`optuna\|final_fit\|calibration\|done`), `trial`, `trial_total`, `fold`, `fold_total`, `iteration`, `num_boost_round`, `best_value`, `elapsed_s`, `eta_s`, `progress`, `updated_at` |
| `checkpoints/train_curve.jsonl` | `CurveRecorder` (every 10 rounds) | `{run, iteration, split: "train"\|"calibration", multi_logloss, accuracy}`; resume-safe via the `completed_rounds` offset |
| `checkpoints/trials/fold_diagnostics.jsonl` | per-fold diagnostics (pre-existing) | per-fold per-class recall, score, top-15 gain importances |

**Model-preservation guarantee.** The final fit gains
`valid_sets=[("calibration", ...)]` *only* for recording. There is no
`early_stopping` callback on the final fit, no `best_iteration` selection, and no
recorded number ever feeds back into `num_boost_round`, the params or pruning.
`train_with_resume` gains optional `X_cal`/`y_cal` kwargs; fold selection logic is
untouched. Instrumentation cannot change the model.

`GET /dashboard/model/live` returns heartbeat + last 400 curve points (one run
only — a curve mixing two signatures would draw a step nobody trained) + fold
dots + `training: bool` (heartbeat status in-flight and age ≤ 120 s). A `done`
heartbeat is a finished run, however recent. The Live tab polls at 5 s while
training and 60 s otherwise; the Behaviour tab keeps showing the last completed
report, labelled with `generated_at`, so live progress and finished quality are
never confused.

---

## 7. Per-case explanation

| Source | Artifact | Produced by |
| --- | --- | --- |
| Sampled rows of any split | `results/explanations_splits.parquet` | `model-report` |
| Today's serving rows, one per symbol | `results/explanations_serving.parquet` | `python -m stockml.main explain [--symbol X] [--top-k 12]` |

`explain` is **not** wired into `refresh` (it is the slowest optional step); the
UI shows a stale banner until it has run. It reuses `predict._load_latest_features`
and skips a symbol whose production features are incomplete rather than
zero-filling them.

`GET /dashboard/explain/{symbol}?date=&split=` reads the serving file when
`date`/`split` are absent, else the split file (falling back to the split file
with a warning if serving is missing). It returns `source, symbol, date, split,
actual, prediction, probabilities, base_value, contributions[{class, feature,
feature_value, contribution}], margin, method: "tree_shap"`.

Tree SHAP is additive in raw score space, so
`base_value + Σ contributions = class raw score` holds to ~1e-6, which is
asserted in `tests/test_model_diagnostics.py`. The drawer draws that waterfall and
states how far the top feature would have to move to cross the decision margin.

Because the explanations come from a **stratified sample**, every surface that
shows them says so — never "all rows".

---

## 8. API surface

All routes are `GET`, envelope-wrapped, and answer `data: null` + warning when an
artifact is missing (HTTP 200, never 500).

| Route | Payload |
| --- | --- |
| `/dashboard/model/report` | `meta`, `consistency`, `splits{train,calibration,test}`, `comparison`, `thresholds`, `folds`, `importance`, `drivers`, `hypotheses` |
| `/dashboard/model/live` | `training`, `heartbeat`, `heartbeat_age_seconds`, `curve[]`, `folds[]`, `checkpoint_rounds`, `limits` |
| `/dashboard/model/cases?split=&predicted=&actual=&symbol=&correct=&limit=&offset=` | paginated rows from `predictions_splits.parquet` + `total` |
| `/dashboard/explain/{symbol}?date=&split=` | waterfall (section 7) |

`/dashboard/model/live` is the one route that answers a *shaped* empty payload
(`training: false`, empty `curve`/`folds`) rather than `null`: "no run is
training" is a valid answer the UI can render, and the warning explains why it is
empty.

`model_report()` and `model_cases()` add warnings for a stale report (> 14 days),
a feature store wider than the trained set, a test window that moved, and results
that match no filter.

---

## 9. Frontend `#/model`

`ModelDiagnostics.jsx`, hash-tabbed: **Behaviour · Diagnosis · Cases · Training ·
Live**. Grayscale only, verdicts by glyph, 5 px button borders.

- **Behaviour** — split switcher (train / calibration / test) + **Compare** toggle
  rendering the three plates in one row with shared axes; headline, bias chips,
  `starved:*` chips, evidence list. Below: the τ sweep (accuracy / coverage /
  per-class recall, configured τ marked) and rolling predicted share.
- **Diagnosis** — one card per hypothesis: glyph verdict, the deciding numbers,
  the linked mini-chart (`profile`, `conviction`, `distribution`, `sweep`,
  `split_table`, `importance`), and which split it was measured on.
- **Cases** — filters (split, predicted, actual, symbol, correct) → table →
  waterfall drawer.
- **Training** — model card (rows, features, signature, params hash,
  calibration), split table (accuracy / balanced / MCC / ECE), three confusion
  matrices, fold-recall heatmap, importance with PSI overlay, learning curve.
- **Live** — heartbeat strip (phase, trial k/n, fold, iteration, ETA),
  train-vs-calibration loss per iteration, fold score dots, per-fold recall. Empty
  state explains it needs an instrumented training run.

Chart primitives live in `src/components/charts.jsx` (`BehaviourPlate`,
`PairedShareBars`, `GroupedProfileBars`, `ConvictionHistogram`,
`SignedDriverBars`, `Waterfall`, `RollingLine`, `DualAxisLine`).

---

## 10. Retrain policy

- **Not required.** `model-report` reads the existing booster; train-split
  performance comes from re-scoring `X_train` with that same booster, not from a
  new fit.
- **Allowed and model-neutral** if the booster/checkpoint is missing:
  `python -m stockml.main train` with `config/config.yaml` unchanged, then
  `evaluate` → `model-report`. The added instrumentation cannot change the model
  (section 6).
- **Guarded.** After any retrain the page shows `consistency.max_abs_diff` and the
  `dataset_signature` match state. If the new booster differs, it says "report and
  metrics were built from different runs" rather than blending them.
- **Never from the dashboard.** No route trains, retrains or changes τ.
- Determinism is not promised across hardware or thread counts; the guard
  reports, it does not pretend.

---

## 11. What these numbers cannot tell you

- The plates describe **decisions**, not profitability. Economics live in
  `/dashboard/backtest`.
- SHAP explains the booster's raw score, not causality, and only on the sampled
  rows stated in `meta.shap_sample_rows` / `meta.explain_sample_rows`.
- `calibration` is a temperature-fitted slice, not an untouched holdout. Only
  `test` is sealed; that is why `overfit` compares train against test.
- Tuning τ on test (as `thresholds[]` allows you to inspect) is **not** a licence
  to change it — `flat_threshold` remains the authority for τ.
- A missing artifact is reported as absent. Nothing here interpolates a series it
  did not read.

---

## 12. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\test_model_diagnostics.py tests\test_dashboard_api.py -q
```

`tests/test_model_diagnostics.py` builds synthetic frames with known shares and
pins: distribution shares and bias, confusion counts/recall, `separation` (0.0 for
identical profiles, 0.4 for corner vectors), the five conviction bins, the τ sweep
(FLAT iff `p_flat ≥ τ`), **all five modes plus `unknown`**, both all-SUPPORTED and
all-REJECTED hypothesis sets, and waterfall reconstruction within 1e-6. The
always-DOWN fixture reaching `collapsed_prior` with `fallback_down_share == 1.0`
proves the page is not hard-wired to any one class.
