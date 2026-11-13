# Architecture — NSE Stock Direction Platform

One picture of the whole system: **inputs → features → labels → models → decision → surface**, with the LLM experiment as a separate lane at the bottom that never feeds the main model.

Read **top to bottom**, one lane at a time. Every node carries a number (1 → 36); `project_QA.md` §19 explains each one in plain words, and this file holds the drawing.

```mermaid
flowchart TD
  subgraph L1["LANE 1 — INPUTS: all external"]
    S1(["1 · Yahoo Finance yfinance<br/>daily OHLCV, 50 symbols, 16 years"])
    S2(["2 · NSE bhavcopy<br/>delivery qty and pct, 2020 onward"])
    S3(["3 · NSE corporate announcements"])
    S4(["4 · Alpha Vantage + GDELT DOC<br/>+ RSS headlines"])
    S5(["5 · GDELT events via BigQuery<br/>historical backfill"])
    S6(["6 · Reference data<br/>calendar, sector and group maps"])
  end

  subgraph L2["LANE 2 — FEATURES, LABELS, SPLITS"]
    R7["7 · Refresh chain<br/>resumable, per-step state + lock"]
    D8{"8 · Published before<br/>15:30 IST?"}
    A8A(["Same session"])
    A8B(["Next real session<br/>weekends and holidays skipped"])
    F9["9 · Feature engineering<br/>183 columns per symbol-date"]
    X10["10 · Cross-sectional ranks<br/>each stock ranked 0-1 vs the other 49"]
    T11["11 · Triple-barrier labels<br/>plus or minus 1 sigma, 5 sessions"]
    M12[("12 · Feature matrix<br/>one row per symbol and date<br/>136 pinned columns")]
    D13{"13 · Which split?"}
    W14["14 · Purged walk-forward<br/>5 folds, 5-day purge, 5-day embargo"]
  end

  subgraph L3["LANE 3 — MODELS"]
    H15{{"15 · FinBERT classifier<br/>FIRST model touched<br/>scores announcement text"}}
    H16{{"16 · LightGBM multiclass<br/>primary predictor<br/>up to 1000 rounds"}}
    O17["17 · Optuna search<br/>25 trials in a SQLite study"]
    V18["18 · Per-class vote<br/>every tree votes, votes summed<br/>into 3 raw scores"]
    C19["19 · Temperature scaling<br/>scores to probabilities, T = 0.978"]
    K20[("20 · Model file, best params,<br/>metadata, checkpoint each round")]
  end

  subgraph L4["LANE 4 — DECISION, BACKTEST, SURFACE"]
    D21{"21 · p of FLAT<br/>at least 0.40?"}
    A21A(["Call FLAT"])
    A21B(["Call the larger of<br/>DOWN and UP"])
    D22{"22 · Confidence<br/>at least 0.60?"}
    A22A(["Act, trade"])
    A22B(["Signal only,<br/>do not trade"])
    B23["23 · Backtest<br/>no overlap per symbol<br/>31.5 bps round trip"]
    E24["24 · Evaluation<br/>accuracy, balanced, MCC<br/>DM test, ECE, SHAP"]
    D25{"25 · Gate result?"}
    C26[("26 · Artifacts<br/>metrics, metadata, calibration,<br/>predictions, latest CSV")]
    P27["27 · FastAPI read-only API<br/>dashboard endpoints, GET only"]
    U28["28 · React dashboard<br/>8 pages"]
  end

  subgraph L5["LANE 5 — TRACK Q: LLM experiment, parallel, does not feed the main model"]
    Q29["29 · Pool C<br/>1500 fixed iterations, pre-holdout<br/>500 DOWN, 250 FLAT, 750 UP"]
    Q30["30 · Prompt assembly<br/>news text + date + JSON rule<br/>2048-token context"]
    Q31{{"31 · Qwen3 8B Q4_K_M<br/>local via Ollama<br/>used only here"}}
    Q32["32 · Harness<br/>parse JSON, retries<br/>300 s per call"]
    Q33["33 · Leakage scanner<br/>regex audit of the reasoning"]
    Q34[("34 · Rating ledger, traces,<br/>heartbeat, GPU telemetry")]
    Q35{"35 · Qwen beats FinBERT<br/>by more than 2x standard error?"}
    Q36(["36 · Verdict file<br/>only when all 1500 are judged"])
  end

  S1 -->|raw downloads| R7
  S2 -->|raw downloads| R7
  S3 -->|raw downloads| R7
  S4 -->|raw downloads| R7
  S5 -->|raw downloads| R7
  S6 -->|calendar, maps| R7
  R7 -->|timestamped news| D8
  D8 -->|yes| A8A
  D8 -->|no| A8B
  A8A -->|aligned rows| F9
  A8B -->|aligned rows| F9
  F9 -->|same-day closes| X10
  X10 --> T11
  T11 -->|label + event_end| M12
  M12 -->|dataset| D13
  D13 -->|folds| W14
  W14 -->|training rows| H16
  S3 -->|announcement text| H15
  H15 -->|sentiment features| F9
  O17 -->|tuned hyperparameters| H16
  H16 -->|raw scores| V18
  V18 -->|logits| C19
  C19 -->|probabilities| D21
  D21 -->|yes| A21A
  D21 -->|no| A21B
  A21A --> D22
  A21B --> D22
  D22 -->|yes| A22A
  D22 -->|no| A22B
  A22A -->|tradeable rows| B23
  B23 -->|trade log| E24
  E24 -->|metrics| D25
  D25 -->|PASS| C26
  D25 -->|FAIL or PAUSE| C26
  H16 -->|model file| C26
  H16 -->|checkpoint| K20
  K20 -->|resume| H16
  C26 -->|JSON artifacts| P27
  P27 -->|read-only JSON| U28
  Q29 -->|iteration text| Q30
  Q30 -->|prompt| Q31
  Q31 -->|raw JSON vote| Q32
  Q32 -->|vote and trace| Q33
  Q33 -->|clean or flagged| Q34
  Q32 -->|judged votes| Q35
  H15 -.->|baseline score| Q35
  Q35 -->|yes| Q36
  Q35 -->|no| Q36

  linkStyle 18 stroke-width:4px
```

**Legend.** Rounded stadium = something outside this repository. Rectangle = a process step. Cylinder = something written to disk. Diamond = a fork in the road. Hexagon = a model. Thick arrow = the only model feeding the feature table. Dotted arrow = a comparison against a baseline, not a data dependency.

---

## Reading it, lane by lane

| Lane | What it settles | Detail |
| --- | --- | --- |
| 1 — INPUTS | Six external sources, nothing above them | `project_QA.md` §2 |
| 2 — FEATURES, LABELS, SPLITS | One row per symbol and date; the answer is made *before* any training | §4, and §4.2–§4.4 for the two hard parts |
| 3 — MODELS | FinBERT is met first (node 15), LightGBM predicts (node 16) | §4.5, §6 |
| 4 — DECISION, BACKTEST, SURFACE | The threshold rule, then the gate, then the artifacts and the browser | §4 (step 7), §14 |
| 5 — TRACK Q | The LLM as a **rater being measured**, not as a predictor | §8, §9, §10 |

## The five questions this picture answers

| Question | Answer | Nodes |
| --- | --- | --- |
| How does input reach the model? | Sources → one row per symbol and date → 183 columns → 136 pinned columns | 1–6 → 7 → 9 → 10 → 11 → 12 |
| Which model is touched first? | **FinBERT**, during data preparation; its output is a *feature*, not a prediction | 15 (thick arrow into 9) |
| How are votes counted? | LightGBM: every tree votes per class, votes summed into 3 raw scores. Qwen: one JSON vote per iteration | 16 → 18 → 19, and 31 → 32 |
| When is the LLM used, and for what? | Only in lane 5, only to be measured against FinBERT | 29 → 36 |
| How is a decision taken? | FLAT if p(FLAT) ≥ 0.40, otherwise the larger of DOWN and UP; trade only at ≥ 0.60 confidence. Separately, a research gate decides whether a track continues | 19 → 21 → 22, and 25 / 35 |

## What this diagram deliberately leaves out

No embeddings, chunking, vector database, RAG, retrieval, message queue, Kafka, Kubernetes, cloud storage, GPU cluster, and no second database. There is also no fine-tuned LLM anywhere on the path to node 21. The reason is in `project_QA.md` §19.5: this model reads **numbers in a fixed table**, not text passages, so there is nothing to retrieve.

One note on maintenance: the diagram exists only here. `project_QA.md` §19 holds the prose and links back to this file, so the drawing cannot drift into two versions.
