# Glossary and Method Notes

**Purpose:** Every technical term in this project, explained in plain words. If someone joins the project and doesn't know finance or machine learning, they start here.

**Format:** everyday-words explanation first, technical definition second. No jargon without a prior definition.

---

## 1. Money Terms

### bps (basis points)

A way to talk about very small percentages. Think of cents inside a dollar. 100 bps = 1%. So 31.5 bps = 0.315%. That's what you lose on every round-trip trade (buy once, sell once) after accounting for all fees: brokerage, taxes, slippage, and miscellaneous charges. A 31.5 bps round-trip means if you buy a stock for Rs 100 and sell it later, the total cost of those two actions is roughly Rs 0.32. Sounds tiny. But if the stock only moved Rs 0.96 in a day (96 bps), you've already given 33% of that move to costs. Over hundreds of trades, this adds up fast and can turn a good-looking model into a money-losing one.

### Move / cost ratio

How many times bigger the typical price move is than your trading cost. At the 1-day horizon, the median move is 96 bps and the cost is 31.5 bps, so the ratio is 96 / 31.5 = 3x. Imagine winning a prize of Rs 96 that costs Rs 31.50 to collect. You keep Rs 64. At the 5-day horizon, the median move is 231 bps, same Rs 31.50 cost. Now you keep Rs 199. Same win rate, triple the kept profit. Higher ratio means more room for a model to be profitable even when it's wrong sometimes.

### Sharpe ratio

A single number that answers: "for every unit of risk (wobble) I took, how much return did I get?" If two models earn the same profit but one has a smoother equity curve, the Sharpe ratio picks the smoother one. A Sharpe of 0 means you earned exactly what the risk warranted — nothing extra. Below 0 means you took risk and got nothing for it. Our G1 gate demands Sharpe >= 0 before any other metric is looked at. If a model can't clear this bar, it's a coin flip with extra steps.

### Max drawdown

The biggest peak-to-trough drop in the equity curve. If your portfolio went from Rs 100 to Rs 70 to Rs 120, the max drawdown is 30% (from 100 down to 70). It measures how bad the worst losing streak was. Important because a model that makes 50% overall but has a 40% drawdown will psychologically (and financially) destroy the person running it.

### CAGR (Compound Annual Growth Rate)

The annualized return, assuming growth compounded smoothly each year. If you started with Rs 100 and ended with Rs 150 over 3 years, CAGR is about 14.5% per year. More honest than total return because it accounts for how long you waited.

---

## 2. Horizon and Labels

### Horizon (h)

The number of days ahead we're predicting. If h=5, the model sees everything up to today and must predict what happens over the next 5 trading days. Not tomorrow. Not next month. Five days. We chose this because:

- At h=1, the typical price move is 96 bps and costs eat 33% of it. You're playing a game where the house takes a third.
- At h=5, the typical move is 231 bps and costs eat only 14%. Same game, much better odds.
- At h=6 and beyond, UP share drifts past 0.53 and the model starts "predicting" that stocks drift upward over time — which is true but not intelligence. It's just buying and holding with extra steps.

### The h=1 to h=10 comparison (measured on NIFTY 50, full panel, 188k rows)

| h (days) | UP share | Median move (bps) | Move / cost ratio |
|---|---|---|---|
| 1 | 0.505 | 96 | 3.0x |
| 2 | 0.515 | 141 | 4.5x |
| 3 | 0.519 | 175 | 5.6x |
| 4 | 0.525 | 204 | 6.5x |
| **5** | **0.530** | **231** | **7.3x** |
| 6 | 0.533 | 255 | 8.1x |
| 7 | 0.536 | 277 | 8.8x |
| 8 | 0.538 | 298 | 9.5x |
| 9 | 0.541 | 317 | 10.1x |
| 10 | 0.543 | 336 | 10.7x |

Lock-in verdict: h=5 primary. h=3 kept as secondary diagnostic bar (best balance-per-unit-cost point at 5.6x). h=1 rejected: costs eat 1/3 of the move, autocorrelation is -0.007 (pure coin flip).

### UP share

Out of all past cases at a given horizon, what fraction went up. At h=1, 50.5% went up. At h=10, 54.3%. The number creeps upward because stock markets drift upward over time (equity risk premium — investors demand higher expected returns for holding risky stocks). This drift is real but not tradable by a model, because "always guess UP" would score 54% accuracy at h=10 without learning anything. A higher UP share raises the bar for the model to prove it's actually smart rather than just riding the drift.

### Triple-barrier labels

The labeling method. For each stock at each date:

- The stock starts at its current price.
- Draw a virtual ceiling (upper barrier) and a virtual floor (lower barrier) around that price. The width of these barriers is set by the stock's recent volatility times a multiplier (default 1.0x).
- Start a clock for h=5 trading days.
- If the price hits the ceiling first: label = UP.
- If the price hits the floor first: label = DOWN.
- If neither barrier is hit before the clock runs out: label = FLAT.

Why "triple" barrier? Because there are three possible outcomes: up, down, or flat. This is different from just looking at "is price higher after 5 days?" because it accounts for the path, not just the endpoint. A stock that goes up 5%, then crashes 6%, then recovers 2% is FLAT (never touched the ceiling after the initial spike — the barrier is wider than 5% for most stocks).

### Autocorrelation

Does today's direction remember yesterday's? If today went up, is tomorrow more likely to go up too? Autocorrelation measures this. At h=1, our autocorrelation is -0.007 — essentially zero. Tomorrow does not remember today. It's a coin flip. This is the single strongest reason to reject h=1: you can't learn a pattern that doesn't exist.

---

## 3. How We Judge Models

### Accuracy

What fraction of predictions were correct. Simple, but lies badly when classes are imbalanced. If 80% of cases are UP and you always guess UP, you get 80% accuracy without learning anything. Always read alongside class-level recall.

### AUC (Area Under the Curve)

A number between 0 and 1 that measures how well the model separates classes. 0.5 = random guessing. 1.0 = perfect separation. Our UP-AUC of 0.52 means the model is only slightly better than a coin flip at ranking UP vs not-UP. This is the primary metric for ablation comparisons.

### F1 score

The harmonic mean of precision and recall. Balances "how many of my positive predictions were right" (precision) with "how many of the real positives did I catch" (recall). Useful when both matter equally. Can be misleading if one class dominates.

### Precision

Of all the times the model said "UP", how often was it actually UP? High precision means few false alarms. The DOWN-flag reformulation tried to optimize precision on a base of 0.48 — and moved it from 0.470 to 0.471. Not useful.

### Recall

Of all the actual UP cases, how many did the model catch? High recall means few missed opportunities. The AFML (Advances in Financial Machine Learning) book says: build a high-recall primary first, then filter for precision second. We did it backwards with the DOWN-flag.

### Majority baseline

The dumbest possible model: always guess the most common class. If 53% of cases are UP, "always guess UP" gets 53% accuracy. Any model that doesn't beat this baseline isn't learning — it's just riding the coin's bias. Our gate: model must beat the majority baseline with statistical significance (DM test p < 0.005) before any other metric is considered.

### 2xSE bars

Statistical significance in plain words. Every metric has some natural wobble (standard error). If model A scores 0.53 and model B scores 0.52, but the wobble on each is 0.01, the real gap could be anywhere from 0.01 to 0.03 — or even negative (B might actually be better). The rule: A must beat B by at least twice the wobble (2 x SE) to count as a real improvement, not noise. Pre-registered before looking at results, applied to every ablation in this project.

### Diebold-Mariano test

A statistical test that answers: "did model A truly beat model B, or was it luck?" It accounts for the fact that financial predictions are correlated day-to-day (unlike coin flips). Newey-West correction handles autocorrelation in the loss series. We use p < 0.005 as the threshold — meaning there's less than a 0.5% chance the improvement is just luck.

### PBO (Probability of Backtest Overfitting)

A number between 0 and 1. High PBO means your backtest results are likely a mirage — the model overfit to the test period specifically, not to the market generally. Our G1 kill criterion includes PBO < 0.3. If PBO is higher, even good backtest results are suspect.

### Touch ledger

A JSONL file (holdout_touches.jsonl) that records every time anyone looks at holdout data, what they looked at, and what decision was made. Purpose: prevent unconscious bias. If you peek at test results and then tweak the model, that peek counts as a "touch" and must be logged. Twenty-three+ touches logged so far. The discipline: every touch is a ledger entry, every ledger entry is a commit, every commit is auditable.

---

## 4. Safety Machinery

### Embargo

A gap between training data and test data. If training ends on day T, test data starts on day T + embargo_days. This prevents the model from accidentally learning patterns that bridge the gap (like a news event that affects both the last training day and the first test day). Our default: 5 trading days (one week).

### Purge

Removing training examples whose labels overlap with test dates. If a label at day T covers days T through T+5, and the test set starts at T+3, that training example would have "seen" part of the test period through its label. Purge removes it. Our default: 5 days.

### Anti-memorization tripwire

A built-in alarm. After each fold, we compare training accuracy to test accuracy. If the gap exceeds 20 percentage points, a "MEMORIZATION FLAG" fires — meaning the model memorized the training data instead of learning generalizable patterns. If the gap exceeds 0.65 (65 points), a "TOO-GOOD FLAG" fires — meaning the result is so good it's almost certainly memorization. This is the formal version of your rule: "if it achieves too much accuracy like 95% then it means it is weaker to predict."

### PSI (Population Stability Index)

Measures how much a distribution has shifted. If the model was trained on features with mean 0.3 and 0.5, but today's features have mean 0.6, PSI captures that shift. High PSI (above 0.25) means the market has changed enough that the model's learned patterns may no longer apply. Used in monitoring, not training.

### Drift detection

The general concept of detecting when the world changes under a model. PSI is one method. Others include ADWIN, Page-Hinkley. Our monitoring layer runs PSI on features and predictions periodically and writes results to monitoring.json.

---

## 5. Neural-Track Terms

### LoRA (Low-Rank Adaptation)

A way to fine-tune a large model without retraining all of it. Instead of adjusting billions of parameters, you freeze the original model and train a small set of "adapter" matrices (low-rank) on top. On Qwen3-4B, LoRA means training ~5% of the parameters while the other 95% stay frozen. Result: fits in 6GB VRAM, trains in hours instead of days, and the base model's knowledge is preserved.

### STaR (Self-Taught Reasoning)

A training method where the model generates reasoning chains, then we keep only the ones that led to the correct answer. The model learns from its own correct reasoning — not from human-written chains. Applied here: Qwen reasons about a news article, predicts a direction, we compare to the historical label. Correct reasoning -> keep for LoRA training. Wrong reasoning -> discard. The model gradually learns "what reasoning leads to correct predictions" without memorizing prices it was never shown.

### Reasoning trace

The step-by-step thought process the model generates before committing to a direction. Unlike a simple label (UP/DOWN), a reasoning trace says: "The announcement mentions a leadership change. Historically, similar announcements in this sector preceded short-term dips followed by recovery within 5 days. Combined with the stock's current position near its 20-day high, I predict UP." These traces are saved and auditable — when the model is wrong, you read why, not just that it was wrong.

### Leakage scanner

An automated check that scans reasoning traces for phrases that reveal future information. If the model writes "given that the stock rose 8% in the following week," that's leakage — the model is peeking at the future. The scanner flags any mention of post-decision-date events, actual outcomes, or price levels after T. Flagged traces are discarded even if they led to the correct prediction (correct for the wrong reason is still wrong).

### Foundation model

A large model pre-trained on a massive general dataset (like all of English text, or all of public time-series data). It "knows" a lot about many things but nothing specific about your task. Fine-tuning adapts it to your specific problem. Examples: Qwen3-4B (language), Chronos (time-series), TimesFM (time-series). The tradeoff: foundation models bring general knowledge but may not specialize well for your specific market at your specific horizon.

### Zero-shot

Using a foundation model without any fine-tuning — just asking it the question directly. "Given this article about Reliance Industries, will the stock go up or down over the next 5 days?" Zero-shot tests whether the model's general knowledge is already good enough for the task. Our concern: zero-shot can fail for bad reasons (bad prompt, wrong framing) while the model itself is capable. That's why we have the two-stage gate: zero-shot probe first, LoRA rescue if it fails, kill only if both fail.

---

## 6. Key Constraints

- No stacking/ensemble in Phase 1 (MASTER_PLAN section 9.1). Single unified LightGBM with feature_fraction 0.6-0.8.
- Feature-fraction 0.6-0.8 to prevent memorization.
- Anti-memorization tripwire: train-test gap > 20pp triggers warning.
- Pre-registered significance bars: 2xSE for all ablation comparisons.
- India-only scope: NSE symbols, INR costs, IST timestamps.
- Free data sources only: no paid API subscriptions.
- Delivery cost model: 31.5 bps round-trip (realistic A-band).
- Every look at holdout data is logged in the touch ledger.
- "95% accuracy means weaker" — if a model is too good, it memorized.
