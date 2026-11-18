# Viva questions and answers — the CV lane

These are the questions a teacher or examiner is likely to ask about this
computer-vision project. The answers use simple words. When a technical word
appears, it is explained right there. Each answer also says **why**, because in
a viva the reasoning matters more than the definition.

---

## Basic ideas

### 1. What is computer vision?

**Answer:** It is teaching a computer to understand pictures. Instead of feeding
it text or numbers, we feed it an image and it gives back an answer about what
is in that image.

**Why:** In our project the "picture" is a candlestick chart, and the answer is
the chart pattern and the likely price direction. We use it because some traders
believe chart shapes carry information that plain numbers do not.

### 2. Why use candlestick chart images at all? Why not just use the prices?

**Answer:** The main model already uses prices as numbers (a *tabular* model,
meaning a model that reads a table of numbers). The CV model tries a different
view: it looks at the chart the way a human trader would. A pattern like a
"double top" is a shape, and shapes are natural for images.

**Why:** We want to test whether a shape-based view adds anything new. It is a
research question, not a replacement for the table model.

### 3. What is a candlestick?

**Answer:** One candle is a small bar that shows four prices for one day: the
price when trading started (open), the highest price, the lowest price, and the
price when trading ended (close). Many candles in a row make a chart.

**Why:** The model needs to see these four values for each day. Drawing them as
candles packs a lot of information into a small picture.

---

## The data

### 4. What are synthetic images and why do we use them?

**Answer:** "Synthetic" means made by us. We create a random price series and
then deliberately draw a known pattern into it, for example a double top. Since
we drew it, we know the correct answer.

**Why:** Real charts with clean, verified patterns are hard to find. Synthetic
charts are cheap, unlimited, and perfectly labelled, which lets the model first
learn what patterns look like.

### 5. Where do the real images come from?

**Answer:** From this project's own price files (`data/raw/*.parquet`). In
`make_own_dataset.py` we draw the last 60 candles into an image.

**Why:** Using our own data keeps the CV lane consistent with the rest of the
project.

### 6. How are the labels for real images made?

**Answer:** We use a function called `triple_barrier_labels`. It looks at the
next 5 sessions. If the price goes up past a barrier, the label is UP; if it goes
down past a barrier, DOWN; if it stays inside a band, FLAT.

**Why:** This is the *same* function the main project uses, so the CV model
predicts the same UP / FLAT / DOWN answer as the main model. That makes their
results comparable.

### 7. What is data leakage and how do we avoid it?

**Answer:** Data leakage is when the model secretly sees the answer while
learning. Here it would mean putting future prices inside the image. We avoid
it by drawing only candles up to day T, while the label uses days T+1 to T+5.
The future is never in the picture. The command `audit-labels` checks this.

**Why:** If future data leaked in, the model would look brilliant in testing but
useless in real life. Avoiding leakage keeps the result honest.

---

## The model

### 8. What is a neural network, in simple words?

**Answer:** It is a program made of many small maths steps stacked in layers.
Each step adjusts numbers a little. After seeing many examples, the stack learns
to map an input (a picture) to an output (an answer).

**Why:** It is the tool that can learn complicated patterns from data without a
human writing explicit rules.

### 9. What is ResNet18 and why is it "frozen"?

**Answer:** ResNet18 is a famous image model that already learned from millions
of photos. "Frozen" means we do not change what it learned. We only use it to
turn a chart picture into 512 numbers (a *feature vector*, a compact
description of the picture).

**Why:** Our chart dataset is small. If we let ResNet18 keep learning, it would
memorise our few pictures. Freezing it keeps the useful general features and
makes training faster.

### 10. What is a feature vector or embedding?

**Answer:** It is a short list of numbers that describes a picture. A human
might say "this chart has two peaks". The feature vector is the computer's
version of that description.

**Why:** The rest of the model works with these 512 numbers instead of the raw
pixels, which is simpler and faster.

### 11. Why do we have two heads?

**Answer:** A "head" is a small network that produces one answer from the 512
numbers. One head guesses the **pattern** (6 choices). The other guesses the
**direction** (3 choices). The shared backbone feeds both.

**Why:** The two questions are related but different. Sharing one backbone saves
work, and having two heads lets each answer be trained separately.

### 12. What are the 6 patterns and the 3 directions?

**Answer:** Patterns: random, head and shoulders, double top, double bottom,
bull flag, descending triangle. Directions: bullish (UP), bearish (DOWN),
neutral (FLAT).

**Why:** These are common chart shapes traders talk about, and three directions
match the labels used by the rest of the project.

---

## Training

### 13. What does "training" actually do?

**Answer:** We show the model a picture, it guesses, we measure how wrong it is
(the **loss**), and we nudge its numbers to be less wrong. We repeat this many
times. One full pass over all pictures is an **epoch**.

**Why:** This repeated nudge is how the model learns. There is no human writing
rules for each pattern.

### 14. What is cross-entropy loss?

**Answer:** A way to score a multiple-choice guess. If the model is very sure
about the wrong answer, it is punished heavily. If it is unsure, it is punished
less. Lower loss means better guesses.

**Why:** It pushes the model to be both correct and appropriately confident.

### 15. What are Adam and the learning rate?

**Answer:** Adam is the *optimiser*, the method that does the nudging. The
learning rate is how big the nudge is. A scheduler called `ReduceLROnPlateau`
makes the nudge smaller when learning slows down.

**Why:** Big nudges early, small nudges later, help the model settle on a good
answer instead of jumping around.

### 16. Why two training stages?

**Answer:** Stage 1 trains on clean synthetic charts to learn patterns. Stage 2
uses real charts and only fine-tunes the direction head, keeping the backbone
and pattern head frozen.

**Why:** The model first learns shapes in an easy setting, then adapts to messy
reality. Freezing most of it in stage 2 prevents it from forgetting what it
learned and avoids memorising the small real dataset.

### 17. What is a chronological split?

**Answer:** We train on the earlier 80% of dates and test on the later 20%. It
is split by time, not randomly.

**Why:** In real life we predict the future from the past. A random split would
mix future and past and give an overly happy result.

### 18. What is class weighting and label smoothing?

**Answer:** Class weighting tells the loss to care more about the rare neutral
class. Label smoothing softens the perfect answer a little so the model is not
overconfident.

**Why:** The neutral class is small, so without extra attention the model
ignores it. Smoothing makes the model more careful.

---

## Explainability and serving

### 19. What is GradCAM?

**Answer:** It makes a heat map showing which part of the chart the model looked
at when it decided. Red or bright areas mattered most.

**Why:** A plain answer like "UP" is hard to trust. Seeing where the model
looked helps a human judge whether the reason makes sense.

### 20. What does the API do?

**Answer:** It is a small web service. `GET /cv/health` says if the model is
ready, `/cv/training` shows progress, `/cv/evaluation` shows scores, and
`POST /cv/analyze` takes an uploaded image and returns the answer with the heat
map.

**Why:** It lets the dashboard ask questions without knowing the details of the
model.

### 21. What is softmax?

**Answer:** It turns the model's raw scores into probabilities that add up to 1.
We pick the largest one as the answer.

**Why:** Probabilities are easy to read: "70% bullish" tells you more than a
raw score like 3.7.

### 22. Why is the API read-only?

**Answer:** These routes never train the model, fetch data, or write files. They
only answer questions about an image.

**Why:** Read-only routes are safer and simpler. They cannot accidentally
change the project while someone is looking at the dashboard.

---

## Honesty and limits

### 23. The pattern model gets 94% accuracy but the direction model only 47%. Why?

**Answer:** The 94% is measured on synthetic charts where patterns are drawn
mathematically and are very clean. Real direction is much harder: it depends on
the future, which is noisy. On real held-out charts the direction model is about
47% accurate, only a little above random guessing (33% for three classes).

**Why:** This is the most important honest point. A high number on easy
synthetic data does not mean the model is good at real prediction.

### 24. Why is the neutral class so weak (F1 about 0.08)?

**Answer:** The model almost never predicts FLAT. The neutral examples are rare
and hard to separate from the others, so the model tends to guess UP or DOWN.

**Why:** It shows the model has not learned the "no clear move" case well. This
is a known weakness, not a hidden strength.

### 25. Is this CV model part of the main prediction model?

**Answer:** No. It is a separate research signal shown on its own dashboard
page. It does not feed the LightGBM model or Track Q.

**Why:** It has not yet proven it adds value. Until it is checked honestly on
holdout data, putting it into the main model could make things worse.

### 26. What would you add to make the project better?

**Answer:** A few things: (1) try the CV direction probability as an extra
feature for the main model and measure whether it helps; (2) fix the weak
neutral class with better weighting or thresholds; (3) use real labelled chart
windows instead of only synthetic ones; (4) test on a proper walk-forward
holdout before trusting it.

**Why:** These steps test whether the idea really works before it is used in the
production path. The rule is: prove it, then use it.

---

## Safety and resume

### 27. What happens if the job crashes halfway?

**Answer:** Nothing is lost. Every image is written safely (atomic write) and
recorded in an append-only log called a ledger. Restarting with `--resume` skips
what is finished and continues.

**Why:** Generating thousands of images and training for hours can be
interrupted. Safe resume saves a lot of time.

### 28. What is a fingerprint and why does it refuse to resume?

**Answer:** A fingerprint is a short code (a *hash*) of the code, config and
data. If you change any of them, the fingerprint changes. The job then refuses
to resume from an old checkpoint and asks you to start fresh with `--no-resume`.

**Why:** Continuing with a checkpoint made by different code could silently
produce wrong results. Refusing is safer than guessing.

### 29. What does the lock do?

**Answer:** A lock file makes sure only one generation or training job runs at
a time. If an old lock belongs to a dead process, it is moved to quarantine.

**Why:** Two jobs writing the same files at once would corrupt progress.

---

## Quick glossary (one line each)

- **Epoch** — one full pass over all training pictures.
- **Loss** — a number for "how wrong the model is"; lower is better.
- **Optimiser (Adam)** — the method that adjusts the model to reduce loss.
- **Learning rate** — how big each adjustment is.
- **Softmax** — turns scores into probabilities that sum to 1.
- **Feature vector** — a short list of numbers describing a picture.
- **Frozen backbone** — the image-reading part is fixed and not trained.
- **Head** — the small network that gives the final answer.
- **Fine-tuning** — small adjustments to an already trained model.
- **Data leakage** — secretly seeing the answer during training; must be avoided.
- **Triple barrier** — the rule that labels a future move as UP / DOWN / FLAT.
- **GradCAM** — a heat map showing which part of the image mattered.
- **Ledger** — an append-only log of what is finished.
- **Fingerprint** — a hash that detects changed code or data.
- **CORS / proxy** — a browser rule and a trick to avoid it during development.

---

## Final one-minute summary you can say in a viva

"This is a computer-vision research lane for stock charts. We draw candlestick
images, both synthetic ones with known patterns and real ones from our own
price data. Real images are labelled with the same triple-barrier rule as the
main project, so the target is UP, FLAT or DOWN. The model uses a frozen
ResNet18 to turn an image into 512 numbers, then two small heads guess the
pattern and the direction. We train in two stages: patterns on synthetic data,
then direction on real data. We use GradCAM to show where the model looked. The
API is read-only and serves the answer to a React dashboard page. The work is
resumable and safe against crashes using ledgers, locks and fingerprints. The
honest result is that pattern recognition is high on synthetic charts, but real
direction accuracy is only about 47%, so this stays a research signal and does
not feed the main model yet."
