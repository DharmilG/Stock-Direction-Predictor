# The architecture of the CV lane, explained simply

This file walks through the computer-vision lane one layer at a time. The idea
is to explain it the way you would explain it to a friend: plain words first,
then the technical name when it is needed. Every technical word is explained
the first time it appears.

If you just want the picture, look at the Mermaid diagram in `README.md`.

## The one-sentence version

We turn candlestick charts into pictures, teach a small "brain" to recognise the
shape of the picture and to guess the direction of the price, and then show that
guess in the dashboard together with a heat map that says *where the brain was
looking*.

## A few words before we start

- **Computer vision** is just a fancy name for "teaching a computer to look at
  pictures". Here the pictures are candlestick charts.
- A **candlestick chart** is the usual price picture: each candle shows the
  opening price, the highest price, the lowest price and the closing price for
  one day (or one session).
- A **model** is the "brain". It is a maths program that takes numbers in and
  gives numbers out. When it learns from examples we call it *training*.
- An **image** is a grid of pixels. A pixel is one tiny square of colour. Our
  images are 224 by 224 pixels.

---

## Layer 1 — Data: making the pictures and the answers

Files: `chartsense/config.py`, `chartsense/data/generator.py`,
`chartsense/data/make_own_dataset.py`

Before a model can learn, we need two things for every picture:

1. the picture itself, and
2. the **correct answer** (called the *label*) for that picture.

We make pictures in two ways.

**Synthetic pictures** ("synthetic" means made by us, not real). In
`generator.py` we start with a random price series and then deliberately draw a
known shape into it. For example `inject_double_top` pushes the price up twice
to make a "double top". Because we drew the shape ourselves, we know the answer:
the pattern is "double top". These charts are clean and ideal. We call them
*synthetic* to remind ourselves they are not real market data.

**Real pictures**. In `make_own_dataset.py` we take the real price files that
this project already has (`data/raw/*.parquet`) and draw 60 candles from them
using `mplfinance` (a drawing library). These look like real charts.

The labels for the real pictures come from `triple_barrier_labels`, which is a
function that already exists in this project. It decides, for a given day,
whether the price in the next 5 sessions went up enough, down enough, or stayed
inside a band. The answer is one of three words:

- **bullish** → UP
- **bearish** → DOWN
- **neutral** → FLAT

This matters: because we use the *same* labelling function as the main project,
the CV model is trying to predict the same UP / FLAT / DOWN answer as the tabular
model. They speak the same language.

**No cheating (no data leakage).** A chart image is drawn using candles only up
to day T. The label uses the future days T+1 to T+5. The future never appears in
the picture. The CLI command `audit-labels --sample 60` checks this rule.

**Why this layer exists:** a model is only as good as its pictures and labels.
Making them carefully, and saving progress safely, is the foundation.

---

## Layer 2 — Dataset: handing pictures to the model

File: `chartsense/data/dataset.py`

The pictures and labels are stored on disk as PNG files and CSV files. A CSV is
a simple text table, like a spreadsheet saved as text. This layer wraps them in
a form that PyTorch can read. PyTorch is the library we use to build and train
the model.

There are two wrappers:

- `ChartDataset` is for the synthetic pictures. It gives back a picture plus
  **two** answers: the pattern and the direction.
- `RealChartDataset` is for the real pictures. It gives back a picture plus
  **one** answer: the direction.

**Why this layer exists:** training code wants to say "give me picture number 7
and its answer" over and over, without caring where the files are. This layer
does exactly that.

---

## Layer 3 — Model: the brain

Files: `chartsense/model/backbone.py`, `heads.py`, `chartsense.py`

This is the core. The model has three parts.

**The backbone (`backbone.py`).** We use `ResNet18`, which is a well-known image
model that came pre-trained on millions of ordinary photos (a collection called
ImageNet). "Pre-trained" means it has already learned to see edges, shapes and
textures. We **freeze** it, which means we do not change what it learned. We only
use it to turn a chart image into a list of **512 numbers** (called a *feature
vector* or *embedding*). Think of it as a short description of the picture.

Why freeze it? Two reasons. First, our chart dataset is small, so a frozen
backbone stops the model from memorising. Second, it is much faster to train.

**The heads (`heads.py`).** A "head" is a small extra network that turns the 512
numbers into a final answer. We have two heads:

- `PatternHead` guesses one of **6 patterns**: random, head and shoulders,
  double top, double bottom, bull flag, descending triangle.
- `DirectionHead` guesses one of **3 directions**: bullish, bearish, neutral.

Each head is a simple stack: a linear layer, a ReLU, a dropout, then a linear
layer. A **linear layer** mixes the numbers. **ReLU** keeps only positive values
(a simple switch). **Dropout** randomly ignores some values during training so
the model does not rely too much on one clue.

**The join (`chartsense.py`).** `ChartSenseModel` puts the backbone and the two
heads together. One picture goes in, two answers come out.

**Why this layer exists:** splitting the job into "understand the picture"
(backbone) and "give the answer" (heads) is a common and simple design. It lets
us train one part without touching the other.

---

## Layer 4 — Training: learning from examples

File: `chartsense/training/train.py`

Training means showing the model many pictures, letting it guess, measuring how
wrong it was, and nudging it to be less wrong next time.

The "how wrong" number is called the **loss**. We use **cross-entropy loss**,
which is a standard way to score a multiple-choice guess: confident and wrong is
punished more than unsure and wrong. The nudging is done by an **optimiser**
called **Adam**. A **scheduler** called `ReduceLROnPlateau` slowly lowers the
step size when progress stops; the step size is called the **learning rate**.

We train in **two stages**:

- **Stage 1** uses the synthetic pictures. Both heads learn. The learning rate
  is `1e-3` (that is 0.001).
- **Stage 2** uses the real pictures. We freeze the backbone and the pattern
  head, and only train the direction head. The learning rate is smaller,
  `1e-4` (0.0001), because we are making fine adjustments. This is called
  *fine-tuning*.

Two extra tricks in stage 2:

- **Class weights**: the neutral answer is rare, so we tell the loss to pay more
  attention to it. This is called *inverse-frequency class weighting*.
- **Label smoothing** (`--label-smoothing`): instead of being 100% sure, the
  target is softened a little. This stops the model from becoming overconfident.

We split the real data **by time**: the first 80% of dates train the model, the
last 20% are held back for testing. This is a **chronological split**, and it is
honest because it mimics predicting the future.

**Why this layer exists:** training is where learning happens. Every epoch (one
full pass over the pictures) is saved so we can stop and continue later.

---

## Layer 5 — Fault tolerance: never losing work

Files: `chartsense/fault/atomic.py`, `ledger.py`, `lock.py`, `fingerprint.py`,
`signals.py`, `state.py`

Long jobs get interrupted. This layer makes interruptions harmless.

- **Atomic writes**: a new file is written to a temporary name first, then
  renamed into place in one step. So you never get a half-written file.
- **Ledger**: an append-only text log (one line per image). If we restart, we
  read the ledger and skip pictures that are already done.
- **Locks**: a file lock stops two jobs from writing at the same time. A stale
  lock from a dead process is moved to quarantine.
- **Fingerprints**: a short hash of the code, config and data. If you change the
  code, the fingerprint changes and the job **refuses to resume** with the old
  checkpoint. This prevents silently mixing incompatible work. You must pass
  `--no-resume` to start fresh.
- **Graceful stop**: pressing Ctrl+C sets a flag; the current picture finishes,
  state is saved, and the ledger is flushed.

**Why this layer exists:** generating thousands of images and training for many
epochs can take a long time. Being able to stop and continue safely saves hours.

---

## Layer 6 — Serving: answering questions

Files: `chartsense/api/inference.py`, `chartsense/adapter.py`,
`src/stockml/api/cv.py`, `chartsense/training/gradcam.py`

This layer takes a new picture from a user and returns an answer.

**Loading**. `load_cv_model()` reads `best_model.pth` and keeps it in memory
(cached) so we do not reload it for every request. If the file is missing or
broken it raises `ModelUnavailable`, and the API returns a clear 503 error
instead of crashing.

**Preparing the picture**. The image is resized to 224×224, turned into a
tensor (a fancy word for a multi-dimensional array of numbers PyTorch uses),
and normalised (colours adjusted to a standard range the backbone expects).

**Predicting**. The model returns raw scores called *logits*. We pass them
through a **softmax**, which turns them into probabilities that add up to 1.
We take the biggest probability as the answer. We return the answer plus all the
probabilities, so you can see how sure the model was.

**The heat map**. `gradcam.py` uses **GradCAM** to produce a picture that shows
which part of the chart mattered most for the guess. This is called
**explainability**: it helps a human trust (or doubt) the answer.

**The endpoints** in `src/stockml/api/cv.py`:

- `GET /cv/health` — is the model ready?
- `GET /cv/training` — how is training going?
- `GET /cv/evaluation` — how good is it?
- `POST /cv/analyze` — here is a picture, give me the answer.

These routes are **read-only by design**: they never train, fetch data or write
files. They are under `/cv`, not `/dashboard`, because they do inference on a
user-uploaded image rather than reading saved artifacts.

**Why this layer exists:** it is the bridge between the trained file on disk and
a useful answer in the browser.

---

## Layer 7 — Dashboard: what you see

Files: `stock-direction-dashboard/src/pages/Cv.jsx`,
`stock-direction-dashboard/src/lib/cvApi.js`, `src/stockml/api/server.py`,
`stock-direction-dashboard/vite.config.js`

The React page `#/cv` is called "Chart Vision". It shows training progress,
charts of loss and accuracy, the evaluation report, a form to upload an image,
and the heat map. It asks the API for new data every 5 seconds, so the page
updates without a refresh.

`server.py` registers the CV router, and `vite.config.js` adds a `/cv` proxy so
the browser can talk to the API during development without CORS problems.
(**CORS** is a browser rule that blocks websites from calling a different
address; a *proxy* makes it look like the same address.)

**Why this layer exists:** a model nobody can see is not useful. The dashboard
makes the research visible.

---

## How it all connects to the rest of the project

It mostly does not. This is important to understand.

The main prediction model is **LightGBM** (a table-based model). Track Q is a
**language model** (Qwen) that reads news text. The CV lane reads **images**.
These are three separate signals. Today the CV signal is only shown on its own
page. It does not go into the LightGBM model and it is not part of Track Q.

That is a deliberate choice. The CV model is still a research signal, not a
proven improvement. Before it is trusted, its numbers must be checked honestly.
See the "Honest limitations" section in `README.md`.

## Where to read next

- `README.md` — the Mermaid diagram and the command list.
- `VIVA_QUESTIONS.md` — common questions and answers, in simple words.
- `CHART_SENSE_README.md` — the original project's documentation this was
  adapted from.
