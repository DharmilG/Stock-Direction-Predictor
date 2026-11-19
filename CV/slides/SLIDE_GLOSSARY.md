# Chart Vision — the words, the code, and the tools

This file is for you, not for the slides.
The slides stay short and clean. Here everything is written out properly, in
plain English. If a word on a slide is not clear, find it below.

The technical words are explained first. Then the code is explained folder by
folder, then the libraries, then the real numbers, then the commands, and
finally the honest limits.

---

## 1. The words used in the slides

Each entry says **what it is**, **why we used it**, and **which slide mentions it**.

### CNN (Convolutional Neural Network)
A type of model built for pictures. It looks at small patches of an image and
finds shapes: first edges, then corners, then bigger shapes like two peaks.
Why: candlestick charts are pictures, so a picture model is the natural tool.
Slide 5.

### ResNet18
A famous CNN with 18 layers of processing. It was trained on ImageNet, a huge
library of ordinary photos. "18" means how many weighted layers it has.
Why: it already knows how to see shapes, so we only need a little data to use it.
Slide 5.

### Skip connection (residual connection)
A shortcut that lets information jump past a block of layers. It is like a
bypass road. This makes deep models much easier to train.
Why: without it, 18 layers can stop learning (the "vanishing gradient" problem).
Slide 5.

### Backbone
The part of the model that reads the input and produces a description. Here it
is ResNet18. Think of it as "the reader".
Why: separating the reader from the answer part lets us reuse it.
Slides 5 and 6.

### Frozen
We do not change the backbone's learned numbers during training. It stays as it
was after ImageNet.
Why: our chart dataset is small. A frozen reader cannot memorise it.
Slides 5 and 6.

### 512-d feature (embedding)
The reader turns a 224x224 picture into a list of 512 numbers. That list is the
"description" of the picture. "512-d" means 512 numbers.
Why: a short list of numbers is easier to work with than pixels.
Slides 5 and 6.

### Head
A small add-on network that turns the 512 numbers into one answer. Ours is just:
512 numbers -> 256 hidden values -> answer.
Why: two small heads let us ask two different questions from one reader.
Slide 6.

### Softmax
A maths step that turns raw model scores into probabilities that add up to 1.
Example: UP 0.6, FLAT 0.1, DOWN 0.3.
Why: we can then say "the model is 60% sure it is UP".
Slides 6 and 8 (confidence numbers).

### Cross-entropy loss
A score for "how wrong the answer was". If the model is very sure and wrong,
the score is worse than if it was unsure and wrong. Training tries to lower it.
Why: it pushes the model to be both correct and properly confident.
Slide 6 (training).

### Learning rate
How big a step the model takes when it adjusts itself. 1e-3 = 0.001, 1e-4 =
0.0001. Stage 1 uses the bigger step, stage 2 the smaller one.
Why: big steps early, small steps later help it settle on a good answer.
Slide 6.

### Dropout
During training, some values are randomly ignored. The model cannot rely on one
single clue.
Why: it stops memorising and helps the model generalise to new charts.
Slide 6.

### Class weights
The FLAT class has only 166 images while UP and DOWN have about 1,450 each.
We tell the loss to pay more attention to FLAT.
Why: otherwise the model just ignores the rare class.
Slide 6.

### Label smoothing
Instead of training toward a perfect 100% answer, the target is softened a
little.
Why: it stops the model from becoming overconfident.
Glossary only (a training setting, not on a slide).

### GradCAM
A method that makes a heat map showing which part of the image mattered most for
an answer. Here it looks at the last ResNet block, called layer4.
Why: a human can check whether the reason makes sense.
Slides 7 and 9.

### AUC (one vs rest)
A number between 0 and 1 that says how well the model separates the classes.
1 is perfect, 0.5 is random. 0.564 is only a little above random.
Why: it is a fair score when classes are uneven.
Slide 8.

### F1 score
A single number combining precision and recall. Precision = "of the ones it
called UP, how many were really UP". Recall = "of all real UP, how many did it
find". F1 balances both.
Why: accuracy alone can lie when one class is rare.
Slide 8.

### Macro average
The average of each class's score, giving every class equal weight. Here 0.356.
Why: it shows the weak FLAT class instead of hiding it behind the big classes.
Slide 8.

### Confusion matrix
A table of "what the model said" against "what was true". The diagonal cells are
correct answers. Off-diagonal cells are mistakes.
Why: it shows *which* mistakes the model makes, not just how many.
Slide 8.

### Triple barrier
The labeling rule from the main project. Look 5 sessions ahead. If price hits a
barrier 1 sigma up, label UP. If it hits 1 sigma down, label DOWN. If it stays
inside a flat band of 0.25, label FLAT.
Why: the CV model then predicts the same UP / FLAT / DOWN target as the main
model.
Slides 3 and 4.

### Sigma
A measure of how much the price usually moves. We use a 20-session window to
estimate it. "1 sigma" means one typical move.
Why: barriers adjust to each symbol instead of using a fixed rupee amount.
Slide 3 (barriers).

### Chronological split
Split by date, not at random. The first 80% of dates train the model; the last
20% test it.
Why: in real life we predict the future from the past. A random split would mix
them and give an overly happy result.
Slides 3 and 8.

### Holdout
The part of the data kept back and never used in training. Here the future 20%.
Why: it is the only honest test of new data.
Slide 8.

### Overfitting
When a model memorises its training examples and fails on new ones.
Why we care: a frozen backbone and dropout both reduce this risk.
Slides 5 and 6.

### Fine-tuning
Small adjustments to a model that already learned something. Stage 2 fine-tunes
the direction head on real charts.
Why: cheap and needs less data than training from scratch.
Slide 6.

### ImageNet
A large public collection of ordinary photos with labels. ResNet18 was pretrained
on it.
Why: the reader arrives already knowing edges, textures and shapes.
Slide 5.

### OHLCV
Open, High, Low, Close, Volume — the five numbers that describe one candle.
Why: those are the raw facts we draw into each chart image.
Slide 4.

### mplfinance
A Python library that draws candlestick charts from OHLCV data.
Why: it makes the images the model reads.
Glossary (tool, not a slide word).

### 224 x 224
The image size in pixels. The model resizes every chart to this square.
Why: ResNet18 expects this input size.
Slides 1 and 4.

---

## 2. The code, folder by folder

The CV lane is its own small project inside `CV/`. It does not touch the main
LightGBM model or Track Q.

### `CV/` top level
- `README.md` — short guide: what the lane does, how to set it up, the commands.
- `CHART_SENSE_README.md` — the original upstream project's documentation.
- `LICENSE` — MIT license from the original project.
- `pyproject.toml` — says the package is called `chartsense` and needs Python 3.10+.
- `requirements-cv.txt` — the extra libraries CV needs (see section 3).
- `run_cv.ps1` — a small PowerShell helper. It sets `PYTHONPATH` to `CV/` and
  runs `python -m chartsense.cli` with whatever arguments you pass.
- `.gitignore` — ignores generated data, models, artifacts and pycache.
- `evaluation/` — the saved results and plots (see section 4).
- `slides/` — this deck, its builder, its images, and this glossary.
- `templates/index.html` — a standalone web page for the optional CV-only app.
- `notebooks/exploration.ipynb` — early exploration notebook.

### `CV/chartsense/` — the package itself
- `__init__.py` — marks the folder as a Python package.
- `config.py` — the single place for paths, label names, version stamps and the
  direction mapping (`bullish -> UP`, `bearish -> DOWN`, `neutral -> FLAT`).
  Nothing is created just by importing it.
- `adapter.py` — the bridge used by the platform API.
  `get_status()` reports readiness and stage progress.
  `analyze_chart(image)` loads the model, predicts, and adds a GradCAM heat map.

### `CV/chartsense/model/` — the model definition
- `backbone.py` — `get_backbone()` loads ResNet18 with ImageNet weights, freezes
  every parameter, removes the final classification layer, and returns the
  model plus its feature size (512).
- `heads.py` — `PatternHead` and `DirectionHead`. Each is a linear layer to 256,
  ReLU, dropout 0.3, then a linear layer to the number of classes.
- `chartsense.py` — `ChartSenseModel` joins them: one image in, two answers out.

### `CV/chartsense/data/` — making the pictures
- `generator.py` — makes the **synthetic** charts. It starts from a random walk
  and mathematically injects a known pattern (double top, head and shoulders,
  etc.). Each image is written atomically and recorded in a ledger.
- `make_own_dataset.py` — makes the **real** charts from this platform's own
  OHLCV parquet files. It draws 60 candles ending at day T, and labels each with
  the triple-barrier result from days T+1 to T+5. This is where "no peeking"
  is enforced. It imports `triple_barrier_labels` from the main project.
- `dataset.py` — `ChartDataset` (synthetic: image + pattern + direction) and
  `RealChartDataset` (real: image + direction only). PyTorch uses these to feed
  the model in batches.
- `fetcher.py` — a compatibility wrapper. Training uses local files, so no
  network call happens. The old yfinance fetch logic is kept only as an
  optional helper.

### `CV/chartsense/training/` — learning and checking
- `train.py` — the resumable two-stage trainer. Per epoch it saves an atomic
  checkpoint containing model, optimiser, scheduler, RNG state, epoch and
  fingerprints. Stage 1 trains both heads on synthetic charts. Stage 2 freezes
  the reader and pattern head, and fine-tunes only the direction head on real
  charts, using a chronological split and class weights.
- `evaluate.py` — evaluates the serving checkpoint, builds the confusion matrix
  and the classification report.
- `gradcam.py` — makes the heat map with the `grad-cam` library, targeting
  `backbone.layer4[-1]`.

### `CV/chartsense/api/` — serving
- `inference.py` — lazy model loading (cached, rebuilt if the checkpoint
  changes), image preprocessing (resize 224, to tensor, normalise), softmax
  prediction, and the base64 heat-map helper.
- `app.py` — an optional standalone FastAPI app for developers. It never loads
  a model at import time, so it starts even without a checkpoint.

### `CV/chartsense/fault/` — never losing work
- `atomic.py` — write to a temp file, flush, fsync, then `os.replace`. A crash
  leaves either the old file or the new one, never a half file.
- `ledger.py` — append-only JSONL log. `labels.csv` is derived from it, so a
  broken CSV can never destroy progress.
- `lock.py` — a process lock with pid liveness and heartbeat, so two jobs can
  never write the same files at once.
- `signals.py` — graceful stop: finish the current item, flush, then stop on
  Ctrl+C.
- `state.py` — small JSON state files for each stage.
- `fingerprint.py` — short hashes of code, config and data. A mismatch refuses
  to resume instead of silently training on incompatible state.
- `__init__.py` — exports all of the above.

### `CV/chartsense/cli.py` — the command centre
Every command is idempotent and resumable:
`status`, `doctor`, `generate-synthetic`, `generate-own`, `generate-all`,
`validate`, `audit-labels`, `train`, `train-all`, `evaluate`.
Flags include `--resume` (default on), `--reinit-head`, `--unfreeze-layer4`,
`--label-smoothing`, `--no-resume`.

### Outside `CV/`
- `src/stockml/api/cv.py` — the read-only platform API: `GET /cv/health`,
  `/cv/training`, `/cv/evaluation`, `POST /cv/analyze`. It never trains or writes.
- `src/stockml/api/server.py` — registers that router and allows POST.
- `stock-direction-dashboard/src/pages/Cv.jsx` — the "Chart Vision" page.
- `stock-direction-dashboard/src/lib/cvApi.js` — the frontend calls to the API.

---

## 3. Libraries used, and why

### Installed as a CV overlay (from `CV/requirements-cv.txt`)
| Library | Version | Why |
|---|---|---|
| `mplfinance` | 0.12.10b0 | Draw candlestick charts from OHLCV data |
| `grad-cam` | 1.5.7 | Build the GradCAM heat map |
| `opencv-python-headless` | 4.12.0.88 | Image operations without a GUI window |
| `python-multipart` | >=0.0.20 | Lets the API accept an uploaded image |

`torchvision` is installed first, by hand, from the PyTorch CPU index. It is not
in the requirements file because the `+cpu` build is not on PyPI. Installing
`grad-cam` first would otherwise pull a CUDA build.

### Already in the main project's `requirements.txt`
| Library | Why in CV |
|---|---|
| `torch`, `torchvision` | Build and train the model, load ResNet18 |
| `numpy`, `pandas` | Price arrays, labels, metrics |
| `scikit-learn` | Accuracy, F1, AUC, classification reports |
| `matplotlib`, `seaborn` | Plots and confusion matrix |
| `Pillow` | Open and save images |
| `fastapi`, `uvicorn` | Serve the `/cv` endpoints |
| `yfinance` | Only in the optional fetch helper; training uses local files |

### Used to build this deck
| Library | Why |
|---|---|
| `python-pptx` (1.0.2) | Create and overwrite the PowerPoint file |
| `Pillow` | Fit images into the slides |

---

## 4. The real numbers (and where they came from)

| Number | Value | Source file |
|---|---|---|
| Real chart images | 3,100 | `CV/data/real_labels.csv` |
| Symbols | 50 | `CV/data/real_labels.csv` |
| Date range | 2010-12-13 to 2026-10-21 | `CV/data/real_labels.csv` |
| UP / FLAT / DOWN | 1,486 / 166 / 1,448 | `CV/data/real_labels.csv` |
| Synthetic images | 120 | `CV/data/synthetic/` |
| Pattern classes | 6 | `CV/chartsense/config.py` |
| Horizon / sigma window | 5 / 20 sessions | main `config.yaml` via `make_own_dataset.py` |
| Barriers / flat band | 1.0 / 1.0 / 0.25 | main `config.yaml` |
| Pattern accuracy | 94% on 3,000 synthetic | `CV/evaluation/pattern_report.txt` |
| Direction accuracy | 47.2% on 676 holdout | `CV/evaluation/cv_metrics.json` |
| AUC | 0.564 | `CV/evaluation/cv_metrics.json` |
| Macro F1 | 0.356 | `CV/evaluation/cv_metrics.json` |
| Random guess baseline | 33.3% | 3 classes, by definition |

Per-class direction scores (holdout): neutral F1 0.08, bullish F1 0.51,
bearish F1 0.48.

---

## 5. Commands you can run

From the repo root:

```powershell
# rebuild the images used by the deck (needs a trained checkpoint for GradCAM)
.venv\Scripts\python.exe CV\slides\make_assets.py

# overwrite the PowerPoint deck
.venv\Scripts\python.exe CV\slides\build_slides.py

# CV lane status
.\CV\run_cv.ps1 status

# generate data (resumable)
.\CV\run_cv.ps1 generate-all --resume

# check the images and the labels
.\CV\run_cv.ps1 validate
.\CV\run_cv.ps1 audit-labels --sample 60

# train both stages (resumable)
.\CV\run_cv.ps1 train-all --resume

# evaluate
.\CV\run_cv.ps1 evaluate
```

---

## 6. Honest limits

- The 94% shape score is on **synthetic** charts where the shape is drawn by a
  formula. Real charts are messier.
- The direction score of 47.2% is only a little above the 33.3% random baseline.
  It is a research signal, not a trading edge.
- The FLAT class is weak (F1 0.08). The model almost never predicts FLAT.
- The holdout number is optimistic: the checkpoint was trained before the
  chronological split was fixed. A clean retrain is needed.
- Chart shape alone is not enough. Volume, sentiment and macro context matter.
- That is exactly why this lane stays separate from the main model.
