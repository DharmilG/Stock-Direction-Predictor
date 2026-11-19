# Chart Vision — the words, the code, and the tools

This file is for you, not for the slides.
The slides stay short and clean. Here everything is written out properly, in
plain English. If a word on a slide is not clear, find it below.

The CNN deep-dive comes first, then the short word list, then the code folder by
folder, then the libraries, then the real numbers, then the commands, and
finally the honest limits.

---

## The CNN, explained properly (in easy words)

This is the detail behind slides 5 and 6. Read it once and the picture will make
sense. There is no maths here - just what actually happens to the image.

### What "scanning an image" really means

A computer does not see a chart the way you do. To it, the chart is a grid of
numbers: 224 rows by 224 columns, and three colour values at each point (red,
green, blue). That is 224 x 224 x 3 numbers.

The model scans this grid with a **filter**. A filter is a small square of
numbers - like a 3 x 3 or 7 x 7 window. The window sits on top of the image and
covers a few pixels. The model multiplies the numbers in the window with the
pixels underneath, adds them up, and writes one single number into a new image
called a **feature map**. Then it slides the window along and does it again.
That is the "scan".

- **Stride** is how far the window moves each step. Stride 2 means it jumps two
  pixels, so it is faster and the picture gets smaller.
- **Padding** means adding a border of zeros around the image so the window can
  sit at the edges too.
- **Pooling** shrinks the picture further by keeping only the strongest value in
  a small area.

Why do this? Because the first filters notice tiny things - a horizontal line,
a vertical edge, a colour change. The next filters combine those into candle
bodies and wicks. Deeper filters combine those into a double top or a flag. The
picture gets smaller at each step, but the description gets richer.

### One stop of the filter, up close

Let us zoom into one single stop. This is the smallest thing the CNN does.
Everything else is just this, repeated thousands of times.

Imagine the window is 3 x 3. It sits over nine pixels. Each pixel is a number
for brightness. The filter itself is also nine numbers - its "weights". Here is
a filter that looks for a bright horizontal line:

| | | |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 0 | 0 | 0 |

Now suppose the nine pixels under it look like this:

| | | |
|---|---|---|
| 10 | 10 | 10 |
| 200 | 200 | 200 |
| 10 | 10 | 10 |

What happens at this one stop:

1. **Multiply.** Each filter weight is multiplied by the pixel under it.
   0 x 10 = 0, then 0 x 10, then 0 x 10. Then 1 x 200 = 200, three times.
   Then 0 x 10 three more times.
2. **Add.** Add the nine results: 0 + 0 + 0 + 200 + 200 + 200 + 0 + 0 + 0 =
   **600**.
3. **Add a bias.** A small fixed number is added. Think of it as a threshold
   nudge. We ignore it here to keep the example clean.
4. **Activate.** The number goes through **ReLU**: if it is negative it becomes
   zero, if positive it stays. Here it stays 600.
5. **Write one cell.** That 600 is written into one cell of the new feature
   map, at the same place where the window sat.

If the window had sat on a dull patch with all pixels at 10, the answer would be
30. So a big number means "yes, this shape is here". A small number means "no".
That is all a filter does - it measures how much the shape it cares about is
present at that spot.

Now the important part:

- The window **slides one stride** and the same five steps happen again. That
  produces the next cell. Do it across the whole picture and you have a full
  feature map.
- There is not one filter but **64** at conv1. All 64 sit at the same stop and
  each writes one number. So that one spot becomes 64 numbers - 64 channels.
  One filter might find a horizontal edge, another a vertical edge, another a
  colour change.
- Each later layer takes the feature maps from the layer before it as its
  "image". So a deep filter is looking at earlier findings, not at raw pixels.
  That is how edges become candle bodies and candle bodies become a double top.

Nothing is learned live. These filter weights came from ImageNet training long
ago, and in our model they are frozen. We only use them to measure shapes.

### The layers, one by one

This is the real ResNet18 used in this project. The sizes below were measured by
running the model, not copied from a textbook.

| Step | What it does | Image size after it | Channels |
|---|---|---|---|
| Input | the 224 x 224 colour chart | 224 x 224 | 3 |
| conv1 | one 7 x 7 filter, stride 2, finds edges | 112 x 112 | 64 |
| maxpool | keeps the strongest values, shrinks | 56 x 56 | 64 |
| layer1 | 2 blocks, detail stays local | 56 x 56 | 64 |
| layer2 | 2 blocks, picture halves | 28 x 28 | 128 |
| layer3 | 2 blocks, picture halves again | 14 x 14 | 256 |
| layer4 | 2 blocks, final and richest | 7 x 7 | 512 |
| average pool | one number per channel | 1 x 1 | 512 |
| output | its old photo answer removed | - | **512 numbers** |

"Channels" just means how many different filters looked at the same spot. More
channels mean more kinds of things the model can notice. The last 512 numbers
are the compact description we hand to the two heads.

### Skip connections - all 8 of them

ResNet18 has 4 layers, and each layer has 2 blocks. That is 4 x 2 = **8 blocks**.
Every block has exactly **one skip connection**. So there are **8 skip
connections** in total.

A skip connection is a wire that carries the input of a block straight to its
output, adding it back in. Think of it as a bypass road next to a busy street.

- **5 of them are identity skips.** The input already has the right size, so the
  wire just passes it through unchanged. These are both blocks in layer1, and
  the second block of layer2, layer3 and layer4.
- **3 of them are projection skips.** The size changes at the first block of
  layer2, layer3 and layer4, so a tiny 1 x 1 filter resizes the signal before it
  is added back. That tiny filter is called a projection.

Why does this matter? In a deep model, the learning signal can fade as it travels
backwards - this is called the **vanishing gradient**. The skip wire gives the
signal a direct route, so an 18-layer model can actually be trained. Without it,
training would stall.

In our model every skip is frozen, just like the rest of the backbone. We never
change these wires during training.

### The full journey in one paragraph

The 224 x 224 chart goes in. conv1 finds edges and produces 112 x 112. Maxpool
shrinks it to 56 x 56. layer1 keeps local details. layer2, layer3 and layer4 keep
halving the picture while growing the description from 64 to 128 to 256 to 512
channels, ending at 7 x 7 x 512. Eight skip connections kept the learning signal
alive along the way. Average pooling squashes those 7 x 7 grids into one number
per channel, giving the final 512 numbers. The old 1000-photo answer layer is
removed. Those 512 numbers go to the pattern head and the direction head.

### Feeding one image to the CNN, step by step

Here is the exact path of one chart image, from the moment we hand it over to
the moment we get an answer. This is what the code in `api/inference.py` does
when someone uploads a picture in the dashboard.

1. **Load.** The PNG is opened and turned into colour pixels (red, green, blue).
   It can be any size to start with.
2. **Resize.** It is squashed to 224 x 224. ResNet18 expects that size. A big
   chart and a small chart both become the same grid.
3. **To numbers.** The picture is converted to a tensor - just a fancy name for
   a structured list of numbers. Three grids of 224 x 224, one per colour.
4. **Normalise.** We subtract a fixed average colour and divide by a fixed
   spread. These are the same values the model saw during ImageNet training,
   so the numbers sit in the range it expects. Without this, the same chart
   would look different to the model.
5. **Add a batch of one.** The model is built to handle a stack of pictures, so
   we wrap our single image in a stack of size one. The shape becomes
   1 x 3 x 224 x 224.
6. **Forward pass.** The image goes through conv1, maxpool, the four layers with
   their eight skip connections, and average pooling. Out come the 512 numbers.
   This is the reader doing its job. Nothing is trained here - it is pure
   reading.
7. **Two answers.** The 512 numbers go to both heads at once. Each head produces
   raw scores - six scores for the pattern head, three for the direction head.
8. **Softmax.** Each set of scores is turned into percentages that add to 100%.
   The biggest one wins. That is the answer and its confidence.
9. **Map the label.** The direction names are translated for the dashboard:
   bullish becomes UP, bearish becomes DOWN, neutral becomes FLAT.
10. **Optional heat map.** GradCAM runs the image again while recording which
    part of layer4 mattered most, and paints that region on the chart. This is
    the picture shown next to the answer in the dashboard.

Two things to remember:

- Steps 1 to 5 are the same every time. Steps 6 to 8 are the model. Steps 9 and
  10 are presentation.
- The model is cached after the first call, so uploading a second image does
  not reload the checkpoint from disk. If the checkpoint file changes, the
  cache is cleared automatically.

### Short words for the tricky terms

- **Filter** - a small window of numbers that slides over the image.
- **Feature map** - the new small picture a filter produces.
- **ReLU** - a rule that turns negative numbers into zero and keeps positive ones.
- **Tensor** - a structured list of numbers; here, the image stored as three colour grids.
- **Bias** - a small fixed number added to the filter result, like a threshold nudge.
- **Stride** - how many pixels the window jumps each step.
- **Padding** - a zero border so the window can reach the edges.
- **Pooling** - keeping the strongest value in a patch to shrink the picture.
- **Channel** - one more way of looking at the same spot.
- **Block** - two convolutions plus one skip connection.
- **Identity skip** - a shortcut that passes the signal through unchanged.
- **Projection skip** - a shortcut with a 1 x 1 filter that resizes the signal.
- **Vanishing gradient** - the learning signal fading before it reaches early
  layers. Skips fix this.

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
