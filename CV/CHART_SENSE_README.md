![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)

# ChartSense

> A computer vision system that reads candlestick chart images and detects technical trading patterns and price direction — no raw OHLC data, just the image.

---

## What Makes It Different

Most finance ML projects consume structured CSV data: open, high, low, close, volume. ChartSense takes a different approach — it treats charts **as images**, exactly the way a human trader does.

The input is a PNG. The model learns to *see* head-and-shoulders, flags, double tops — the same visual vocabulary traders use. There is no raw price data in the inference pipeline. If you can screenshot a chart, ChartSense can read it.

---

## Demo

Upload a chart image → get pattern detection + GradCAM heatmap showing exactly what the model looked at.

The file `gradcam_test.png` in the repo root is a sample output — the red/yellow overlay shows which region of the chart the model weighted most heavily when making its prediction.

---

## Architecture

The full pipeline, end to end:

```
Chart PNG
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Synthetic Data Generator                                │
│     mplfinance renders 3,000 candlestick chart images       │
│     across 6 pattern classes with mathematical injection    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PyTorch Dataset                                         │
│     Loads images + labels from labels.csv                   │
│     Standard ImageNet normalization transforms              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ResNet18 Backbone (pretrained ImageNet, frozen)         │
│     Transfer learning — low-level visual features reused    │
│     Final FC layer removed; outputs 512-dim feature vector  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Dual-Head Model                                         │
│     ├── Pattern head  → 6-class softmax (pattern type)      │
│     └── Direction head → 3-class softmax (bull/bear/neutral)│
│     Single forward pass, two losses, one optimizer          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  5. GradCAM                                                 │
│     Gradient-weighted class activation mapping              │
│     Highlights which spatial region drove the prediction    │
│     Overlaid as a heatmap on the original chart image       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  6. FastAPI Backend                                         │
│     POST /predict — returns pattern, direction, confidence  │
│     POST /gradcam  — returns heatmap PNG as base64          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Frontend Dashboard                                      │
│     Drag-and-drop chart upload                              │
│     Confidence scores for all 6 pattern classes             │
│     GradCAM heatmap rendered inline                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Patterns Detected

| Pattern | Signal |
|---|---|
| Head and Shoulders | Bearish reversal |
| Double Top | Bearish reversal |
| Double Bottom | Bullish reversal |
| Bull Flag | Bullish continuation |
| Descending Triangle | Bearish continuation |
| Random / No Pattern | Neutral |

---

## Tech Stack

| Component | Library |
|---|---|
| Model training | PyTorch + torchvision |
| Chart rendering | mplfinance |
| Real market data | yfinance (fetcher built, ready to use) |
| GradCAM | pytorch-grad-cam |
| Backend API | FastAPI + uvicorn |
| Image processing | OpenCV + Pillow |
| Evaluation | scikit-learn |
| Data handling | pandas + numpy |

---

## Project Structure

```
ChartSense/
├── chartsense/                     # Main Python package
│   ├── __init__.py
│   ├── api/
│   │   ├── app.py                  # FastAPI application, routes, frontend serving
│   │   └── inference.py            # Model loading, prediction, GradCAM orchestration
│   ├── data/
│   │   ├── generator.py            # Synthetic chart image generator (mplfinance)
│   │   └── dataset.py              # PyTorch Dataset — image loading and transforms
│   ├── model/
│   │   ├── backbone.py             # ResNet18 feature extractor (pretrained, frozen)
│   │   ├── heads.py                # Pattern classification + direction prediction heads
│   │   └── chartsense.py           # Dual-head model combining backbone and heads
│   └── training/
│       ├── train.py                # Training loop with dual loss and checkpointing
│       └── gradcam.py              # GradCAM implementation and heatmap generation
├── data/
│   ├── synthetic/                  # Generated candlestick chart images (PNG)
│   ├── real/                       # Real chart images (yfinance pipeline ready)
│   └── labels.csv                  # Image paths, pattern labels, direction labels
├── models/
│   └── best_model.pth              # Best checkpoint saved during training
├── templates/
│   └── index.html                  # Frontend dashboard (drag-and-drop UI)
├── notebooks/                      # Exploratory analysis notebooks
├── tests/                          # Unit tests
├── gradcam_test.png                # Sample GradCAM output (model attention heatmap)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package metadata and build config
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harsh-617/ChartSense.git
cd ChartSense
```

### 2. Create and activate a conda environment

```bash
conda create -n chartsense python=3.11
conda activate chartsense
```

### 3. Install PyTorch (CPU build)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install dependencies and package

```bash
pip install -r requirements.txt
pip install -e .
```

### 5. Generate synthetic training data

```bash
python chartsense/data/generator.py
```

This renders 3,000 candlestick chart images across 6 pattern classes into `data/synthetic/` and writes `data/labels.csv`.

### 6. Train the model

```bash
python -m chartsense.training.train
```

Checkpoints are saved to `models/best_model.pth`.

### 7. Start the API server

```bash
uvicorn chartsense.api.app:app --reload
```

### 8. Open the dashboard

```
http://127.0.0.1:8000/app
```

---

## Results

| Phase | Data | Pattern Accuracy | Direction Accuracy |
|-------|------|-----------------|-------------------|
| Phase 1 | 3,000 synthetic charts | ~100% | ~100% |
| Phase 2 fine-tune | 300 real charts (yfinance) | frozen | 41.7% val |

**Key insight:** Synthetic accuracy is high because patterns are mathematically clean. Real direction prediction at 41.7% is above the 33.3% random baseline — the model learns something genuine from real market visuals, but visual patterns alone have a clear ceiling. This is consistent with what quantitative research shows — price direction requires more than chart shape alone.

---

## Evaluation

Evaluation results are generated by `chartsense/training/evaluate.py`.

![Confusion Matrix](evaluation/pattern_confusion_matrix.png)

> The diagonal shows near-perfect separation between all named patterns. The only notable confusion is random charts being misclassified as patterns (0.33 error rate) — the model occasionally sees structure in noise, which mirrors how human traders behave.

| Pattern | F1 Score |
|---------|----------|
| random | 0.80 |
| head_and_shoulders | 0.97 |
| double_top | 0.96 |
| double_bottom | 0.94 |
| bull_flag | 0.97 |
| descending_triangle | 0.99 |

---

## What the GradCAM Shows

GradCAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the final convolutional feature map, then uses those gradients as spatial weights. The resulting heatmap shows *where in the image* the model concentrated its attention.

For a head-and-shoulders chart, the attention zone (red/yellow in `gradcam_test.png`) centers on the three peaks — the head and both shoulders. The model did not just memorize a label; it learned to locate the defining visual structure of the pattern. This is the key validation that the model is doing genuine visual pattern recognition rather than exploiting spurious correlations.

![GradCAM sample output](gradcam_test.png)

---

## Honest Limitations

**Pattern classification — synthetic only**
Pattern labels (head and shoulders, double top, etc.) are only available for synthetic data because real charts require human annotation. Pattern accuracy of ~100% on synthetic data is inflated — patterns are mathematically injected and idealized. Real charts have noise, partial patterns, and ambiguous formations.

**Direction prediction — real data is hard**
Direction fine-tuning on 300 real charts achieves 41.7% validation accuracy on a 3-class problem (bullish / bearish / neutral). Random baseline is 33.3%, so the model is learning something real — but it is not impressive. Predicting 5-day price direction from visuals alone is genuinely unsolved. Professional quant funds with terabytes of data and alternative signals struggle with this. Visual patterns alone are not sufficient — you would also need volume analysis, sentiment data, and macro context.

**Dataset size**
3,000 synthetic + 300 real images is small by deep learning standards. More data, especially human-labeled real charts, would improve both tasks significantly.
