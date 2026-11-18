"""Central configuration for the CV lane.

Every path is anchored to CV_ROOT so commands work from any working directory.
No model, data or state file is created by importing this module.
"""

from __future__ import annotations

from pathlib import Path

CV_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = CV_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REAL_DIR = DATA_DIR / "real"
LABELS_CSV = DATA_DIR / "labels.csv"
REAL_LABELS_CSV = DATA_DIR / "real_labels.csv"

MODELS_DIR = CV_ROOT / "models"
ARTIFACTS_DIR = CV_ROOT / "artifacts"
STATE_DIR = ARTIFACTS_DIR / "state"
LOCK_DIR = STATE_DIR / "locks"
LEDGER_DIR = ARTIFACTS_DIR / "ledger"
LOGS_DIR = ARTIFACTS_DIR / "logs"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
QUARANTINE_DIR = ARTIFACTS_DIR / "quarantine"
EVALUATION_DIR = CV_ROOT / "evaluation"
TEMPLATES_DIR = CV_ROOT / "templates"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
STAGE1_BEST_PATH = MODELS_DIR / "stage1_best.pth"
STAGE1_LAST_PATH = MODELS_DIR / "stage1_last.pt"
STAGE2_LAST_PATH = MODELS_DIR / "stage2_last.pt"

# ---------------------------------------------------------------------------
# Label vocabularies (kept identical to the original ChartSense project)
# ---------------------------------------------------------------------------
PATTERNS = [
    "random",
    "head_and_shoulders",
    "double_top",
    "double_bottom",
    "bull_flag",
    "descending_triangle",
]
DIRECTIONS = ["neutral", "bullish", "bearish"]

# CV direction -> platform signal
SIGNAL_MAP = {"bullish": "UP", "bearish": "DOWN", "neutral": "FLAT"}

# triple_barrier_labels label_raw -> CV direction
LABEL_TO_DIRECTION = {-1: "bearish", 0: "neutral", 1: "bullish"}

# ---------------------------------------------------------------------------
# Version stamps. A mismatch refuses to resume instead of silently corrupting.
# ---------------------------------------------------------------------------
MODEL_ARCH = "chartsense-dual-v1"
CHECKPOINT_VERSION = 1
GENERATOR_VERSION = "synthetic-v1"
RENDER_VERSION = "mplfinance-224-v1"
DATASET_VERSION = "own-triple-barrier-v1"

ALL_DIRS = (
    DATA_DIR,
    SYNTHETIC_DIR,
    REAL_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    STATE_DIR,
    LOCK_DIR,
    LEDGER_DIR,
    LOGS_DIR,
    METRICS_DIR,
    QUARANTINE_DIR,
    EVALUATION_DIR,
)


def ensure_dirs() -> None:
    """Create the (empty) CV tree. Safe to call any number of times."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
