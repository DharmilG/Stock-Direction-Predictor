"""Prepare black-and-white assets for the Chart Vision slide deck."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps

CV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CV_ROOT.parent
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# --- sample chart -----------------------------------------------------------
sample = sorted((CV_ROOT / "data" / "real").glob("*.png"))[0]
chart = Image.open(sample).convert("RGB")
gray = ImageOps.grayscale(chart)
gray.save(OUT / "chart_input.png")
print("chart_input from", sample.name)

# --- GradCAM (grayscale) ----------------------------------------------------
sys.path.insert(0, str(CV_ROOT))
from chartsense.adapter import get_status  # noqa: E402
from chartsense.api.inference import load_cv_model  # noqa: E402
from chartsense.training.gradcam import get_gradcam  # noqa: E402

if get_status()["ready"]:
    model = load_cv_model()
    heatmap = get_gradcam(chart, model=model)
    cam_gray = ImageOps.grayscale(Image.fromarray(heatmap))
    cam_gray = ImageOps.autocontrast(cam_gray)
    cam_gray.save(OUT / "gradcam_gray.png")
    print("gradcam_gray written")
else:
    print("no checkpoint; gradcam_gray skipped")

# --- confusion matrix (grayscale) ------------------------------------------
matrix_path = CV_ROOT / "evaluation" / "cv_confusion_matrix.png"
if matrix_path.exists():
    matrix = ImageOps.grayscale(Image.open(matrix_path).convert("RGB"))
    matrix.save(OUT / "confusion_gray.png")
    print("confusion_gray written")

for path in sorted(OUT.glob("*.png")):
    print(path.name, path.stat().st_size)
