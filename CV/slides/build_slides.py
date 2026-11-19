"""Build the 7-slide black-and-white Chart Vision review deck."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "Chart_Vision_Review.pptx"

BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)
GRAY = RGBColor(110, 110, 110)
LIGHT = RGBColor(242, 242, 242)
FONT = "Aptos"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_text(slide, text, left, top, width, height, size=14, *, bold=False, color=BLACK, font=FONT, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_rule(slide, left, top, width, *, color=BLACK, height=0.015):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def base(number: int, kicker: str, headline: str, subtitle: str, footnote: str = ""):
    slide = prs.slides.add_slide(BLANK)
    add_text(slide, f"{number:02d} / 07", 0.6, 0.24, 1.6, 0.3, 9, color=GRAY, font=MONO)
    add_text(slide, "CHART VISION · CV LANE", 9.6, 0.24, 3.1, 0.3, 9, color=GRAY, font=MONO, align=PP_ALIGN.RIGHT)
    add_rule(slide, 0.6, 0.62, 12.13)
    add_text(slide, kicker, 0.6, 0.82, 12.1, 0.35, 12, bold=True)
    add_text(slide, headline, 0.6, 1.18, 12.1, 0.9, 30, bold=True)
    add_text(slide, subtitle, 0.6, 2.12, 12.1, 0.45, 14, color=GRAY)
    add_rule(slide, 0.6, 6.98, 12.13, color=GRAY)
    if footnote:
        add_text(slide, footnote, 0.6, 7.06, 12.1, 0.3, 8.5, color=GRAY, font=MONO)
    return slide


def add_table(slide, rows: list[list[str]], left, top, width, height, col_widths=None, font_size=10.5):
    table = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height)).table
    if col_widths:
        for index, value in enumerate(col_widths):
            table.columns[index].width = Inches(value)
    for row_index, row in enumerate(rows):
        table.rows[row_index].height = Inches(height / len(rows))
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLACK if row_index == 0 else (LIGHT if row_index % 2 else WHITE)
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = cell.text_frame
            frame.clear()
            paragraph = frame.paragraphs[0]
            run = paragraph.add_run()
            run.text = str(value)
            run.font.name = FONT if row_index == 0 else MONO
            run.font.size = Pt(font_size)
            run.font.bold = row_index == 0
            run.font.color.rgb = WHITE if row_index == 0 else BLACK
    return table


def add_image_fit(slide, path: Path, left, top, width, height):
    with Image.open(path) as image:
        ratio = image.width / image.height
    box_ratio = width / height
    if ratio > box_ratio:
        w, h = width, width / ratio
    else:
        h, w = height, height * ratio
    x = left + (width - w) / 2
    y = top + (height - h) / 2
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_box(slide, text, left, top, width, height, *, size=10.5, fill=WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = BLACK
    shape.line.width = Pt(1.1)
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = BLACK
    return shape


def arrow(slide, left, top, width=0.4, height=0.4):
    add_text(slide, "→", left, top, width, height, 20, bold=True, align=PP_ALIGN.CENTER)


# --- 01 cover ---------------------------------------------------------------
slide = base(
    1,
    "COMPUTER VISION · RESEARCH LANE",
    "Can a model read a candlestick chart?",
    "Images in. Pattern + direction out. Isolated from the tabular model.",
    "7 slides · black and white · research lane only",
)
add_image_fit(slide, ASSETS / "chart_input.png", 9.0, 2.8, 3.8, 3.6)
add_text(slide, "3,100", 0.9, 3.1, 3.0, 0.8, 44, bold=True)
add_text(slide, "REAL CHART IMAGES", 0.9, 3.85, 3.2, 0.3, 10, color=GRAY, font=MONO)
add_text(slide, "6", 0.9, 4.5, 3.0, 0.8, 44, bold=True)
add_text(slide, "PATTERN CLASSES", 0.9, 5.25, 3.2, 0.3, 10, color=GRAY, font=MONO)
add_text(slide, "224 × 224", 4.2, 4.5, 3.6, 0.8, 32, bold=True)
add_text(slide, "MODEL INPUT", 4.2, 5.25, 3.0, 0.3, 10, color=GRAY, font=MONO)

# --- 02 question ------------------------------------------------------------
slide = base(
    2,
    "01 · THE QUESTION",
    "Price alone doesn't explain moves.",
    "But traders read shapes first. We test the image.",
    "INPUT  →  FROZEN RESNET18  →  ATTENTION MAP",
)
add_image_fit(slide, ASSETS / "chart_input.png", 0.9, 2.9, 5.4, 3.4)
add_image_fit(slide, ASSETS / "gradcam_gray.png", 7.0, 2.9, 5.4, 3.4)
add_text(slide, "INPUT", 0.9, 6.45, 5.4, 0.3, 10, bold=True, color=GRAY, font=MONO)
add_text(slide, "ATTENTION", 7.0, 6.45, 5.4, 0.3, 10, bold=True, color=GRAY, font=MONO)

# --- 03 data table ----------------------------------------------------------
slide = base(
    3,
    "02 · DATA",
    "Two datasets. One label language.",
    "Textbook patterns teach shape. Real bars teach direction.",
    "50 symbols · flat band 1.0 · image ends at T, label uses T+1…T+5",
)
add_table(
    slide,
    [
        ["DATASET", "IMAGES", "LABELS", "SPLIT"],
        ["Synthetic patterns", "120", "pattern + textbook direction", "random 80/20"],
        ["Own NSE OHLCV", "3,100", "triple barrier · 5 sessions", "chronological 80/20"],
    ],
    0.9, 3.15, 11.5, 2.2,
    col_widths=[3.0, 1.6, 4.4, 2.5],
    font_size=11,
)

# --- 04 architecture diagram only ------------------------------------------
slide = base(
    4,
    "03 · ARCHITECTURE",
    "One backbone. Two heads.",
    "Diagram below — explanation on the next slide.",
)
add_box(slide, "CHART PNG\n224 × 224", 0.8, 3.0, 2.2, 0.85)
arrow(slide, 3.05, 3.2)
add_box(slide, "FROZEN RESNET18\n512-d features", 3.5, 3.0, 2.8, 0.85)
arrow(slide, 6.35, 3.2)
add_box(slide, "PATTERN HEAD\n6 classes", 6.8, 2.45, 2.5, 0.8)
add_box(slide, "DIRECTION HEAD\n3 classes", 6.8, 3.55, 2.5, 0.8)
arrow(slide, 9.35, 2.6)
add_box(slide, "GRADCAM\nlayer4 gradients", 9.8, 2.45, 2.6, 0.8)
add_text(slide, "STAGE 2", 0.9, 4.85, 1.6, 0.3, 9, bold=True, color=GRAY, font=MONO)
add_box(slide, "REAL CHARTS +\nTRIPLE-BARRIER LABELS", 1.0, 5.25, 3.6, 0.8)
arrow(slide, 4.65, 5.45)
add_box(slide, "DIRECTION HEAD\nFINE-TUNE", 5.1, 5.25, 3.0, 0.8)
arrow(slide, 8.15, 5.45)
add_box(slide, "SERVING CHECKPOINT", 8.6, 5.25, 3.4, 0.8)

# --- 05 explanation of the diagram -----------------------------------------
slide = base(
    5,
    "04 · HOW IT LEARNS",
    "Two stages. One frozen backbone.",
    "The diagram, explained.",
    "Backbone never updated · LR 1e-3 → 1e-4 · label smoothing 0.05 · class-weighted direction loss",
)
add_table(
    slide,
    [
        ["STAGE", "INPUT", "TRAINABLE", "OUTCOME"],
        ["1 · pattern", "120 synthetic charts", "both heads", "95.8% pattern accuracy"],
        ["2 · direction", "3,100 real charts", "direction head only", "serving checkpoint"],
    ],
    0.9, 3.15, 11.5, 2.2,
    col_widths=[2.2, 3.2, 3.0, 3.1],
    font_size=11,
)

# --- 06 results -------------------------------------------------------------
slide = base(
    6,
    "05 · RESULTS",
    "Shape is learned. Direction is not.",
    "Chronological holdout · n = 676",
    "Holdout is optimistic — checkpoint trained before the split fix",
)
add_table(
    slide,
    [
        ["METRIC", "VALUE"],
        ["Pattern accuracy", "95.8%"],
        ["Direction accuracy", "47.2%"],
        ["AUC (one-vs-rest)", "0.564"],
        ["Macro F1", "0.356"],
        ["Random baseline", "33.3%"],
    ],
    0.9, 3.0, 5.5, 3.4,
    col_widths=[3.5, 2.0],
    font_size=11,
)
add_image_fit(slide, ASSETS / "confusion_gray.png", 7.0, 2.9, 5.2, 3.5)
add_text(slide, "CONFUSION MATRIX", 7.0, 6.45, 5.2, 0.3, 9, bold=True, color=GRAY, font=MONO)

# --- 07 verdict -------------------------------------------------------------
slide = base(
    7,
    "06 · VERDICT",
    "Use it to see, not to trade.",
    "GradCAM lands on the pattern, not on the future.",
    "KEPT ISOLATED — never feeds LightGBM   |   NEXT — clean retrain, then decide",
)
add_image_fit(slide, ASSETS / "chart_input.png", 1.0, 2.85, 5.2, 3.0)
add_image_fit(slide, ASSETS / "gradcam_gray.png", 7.1, 2.85, 5.2, 3.0)
add_text(slide, "CHART", 1.0, 5.95, 5.2, 0.3, 9, bold=True, color=GRAY, font=MONO)
add_text(slide, "ATTENTION", 7.1, 5.95, 5.2, 0.3, 9, bold=True, color=GRAY, font=MONO)

prs.save(OUT)
print("wrote", OUT, "slides", len(prs.slides))
