"""Build the Chart Vision review deck (overwrites Chart_Vision_Review.pptx).

Plain words, short lines, real numbers from CV/evaluation reports.
CV lane only - no code, no viva content (that lives in SLIDE_GLOSSARY.md).
"""
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


def add_text(slide, text, left, top, width, height, size=14, *, bold=False,
             color=BLACK, font=FONT, align=PP_ALIGN.LEFT):
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


def add_bullets(slide, items, left, top, width, height, *, size=14, gap=6,
                color=BLACK, marker="—"):
    """Short one-idea lines, like a tweet thread. One paragraph per line."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for i, line in enumerate(items):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        run = p.add_run()
        run.text = f"{marker}  {line}"
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return box


def add_rule(slide, left, top, width, *, color=BLACK, height=0.015):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def base(number: int, total: int, kicker: str, headline: str, subtitle: str, footnote: str = ""):
    slide = prs.slides.add_slide(BLANK)
    add_text(slide, f"{number:02d} / {total:02d}", 0.6, 0.24, 1.6, 0.3, 9,
             color=GRAY, font=MONO)
    add_text(slide, "CHART VISION · CV LANE", 9.6, 0.24, 3.1, 0.3, 9,
             color=GRAY, font=MONO, align=PP_ALIGN.RIGHT)
    add_rule(slide, 0.6, 0.62, 12.13)
    add_text(slide, kicker, 0.6, 0.82, 12.1, 0.35, 12, bold=True)
    add_text(slide, headline, 0.6, 1.18, 12.1, 0.9, 30, bold=True)
    add_text(slide, subtitle, 0.6, 2.12, 12.1, 0.45, 14, color=GRAY)
    add_rule(slide, 0.6, 6.98, 12.13, color=GRAY)
    if footnote:
        add_text(slide, footnote, 0.6, 7.06, 12.1, 0.3, 8.5, color=GRAY, font=MONO)
    return slide


def add_table(slide, rows, left, top, width, height, col_widths=None, font_size=10.5):
    table = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height)
    ).table
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


def add_image_fit(slide, path, left, top, width, height):
    with Image.open(path) as image:
        ratio = image.width / image.height
    box_ratio = width / height
    if ratio > box_ratio:
        w, h = width, width / ratio
    else:
        h, w = height, height * ratio
    x = left + (width - w) / 2
    y = top + (height - h) / 2
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    width=Inches(w), height=Inches(h))


def add_box(slide, text, left, top, width, height, *, size=10.5, fill=WHITE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
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


def add_stat_box(slide, number, label, left, top, width, height, *, num_size=40):
    """A bordered box with a big number and a small label inside it."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = BLACK
    shape.line.width = Pt(1.1)
    add_text(slide, number, left + 0.25, top + 0.18, width - 0.5, 0.7, num_size, bold=True)
    add_text(slide, label, left + 0.25, top + height - 0.5, width - 0.5, 0.3, 9,
             color=GRAY, font=MONO)
    return shape


TOTAL = 10

# ---------------------------------------------------------------- 01 cover
slide = base(
    1, TOTAL,
    "COMPUTER VISION · RESEARCH LANE",
    "Can a model read a candlestick chart?",
    "A picture goes in. A shape and a direction come out. Kept separate from the main model.",
    "Real reports: CV/evaluation/ · cv_metrics.json · pattern_report.txt",
)
add_image_fit(slide, ASSETS / "chart_input.png", 9.0, 2.8, 3.8, 3.6)
add_stat_box(slide, "3,100", "REAL CHART IMAGES", 0.9, 2.95, 3.2, 1.45, num_size=40)
add_stat_box(slide, "50", "NSE SYMBOLS", 0.9, 4.6, 3.2, 1.45, num_size=40)
add_stat_box(slide, "224 x 224", "WHAT THE MODEL SEES", 4.35, 4.6, 4.0, 1.45, num_size=28)

# ---------------------------------------------------------------- 02 why
slide = base(
    2, TOTAL,
    "01 · THE QUESTION",
    "Prices are numbers. A chart is a picture.",
    "A table already reads the numbers. Here we test the picture.",
    "The model never sees news or prices after the decision day.",
)
add_image_fit(slide, ASSETS / "chart_input.png", 0.9, 2.85, 5.4, 3.4)
add_bullets(slide, [
    "Traders look at the shape first.",
    "A double top looks like a double top.",
    "That shape is hard to see in a table row.",
    "So we let the model look at the drawing.",
    "Nothing here changes the main prediction model.",
], 7.0, 2.95, 5.5, 3.2, size=14, gap=9)
add_text(slide, "ONE CHART, TWO VIEWS", 0.9, 6.45, 5.4, 0.3, 10,
         bold=True, color=GRAY, font=MONO)

# ---------------------------------------------------------------- 03 data
slide = base(
    3, TOTAL,
    "02 · DATA WE CAPTURED",
    "Two sets. One label language.",
    "Made-up charts teach the shape. Real charts teach the direction.",
    "Real labels come from this project's own triple-barrier rule.",
)
add_table(
    slide,
    [
        ["SET", "IMAGES", "LABELS", "SPLIT"],
        ["Synthetic patterns", "120", "shape + textbook direction", "random 80 / 20"],
        ["Real NSE charts", "3,100", "triple barrier · 5 sessions", "by date, 80 / 20"],
    ],
    0.9, 3.1, 11.5, 2.0,
    col_widths=[3.0, 1.6, 4.4, 2.5],
    font_size=11,
)
add_bullets(slide, [
    "50 symbols, 2010-12-13 to 2026-10-21.",
    "Barriers: 1 sigma up, 1 sigma down, flat band 0.25.",
    "6 shapes: random, head-shoulders, double top, double bottom, bull flag, triangle.",
], 0.9, 5.35, 11.5, 1.3, size=12, gap=5)

# ---------------------------------------------------------------- 04 features
slide = base(
    4, TOTAL,
    "03 · WHAT ONE IMAGE HOLDS",
    "60 candles in one small picture.",
    "The picture ends on day T. The answer uses T+1 to T+5.",
    "No future day is ever drawn into the picture.",
)
add_box(slide, "60 CANDLES\nOPEN · HIGH\nLOW · CLOSE\nVOLUME", 0.9, 3.0, 3.0, 2.3, size=13)
arrow(slide, 4.05, 3.85)
add_box(slide, "224 x 224\nRGB IMAGE", 4.55, 3.0, 2.6, 2.3, size=13)
arrow(slide, 7.3, 3.85)
add_box(slide, "ONE LABEL\nUP / FLAT / DOWN", 7.85, 3.0, 3.0, 2.3, size=13)
add_text(slide, "REAL LABEL COUNTS", 0.9, 5.6, 3.0, 0.3, 10, bold=True, color=GRAY, font=MONO)
add_table(
    slide,
    [["UP", "FLAT", "DOWN"], ["1,486", "166", "1,448"]],
    0.9, 5.95, 11.5, 0.75,
    col_widths=[3.83, 3.83, 3.84],
    font_size=11,
)

# ---------------------------------------------------------------- 05 resnet
slide = base(
    5, TOTAL,
    "04 · ARCHITECTURE · READER",
    "ResNet18 reads the picture.",
    "It is already trained on ordinary photos. We do not change it.",
    "Frozen = the reader stays the same; only the answer part learns.",
)
stages = [
    ("INPUT", "224x224x3"),
    ("CONV1", "64"),
    ("LAYER1", "64"),
    ("LAYER2", "128"),
    ("LAYER3", "256"),
    ("LAYER4", "512"),
    ("OUTPUT", "512 numbers"),
]
x = 0.7
for name, dim in stages:
    add_box(slide, f"{name}\n{dim}", x, 3.0, 1.62, 1.15, size=10.5)
    x += 1.72
add_bullets(slide, [
    "Each block is a small set of filters that finds edges, then shapes.",
    "Skip connections are shortcuts. The next slide shows all 8 of them.",
    "ImageNet = a big photo library the reader learned from before.",
    "We keep the 512 numbers. We throw away its old photo answer.",
], 0.9, 4.55, 11.6, 2.1, size=13, gap=7)

# ---------------------------------------------------------------- 06 scanning
slide = base(
    6, TOTAL,
    "05 · ARCHITECTURE · SCANNING",
    "How the reader scans the picture.",
    "A small window slides over the chart. Every stop becomes one number.",
    "4 layers x 2 blocks = 8 blocks. Each block owns one skip connection.",
)
ladder = [
    ("INPUT", "224x224x3"),
    ("CONV1", "112x112"),
    ("LAYER1", "56x56"),
    ("LAYER2", "28x28"),
    ("LAYER3", "14x14"),
    ("LAYER4", "7x7"),
    ("POOL", "512 numbers"),
]
x = 0.7
for name, dim in ladder:
    add_box(slide, f"{name}\n{dim}", x, 2.95, 1.62, 1.0, size=10)
    x += 1.72
add_bullets(slide, [
    "A filter is a small grid that slides across the image.",
    "Each stop multiplies and adds. One number per stop.",
    "Early filters see edges. Deeper filters see whole shapes.",
    "Stride = step size. Pooling shrinks the picture as it goes deeper.",
    "8 blocks in all. Each block has one skip connection - a shortcut.",
    "5 skips pass straight through. 3 use a tiny 1x1 filter to resize.",
    "The shortcut lets information reach the end. That makes 18 layers trainable.",
    "Last step: average pooling turns 7x7x512 into 512 numbers.",
], 0.9, 4.2, 11.6, 2.6, size=11.5, gap=6)

# ---------------------------------------------------------------- 07 heads
slide = base(
    7, TOTAL,
    "06 · ARCHITECTURE · ANSWERS",
    "Two heads. Two questions.",
    "Head = a small add-on that turns the 512 numbers into one answer.",
    "512 numbers -> 256 hidden values -> 6 shapes or 3 directions.",
)
add_box(slide, "512 NUMBERS\nFROM READER", 0.9, 3.15, 2.9, 1.5, size=12)
arrow(slide, 3.95, 3.6)
add_box(slide, "PATTERN HEAD\n6 SHAPES", 4.45, 2.7, 3.3, 1.1, size=12)
add_box(slide, "DIRECTION HEAD\nUP · FLAT · DOWN", 4.45, 4.0, 3.3, 1.1, size=12)
add_bullets(slide, [
    "Stage 1 learns shapes on made-up charts.",
    "Stage 2 freezes the reader and learns direction on real charts.",
    "Dropout drops some values while learning, so it cannot memorise.",
    "Class weights make the rare FLAT case count more.",
], 8.1, 2.75, 4.4, 3.2, size=12, gap=8)
add_text(slide, "LEARNING RATE: 1e-3 THEN 1e-4", 0.9, 5.15, 3.4, 0.3,
         9, bold=True, color=GRAY, font=MONO)

# ---------------------------------------------------------------- 08 gradcam
slide = base(
    8, TOTAL,
    "07 · EXPLAINABILITY",
    "See where the model looked.",
    "GradCAM paints the part of the chart that drove the answer.",
    "A heat map on the last ResNet block, shown on the same picture.",
)
add_image_fit(slide, ASSETS / "chart_input.png", 0.9, 2.9, 5.4, 3.4)
add_image_fit(slide, ASSETS / "gradcam_gray.png", 7.0, 2.9, 5.4, 3.4)
add_text(slide, "INPUT CHART", 0.9, 6.4, 5.4, 0.3, 10, bold=True, color=GRAY, font=MONO)
add_text(slide, "WHERE IT LOOKED", 7.0, 6.4, 5.4, 0.3, 10, bold=True, color=GRAY, font=MONO)

# ---------------------------------------------------------------- 09 results
slide = base(
    9, TOTAL,
    "08 · RESULTS",
    "It learned the shape. Not the future.",
    "Chronological holdout · 676 real charts the model never saw.",
    "Holdout number is optimistic: the checkpoint was trained before the split fix.",
)
add_table(
    slide,
    [
        ["WHAT", "NUMBER"],
        ["Shape accuracy", "94% (synthetic)"],
        ["Direction accuracy", "47.2%"],
        ["AUC (one vs rest)", "0.564"],
        ["Macro F1", "0.356"],
        ["Random guess", "33.3%"],
    ],
    0.9, 3.0, 5.5, 3.4,
    col_widths=[3.5, 2.0],
    font_size=11,
)
add_image_fit(slide, ASSETS / "confusion_gray.png", 7.0, 2.9, 5.2, 3.5)
add_text(slide, "CONFUSION MATRIX", 7.0, 6.45, 5.2, 0.3, 9,
         bold=True, color=GRAY, font=MONO)

# ---------------------------------------------------------------- 10 verdict
slide = base(
    10, TOTAL,
    "09 · VERDICT",
    "Use it to see, not to trade.",
    "The heat map lands on the pattern, not on tomorrow.",
    "Kept isolated - never feeds the main model. Next: clean retrain, then decide.",
)
add_image_fit(slide, ASSETS / "chart_input.png", 1.0, 2.9, 5.2, 3.0)
add_image_fit(slide, ASSETS / "gradcam_gray.png", 7.1, 2.9, 5.2, 3.0)
add_text(slide, "CHART", 1.0, 6.0, 5.2, 0.3, 9, bold=True, color=GRAY, font=MONO)
add_text(slide, "ATTENTION", 7.1, 6.0, 5.2, 0.3, 9, bold=True, color=GRAY, font=MONO)

prs.save(OUT)
print("wrote", OUT, "slides", len(prs.slides))
