"""Insert a project flow-diagram slide at position 2 of the existing deck.

Preserves every existing slide exactly (including in-PowerPoint edits).
No eyebrow/kicker text is added, matching the user's edited deck.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

P = Path(__file__).resolve().parent / "Chart_Vision_Review.pptx"
prs = Presentation(P)
BLANK = prs.slide_layouts[6]

BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(110, 110, 110)
WHITE = RGBColor(255, 255, 255)
FONT = "Aptos"
MONO = "Consolas"


def add_text(slide, text, left, top, width, height, size, *, bold=False, color=None,
             font=FONT, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    return box


def add_rule(slide, left, top, width, *, color=BLACK, height=0.015):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp


def add_box(slide, title, lines, left, top, width, height):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = WHITE
    shp.line.color.rgb = BLACK
    shp.line.width = Pt(1.2)
    frame = shp.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.12)
    frame.margin_right = Inches(0.08)
    frame.margin_top = Inches(0.08)
    frame.margin_bottom = Inches(0.05)
    p = frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.name = FONT
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = BLACK
    for line in lines:
        p2 = frame.add_paragraph()
        p2.space_after = Pt(1)
        r2 = p2.add_run()
        r2.text = line
        r2.font.name = MONO
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = BLACK
    return shp


# --- build the new slide (appended first, then moved to position 2) ----------
s = prs.slides.add_slide(BLANK)
add_text(s, "17", 0.6, 0.24, 1.6, 0.3, 9, color=GRAY, font=MONO)
add_text(s, "CHART VISION · CV LANE", 9.6, 0.24, 3.1, 0.3, 9,
         color=GRAY, font=MONO, align=PP_ALIGN.RIGHT)
add_rule(s, 0.6, 0.62, 12.13)
add_text(s, "How the project works", 0.6, 0.88, 12.1, 0.85, 30, bold=True)
add_text(s, "Data in  →  features  →  model  →  decision  →  dashboard", 0.6, 1.72, 12.1, 0.4,
         13, color=GRAY)

boxes = [
    ("1  COLLECT", ["6 data feeds", "prices, filings, news,", "Bhavcopy, GDELT,", "calendar maps", "~16 years · 50 stocks"]),
    ("2  PREPARE", ["60-candle charts", "190 features", "triple-barrier labels", "UP / FLAT / DOWN", "time-based split"]),
    ("3  ANALYSE", ["FinBERT reads text", "LightGBM predicts", "ResNet18 reads charts", "softmax → confidence", "calibration"]),
    ("4  DECIDE", ["threshold check", "confidence ≥ 0.60", "backtest with costs", "G1 / G2 / G3 gates", "PASS / FAIL / PAUSE"]),
    ("5  SHOW", ["FastAPI read-only", "React dashboard", "Chart Vision page", "answer + heat map", "ledger + verdict"]),
]
x = 0.7
for i, (title, lines) in enumerate(boxes):
    add_box(s, title, lines, x, 2.55, 2.25, 2.35)
    x += 2.45
    if i < len(boxes) - 1:
        add_text(s, "→", x - 0.22, 3.45, 0.24, 0.4, 18, bold=True, align=PP_ALIGN.CENTER)

add_text(s, "Track Q is a separate offline test (Pool C → Qwen3 8B → verdict). It never feeds the main model.",
         0.7, 5.35, 12.0, 0.35, 11, bold=True, color=GRAY)
add_text(s, "The same 224 x 224 chart image is used when training and when a user uploads a picture.",
         0.7, 5.78, 12.0, 0.3, 10, color=GRAY)
add_rule(s, 0.6, 6.98, 12.13, color=GRAY)

# --- move the new slide from the end to position 2 --------------------------
sldIdLst = prs.slides._sldIdLst
new_id = sldIdLst[-1]
sldIdLst.remove(new_id)
sldIdLst.insert(1, new_id)

prs.save(P)
print("inserted flow slide at position 2; total slides:", len(prs.slides))
