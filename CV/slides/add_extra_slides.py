"""Append extra slides to the existing Chart Vision deck, preserving edits already made in PowerPoint.

Adds: architecture diagram, diagram explained, data years, data candle/60,
ResNet18 image processing, and GradCAM explanation. Does NOT rebuild the file.
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

prs = Presentation(OUT)
BLANK = prs.slide_layouts[6]
START = len(prs.slides) + 1  # 11 for the current 10-slide deck


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
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_bullets(slide, items, left, top, width, height, *, size=13, gap=7):
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
        run.text = f"—  {line}"
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = BLACK
    return box


def add_rule(slide, left, top, width, *, color=BLACK, height=0.015):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def base(number: int, kicker: str, headline: str, subtitle: str):
    slide = prs.slides.add_slide(BLANK)
    add_text(slide, f"{number:02d}", 0.6, 0.24, 1.6, 0.3, 9, color=GRAY, font=MONO)
    add_text(slide, "CHART VISION · CV LANE", 9.6, 0.24, 3.1, 0.3, 9,
             color=GRAY, font=MONO, align=PP_ALIGN.RIGHT)
    add_rule(slide, 0.6, 0.62, 12.13)
    add_text(slide, kicker, 0.6, 0.82, 12.1, 0.35, 12, bold=True)
    add_text(slide, headline, 0.6, 1.18, 12.1, 0.9, 30, bold=True)
    add_text(slide, subtitle, 0.6, 2.12, 12.1, 0.45, 14, color=GRAY)
    add_rule(slide, 0.6, 6.98, 12.13, color=GRAY)
    return slide


def add_table(slide, rows, left, top, width, height, col_widths=None, font_size=10.5):
    table = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height)
    ).table
    if col_widths:
        for index, value in enumerate(col_widths):
            table.columns[index].width = Inches(value)
    for ri, row in enumerate(rows):
        table.rows[ri].height = Inches(height / len(rows))
        for ci, value in enumerate(row):
            cell = table.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLACK if ri == 0 else (LIGHT if ri % 2 else WHITE)
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = cell.text_frame
            frame.clear()
            run = frame.paragraphs[0].add_run()
            run.text = str(value)
            run.font.name = FONT if ri == 0 else MONO
            run.font.size = Pt(font_size)
            run.font.bold = ri == 0
            run.font.color.rgb = WHITE if ri == 0 else BLACK
    return table


def add_box(slide, text, left, top, width, height, *, size=11):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = BLACK
    shape.line.width = Pt(1.1)
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = BLACK
    return shape


def add_image_fit(slide, path, left, top, width, height):
    with Image.open(path) as image:
        ratio = image.width / image.height
    box_ratio = width / height
    if ratio > box_ratio:
        w, h = width, width / ratio
    else:
        h, w = height, height * ratio
    return slide.shapes.add_picture(str(path), Inches(left + (width - w) / 2),
                                    Inches(top + (height - h) / 2),
                                    width=Inches(w), height=Inches(h))


# ---- S11: architecture diagram (screenshot) --------------------------------
s = base(START, "10 · ARCHITECTURE", "The whole platform, in one picture.",
         "A blueprint of every step - from raw data to the final verdict.")
add_image_fit(s, ASSETS / "architecture_diagram.png", 0.6, 2.5, 12.13, 4.2)
add_text(s, "Saved from the Stitch architectural specification page", 0.6, 6.75, 8, 0.2,
         8, color=GRAY, font=MONO)

# ---- S12: explain the diagram ---------------------------------------------
s = base(START + 1, "11 · ARCHITECTURE", "Read the diagram left to right.",
         "Five lanes. Each lane hands its result to the next.")
add_table(s, [
    ["LANE", "IN PLAIN WORDS"],
    ["1 · Inputs", "6 data feeds: prices, filings, news, reference maps"],
    ["2 · Features", "190 columns, triple-barrier labels, time-based split"],
    ["3 · Models", "FinBERT reads text, LightGBM predicts DOWN/FLAT/UP"],
    ["4 · Decision", "Thresholds, backtest, FastAPI, 8-page dashboard"],
    ["5 · Track Q", "Offline LLM test - isolated, never feeds the main model"],
], 0.9, 3.0, 11.5, 3.4, col_widths=[2.4, 9.1], font_size=11)
add_text(s, "No vector database. No cloud. Local, repeatable steps.", 0.9, 6.55, 10, 0.3,
         11, bold=True, color=GRAY)

# ---- S13: data years -------------------------------------------------------
s = base(START + 2, "12 · DATA", "About sixteen years of prices.",
         "One row per trading day, for 50 NIFTY symbols.")
add_table(s, [
    ["WHAT", "NUMBER"],
    ["Daily prices", "2010-09-15 to 2026-10-28 (~16 years)"],
    ["Chart labels", "2010-12-13 to 2026-10-21"],
    ["Unique label dates", "2,136"],
    ["Real chart images", "3,100"],
    ["Synthetic chart images", "120"],
], 0.9, 3.0, 7.2, 3.5, col_widths=[3.0, 4.2], font_size=11)
add_bullets(s, [
    "Trading days only - weekends and holidays are absent.",
    "50 symbols, one price file each.",
    "Every image is made from this local price data.",
], 8.5, 3.1, 3.9, 3.2, size=12, gap=9)

# ---- S14: one candle and 60 candles ---------------------------------------
s = base(START + 3, "13 · DATA & ARCHITECTURE", "Why 60 candles per image?",
         "One candle is one trading session, not one calendar day.")
add_box(s, "ONE CANDLE\nOPEN · HIGH\nLOW · CLOSE\nVOLUME", 0.9, 2.9, 3.0, 1.9, size=12)
add_text(s, "→", 4.05, 3.5, 0.4, 0.4, 20, bold=True, align=PP_ALIGN.CENTER)
add_box(s, "60 CANDLES\n60 trading sessions\n~81-96 calendar days", 4.55, 2.9, 3.6, 1.9, size=12)
add_text(s, "→", 8.3, 3.5, 0.4, 0.4, 20, bold=True, align=PP_ALIGN.CENTER)
add_box(s, "ONE IMAGE\n224 x 224\nfixed size for ResNet18", 8.85, 2.9, 3.5, 1.9, size=12)
add_bullets(s, [
    "20 sessions estimate volatility (sigma) for the barriers.",
    "60 sessions make the chart. 5 sessions after T make the label.",
    "That is 20 + 60 + 5 = 85 sessions of history per image.",
    "The picture ends at T. Future days are never drawn in it.",
], 0.9, 5.1, 11.5, 1.6, size=12, gap=6)

# ---- S15: ResNet18 processing ---------------------------------------------
s = base(START + 4, "14 · ARCHITECTURE", "How the image moves through ResNet18.",
         "A small window scans the chart. The picture shrinks; the meaning grows.")
steps = [("INPUT", "224x224"), ("CONV1", "112x112"), ("POOL", "56x56"),
         ("L1", "56x56"), ("L2", "28x28"), ("L3", "14x14"), ("L4", "7x7"), ("AVG", "512 nums")]
x = 0.7
for name, dim in steps:
    add_box(s, f"{name}\n{dim}", x, 2.95, 1.4, 0.95, size=10)
    x += 1.5
add_bullets(s, [
    "A filter is a small grid that slides over the picture and writes one number.",
    "Early filters find edges. Deeper ones find candle bodies and shapes.",
    "8 blocks, 8 skip connections (5 identity, 3 projection) keep learning alive.",
    "Average pooling turns 7x7x512 into the final 512 numbers for the heads.",
], 0.9, 4.3, 11.5, 2.3, size=12, gap=7)

# ---- S16: GradCAM explained ------------------------------------------------
s = base(START + 5, "15 · EXPLAINABILITY", "Why GradCAM is in this project.",
         "It shows which part of the chart drove the answer - a reason, not just a number.")
add_image_fit(s, ASSETS / "chart_input.png", 0.9, 2.9, 5.3, 3.0)
add_image_fit(s, ASSETS / "gradcam_gray.png", 7.1, 2.9, 5.3, 3.0)
add_text(s, "CHART", 0.9, 6.0, 5.3, 0.3, 9, bold=True, color=GRAY, font=MONO)
add_text(s, "HEAT MAP", 7.1, 6.0, 5.3, 0.3, 9, bold=True, color=GRAY, font=MONO)
add_bullets(s, [
    "GradCAM looks at the gradients of the last ResNet block (layer4).",
    "Bright areas = pixels that mattered most for the pattern guess.",
    "It helps a human check whether the reason makes sense.",
    "It does not make the model more accurate - it only explains it.",
], 0.9, 6.3, 11.5, 0.6, size=9, gap=2)

prs.save(OUT)
print("appended", len(prs.slides) - (START - 1), "slides -> total", len(prs.slides))
