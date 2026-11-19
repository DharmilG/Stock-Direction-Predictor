"""Add the real head/class breakdown to slide 7 of the existing deck.

The model has TWO heads. The number six is the pattern head's class count.
Placed in the free area under the bullets on the right side of slide 7.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

P = Path(__file__).resolve().parent / "Chart_Vision_Review.pptx"
prs = Presentation(P)
slide = prs.slides[6]  # slide 7 (Two heads. Two questions.)

GRAY = RGBColor(110, 110, 110)
MONO = "Consolas"


def add_text(text, left, top, width, height, size, *, bold=False, color=None, align=PP_ALIGN.LEFT):
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
    run.font.name = MONO
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    return box


add_text("2 HEADS - THE 6 ARE PATTERN CLASSES", 8.1, 6.02, 4.4, 0.22,
         8.5, bold=True, color=GRAY)
add_text("random / head & shoulders / double top", 8.1, 6.26, 4.4, 0.18, 8)
add_text("double bottom / bull flag / triangle", 8.1, 6.45, 4.4, 0.18, 8)
add_text("direction: UP / DOWN / FLAT", 8.1, 6.64, 4.4, 0.18, 8)

prs.save(P)
print("patched slide 7")
