#!/usr/bin/env python3
"""Render plos_human_participants_checklist.md -> plos_human_participants_checklist_S2.pdf (Supplementary File S2)."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Flowable)


class Checkbox(Flowable):
    """Draws a real checkbox (square outline with a check) — font-independent."""
    def __init__(self, size=4 * mm, checked=True):
        Flowable.__init__(self)
        self.size = size
        self.checked = checked

    def wrap(self, *a):
        return (self.size, self.size)

    def draw(self):
        c = self.canv
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        c.rect(0, 0, self.size, self.size, stroke=1, fill=0)
        if self.checked:
            c.setLineWidth(1.3)
            c.setStrokeColor(colors.black)
            # check mark: two diagonal strokes inside the box
            c.line(self.size * 0.22, self.size * 0.48,
                   self.size * 0.42, self.size * 0.24)
            c.line(self.size * 0.42, self.size * 0.24,
                   self.size * 0.82, self.size * 0.80)

SRC = r"E:/universal_bci_hypnosis/plos_human_participants_checklist.md"
OUT = r"E:/universal_bci_hypnosis/plos_human_participants_checklist_S2.pdf"

with open(SRC, encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f]

title = ""
intro = []
table_rows = []
notes = []
confirms = []
state = "pre"
for ln in lines:
    if ln.startswith("# "):
        title = ln[2:].strip()
        state = "intro"
    elif ln.strip().startswith("|") and state in ("intro", "table"):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        table_rows.append(cells)
        state = "table"
    elif ln.strip().startswith("- ["):
        confirms.append(re.sub(r"^\s*-\s*\[[ xX]\]\s*", "", ln.strip()))
        state = "confirms"
    elif ln.strip().startswith("-") and state in ("table", "notes", "confirms"):
        notes.append(ln.strip()[1:].strip())
        state = "notes"
    elif ln.strip() == "":
        continue
    else:
        if state == "intro":
            intro.append(ln)

ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Heading1"], fontSize=14, spaceAfter=8)
SUB = ParagraphStyle("SUB", parent=ss["BodyText"], fontSize=9.5, leading=13,
                     alignment=TA_LEFT, spaceAfter=4, textColor=colors.HexColor("#444444"))
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9.5, leading=13,
                      alignment=TA_LEFT, spaceAfter=6)
CELL = ParagraphStyle("CELL", parent=ss["BodyText"], fontSize=8.5, leading=11)
CELLH = ParagraphStyle("CELLH", parent=CELL, fontName="Helvetica-Bold")

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=18*mm, bottomMargin=18*mm)
flow = []
flow.append(Paragraph(title, H))
for p in intro:
    if p.strip():
        flow.append(Paragraph(p, SUB))
flow.append(Spacer(1, 4))

header = table_rows[0]
data = [[Paragraph(h, CELLH) for h in header]]
for row in table_rows[1:]:
    data.append([Paragraph(c, CELL) for c in row])
col_w = [10*mm, 70*mm, 92*mm]
tbl = Table(data, colWidths=col_w, repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f9")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
flow.append(tbl)
flow.append(Spacer(1, 8))
flow.append(Paragraph("<b>Confirmations</b>", BODY))
for c in confirms:
    row = Table([[Checkbox(), Paragraph(c, BODY)]],
                colWidths=[6 * mm, (A4[0] - 36 * mm - 6 * mm)])
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 1.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(row)

doc.build(flow)
print("PDF written:", OUT)
