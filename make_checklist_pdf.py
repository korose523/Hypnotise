#!/usr/bin/env python3
"""Render reporting_checklist.md -> reporting_checklist_S1.pdf (Supplementary File S1)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

SRC = r"E:/universal_bci_hypnosis/reporting_checklist.md"
OUT = r"E:/universal_bci_hypnosis/reporting_checklist_S1.pdf"

with open(SRC, encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f]

# --- parse ---
title = ""
intro = []
table_rows = []
notes = []
state = "pre"
for ln in lines:
    if ln.startswith("# "):
        title = ln[2:].strip()
        state = "intro"
    elif ln.strip().startswith("|") and state in ("intro", "table"):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # skip separator row (only -, :, spaces)
        if set("".join(cells)) <= set("-: "):
            continue
        # map check mark emoji -> text for font safety
        cells = [c.replace("✅", "Yes") for c in cells]
        table_rows.append(cells)
        state = "table"
    elif ln.strip().startswith("-") and state in ("table", "notes"):
        notes.append(ln.strip()[1:].strip())
        state = "notes"
    elif ln.strip() == "":
        continue
    else:
        if state in ("intro",):
            intro.append(ln)

# --- styles ---
ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Heading1"], fontSize=14, spaceAfter=8)
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
        flow.append(Paragraph(p, BODY))
flow.append(Spacer(1, 4))

# table
header = table_rows[0]
data = [[Paragraph(h, CELLH) for h in header]]
for row in table_rows[1:]:
    data.append([Paragraph(c, CELL) for c in row])
col_w = [12*mm, 92*mm, 38*mm, 18*mm]
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
flow.append(Paragraph("<b>Notes</b>", BODY))
for n in notes:
    flow.append(Paragraph("• " + n, BODY))

doc.build(flow)
print("PDF written:", OUT)
