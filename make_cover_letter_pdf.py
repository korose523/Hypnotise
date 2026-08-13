import re

md = "E:/universal_bci_hypnosis/cover_letter.md"
out = "E:/universal_bci_hypnosis/cover_letter.pdf"
title = "Multi-source domain generalization with few-shot calibration for cross-dataset EEG state classification under proxy labels"

raw = open(md, encoding="utf-8").read()
# Normalize unicode dashes/quotes to ASCII to avoid font-encoding issues
norm_map = {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u00a0": " "}
for k, v in norm_map.items():
    raw = raw.replace(k, v)

lines_src = [l.rstrip() for l in raw.splitlines()]

# Isolate the exact title on its own line so it stays a contiguous substring
quoted = '"' + title + '"'
for i, l in enumerate(lines_src):
    if quoted in l:
        before, after = l.split(quoted, 1)
        lines_src[i:i+1] = [before.strip(), quoted, after.strip()]
        break

def wrap(text, width):
    words = text.split()
    out_lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            out_lines.append(cur); cur = w
    if cur:
        out_lines.append(cur)
    return out_lines

WIDTH = 95
rendered = []
for l in lines_src:
    if l == "":
        rendered.append("")
    elif l == quoted:
        rendered.append(l)  # title kept intact on one line (rendered at small font)
    else:
        rendered.extend(wrap(l, WIDTH))

# Build content stream; render the isolated title line at 9pt so it stays on ONE line
# (contiguous, detectable exact title) while the rest uses 11pt.
stream = ["BT", "54 788 Td"]
LEADING = 15
prev_size = None
for ln in rendered:
    if ln == "":
        stream.append("0 -8 Td")
        continue
    size = 7 if ln == quoted else 11
    if size != prev_size:
        stream.append("/F1 %d Tf" % size)
        prev_size = size
    esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.append("(%s) Tj" % esc)
    stream.append("0 -%d Td" % LEADING)
stream.append("ET")
stream_bytes = ("\n".join(stream)).encode("latin-1", "replace")

# Assemble PDF
obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
obj3 = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
obj4 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>"
obj5 = b"<< /Length %d >>\nstream\n" % len(stream_bytes) + stream_bytes + b"\nendstream"

pdf = b"%PDF-1.4\n"
offsets = {}
for num, body in [(1, obj1), (2, obj2), (3, obj3), (4, obj4), (5, obj5)]:
    offsets[num] = len(pdf)
    pdf += b"%d 0 obj\n" % num + body + b"\nendobj\n"
xref_pos = len(pdf)
n = 6
pdf += b"xref\n0 %d\n" % n
pdf += b"0000000000 65535 f \n"
for num in [1, 2, 3, 4, 5]:
    pdf += b"%010d 00000 n \n" % offsets[num]
pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % n
pdf += b"startxref\n%d\n%%%%EOF" % xref_pos

with open(out, "wb") as f:
    f.write(pdf)
print("WROTE", out, "bytes:", len(pdf))

# Verify with pypdf
from pypdf import PdfReader
txt = "\n".join((p.extract_text() or "") for p in PdfReader(out).pages)
print("pypdf opened OK. title present:", title in txt)
if title in txt:
    i = txt.find(title)
    print("context:", repr(txt[i-3:i+40]))
