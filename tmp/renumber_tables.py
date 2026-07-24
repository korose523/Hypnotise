#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renumber all main-body tables sequentially (Table 1..11) + appendix Table B1,
and add captions to the 3 previously-uncaptioned body tables + appendix table.
Updates in-text citations accordingly. Idempotent on already-numbered captions.
"""
import re
SRC = r"H:/universal_bci_hypnosis/paper_final.md"
lines = open(SRC, encoding="utf-8").read().split("\n")

# Step 1: renumber existing "Table X" (X 2..8) -> X+1 (captions + in-text), highest first
def renum(line):
    return re.sub(r'Table (2|3|4|5|6|7|8)\b',
                  lambda m: f'Table {int(m.group(1))+1}', line)
new_lines = [renum(l) for l in lines]

# Step 2: insert captions (bottom-to-top to preserve indices)
# find indices in new_lines
def find(head_pat):
    for i, l in enumerate(new_lines):
        if re.search(head_pat, l):
            return i
    return -1

ins = []  # (index_after, caption_text)
i_cls = find(r'^### Classifier configuration')
i_lim = find(r'^### Limitations \(comprehensive\)')
i_mit = find(r'^### Label collapse and FACED mitigation \(v2\)')
i_app = find(r'^## Appendix B: Dataset label sources detail')

assert i_cls >= 0 and i_lim >= 0 and i_mit >= 0 and i_app >= 0, (i_cls,i_lim,i_mit,i_app)

ins.append((i_app, "**Table B1. Dataset label-source summary.**"))
ins.append((i_mit, "**Table 11. FACED-excluded mitigation results (SMOTE + exclude FACED, 5 seeds).**"))
ins.append((i_lim, "**Table 10. Summary of limitations and mitigation status.**"))
ins.append((i_cls, "**Table 2. Random Forest classifier configuration.**"))

# insert from highest index to lowest
for idx, cap in sorted(ins, key=lambda x: -x[0]):
    new_lines.insert(idx + 1, cap)

open(SRC, "w", encoding="utf-8").write("\n".join(new_lines))
print("Renumber done.")
print(f"classifier@{i_cls} limitations@{i_lim} mitigation@{i_mit} appendix@{i_app}")
# verify
caps = [l for l in new_lines if re.match(r'\*\*(Table \d+\..+?|Table B\d+\..+?)\*\*', l)]
print(f"captioned tables: {len(caps)}")
for c in caps:
    print("  ", c[:75])
