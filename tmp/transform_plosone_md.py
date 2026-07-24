#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform paper_final.md into PLOS ONE manuscript-body format.

Produces a PLOS ONE-compliant markdown (still the source of truth) that the
docx converter will consume. Keeps all numbers/tables/data intact.
"""
import re, io

SRC = r"H:/universal_bci_hypnosis/paper_final.md"

with open(SRC, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ---- 1. Title (sentence case) ----
NEW_TITLE = "# Multi-source domain generalization with few-shot calibration for cross-dataset EEG hypnosis depth classification under proxy labels"

# ---- 2. Heading rename maps (strip number + sentence case) ----
L1 = {
    "## 1. Introduction": "## Introduction",
    "## 2. Related Work": "## Related work",
    "## 3. Methods": "## Materials and methods",
    "## 4. Results": "## Results",
    "## 5. Discussion": "## Discussion",
    "## 6. Ethics Statement": "## Ethics statement",
    "## 7. Author Statements": None,  # REMOVE whole section
    "## 8. Data and Code Availability": "## Data and code availability",
    "## 9. Conclusion": "## Conclusions",
    "## References": "## References",
    "## Appendix A: Experiment Configuration": "## Appendix A: Experiment configuration",
    "## Appendix B: Dataset Label Sources Detail": "## Appendix B: Dataset label sources detail",
}
L2 = {
    "### 2.1 EEG Domain Generalization": "### EEG domain generalization",
    "### 2.2 Hypnosis Depth Classification": "### Hypnosis depth classification",
    "### 2.3 MAHNOB-HCI Self-Assessment Recovery": "### MAHNOB-HCI self-assessment recovery",
    "### 3.1 Datasets": "### Datasets",
    "### 3.2 Experimental Protocol": "### Experimental protocol",
    "### 3.3 Classifier Configuration": "### Classifier configuration",
    "### 3.4 Calibration Strategy": "### Calibration strategy",
    "### 3.5 Data Integrity Statement": "### Data integrity statement",
    "### 4.1 Multi-Source LODO Performance": "### Multi-source LODO performance",
    "### 4.2 Key Observations": "### Key observations",
    "### 4.3 ds006437 Label Leakage: Resolved": "### ds006437 label leakage: resolved",
    "### 4.4 Per-Class Recall and Label Collapse": "### Per-class recall and label collapse",
    "### 4.5 Mahalanobis vs. Fixed-Weight WFSC (exp103)": "### Mahalanobis vs. fixed-weight WFSC (exp103)",
    "### 4.6 EEGNet-v4 Deep-Learning Baseline (exp104)": "### EEGNet-v4 deep-learning baseline (exp104)",
    "### 4.7 SHAP Feature Importance (analyze_shap_rf)": "### SHAP feature importance (analyze_shap_rf)",
    "### 4.8 MAHNOB Label Quality: Both Gain and Loss": "### MAHNOB label quality: both gain and loss",
    "### 5.1 Overall Performance and Label Collapse": "### Overall performance and label collapse",
    "### 5.2 Domain Generalization Baselines": "### Domain generalization baselines",
    "### 5.3 Split-Unit Contamination (Resolved in v6.1)": "### Split-unit contamination (resolved in v6.1)",
    "### 5.4 Calibration Effectiveness": "### Calibration effectiveness",
    "### 5.5 DG Baselines: CORAL, AdaBN, TCA": "### DG baselines: CORAL, AdaBN, TCA",
    "### 5.6 DREAMER Class-0 Absence": "### DREAMER class-0 absence",
    "### 5.7 Limitations (Comprehensive)": "### Limitations (comprehensive)",
    "### 5.8 Label Collapse and FACED Mitigation (v2)": "### Label collapse and FACED mitigation (v2)",
    "### 5.9 Future Work": "### Future work",
    "### 8.1 Code and scripts": "### Code and scripts",
    "### 8.2 Data": "### Data",
    "### 8.3 Reporting Compliance": "### Reporting compliance",
}

# ---- 3. Vancouver references (hardcoded from full author lists) ----
VANCOUVER = """[1] Zheng Y, Wu S, Chen J, Yao Q, Zheng S. Cross-subject motor imagery electroencephalogram decoding with domain generalization. Bioengineering. 2025;12(5):495. doi:10.3390/bioengineering12050495
[2] Imtiaz MN, Khan N. Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation. Comput Biol Med. 2025;184:109394. doi:10.1016/j.compbiomed.2024.109394
[3] Zhang X, Zheng W, Cai H, Li Z, Yang Y, Liu W. Prompt-guided domain generalization for EEG emotion recognition. IEEE Trans Affect Comput. 2026;17(2):1968-1984. doi:10.1109/TAFFC.2026.3658346
[4] Obukhov NV, Naish PL, Solnyshkina IE, Siourdaki TG, Martynov IA. Real-time assessment of hypnotic depth, using an EEG-based brain-computer interface: a preliminary study. BMC Res Notes. 2023;16:288. doi:10.1186/s13104-023-06553-2
[5] Jensen MP, Adachi T, Hakimian S. Brain oscillations, hypnosis, and hypnotizability. Am J Clin Hypn. 2015;57(3):230-253. doi:10.1080/00029157.2014.976786
[6] Soleymani M, Lichtenauer J, Pun T, Pantic M. A multimodal database for affect recognition and implicit tagging. IEEE Trans Affect Comput. 2012;3(1):42-55. doi:10.1109/T-AFFC.2011.25
[7] Sun B, Feng J, Saenko K. Return of frustratingly easy domain adaptation. Proc AAAI Conf Artif Intell. 2016;30(1). doi:10.1609/aaai.v30i1.10306
[8] Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H, Laviolette F, Marchand M, Lempitsky V. Domain-adversarial training of neural networks. J Mach Learn Res. 2016;17(1):2096-2030.
[9] Gretton A, Borgwardt KM, Rasch MJ, Scholkopf B, Smola A. A kernel two-sample test. J Mach Learn Res. 2012;13:723-773.
[10] Li D, Yang Y, Song Y, Hospedales TM. Learning to generalize: meta-learning for domain generalization. Proc AAAI Conf Artif Intell. 2018;32(1). doi:10.1609/aaai.v32i1.11775
[11] Koelstra S, Muhl C, Soleymani M, Lee JS, Yazdani A, Ebrahimi T, Pun T, Nijholt A, Patras I. DEAP: a database for emotion analysis using physiological signals. IEEE Trans Affect Comput. 2012;3(1):18-31. doi:10.1109/T-AFFC.2011.15
[12] Katsigiannis S, Ramzan N. DREAMER: a database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices. IEEE J Biomed Health Inform. 2017;22(3):98-107. doi:10.1109/JBHI.2017.2776951
[13] Zheng WL, Lu BL. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks. IEEE Trans Auton Ment Dev. 2015;7(3):162-175. doi:10.1109/TAMD.2015.2431497
[14] OpenNeuro. EEG correlates of hypnotic depth and suggestion effects (ds004572) [Data set]. 2024. https://openneuro.org/datasets/ds004572
[15] OpenNeuro. LIGHT hypnotherapy (ds006437) [Data set]. 2024. https://openneuro.org/datasets/ds006437"""

out = []
i = 0
n = len(lines)
author_block_done = False
removed_author_section = False
refs_replaced = False
ack_added = False
support_added = False

while i < n:
    raw = lines[i]
    stripped = raw.rstrip("\n")
    # ---- Title ----
    if stripped.startswith("# ") and not stripped.startswith("## "):
        out.append(NEW_TITLE + "\n\n")
        i += 1
        # skip author block until we hit the first "## Abstract" or a "---" right after
        # consume following lines until "## Abstract"
        while i < n and not lines[i].strip().startswith("## Abstract"):
            i += 1
        # ensure one blank line then Abstract (handled by next iteration)
        continue
    # ---- Keywords line removal ----
    if stripped.startswith("**Keywords**:"):
        i += 1
        continue
    # ---- References section: replace content until "## Appendix A" (BEFORE L1 rename) ----
    if stripped == "## References":
        # insert Acknowledgments just before References
        if not ack_added:
            ack = ("\n## Acknowledgments\n\n"
                   "We thank the creators and maintainers of the public EEG datasets used in this study "
                   "(MAHNOB-HCI, DEAP, SEED/SEED-IV, DREAMER, FACED, and OpenNeuro ds004572/ds006437) for "
                   "making their data openly available, and the developers of the open-source tools "
                   "(scikit-learn, MNE-Python, SHAP, and EEGNet) that underpin our analysis. This research "
                   "was conducted under the exemption granted by the Youngsan University Institutional "
                   "Review Board (YSUIRB-202607-HR-219-02).\n")
            out.append(ack)
            ack_added = True
        out.append("## References\n")
        i += 1
        # skip old reference lines
        while i < n and not lines[i].strip().startswith("## Appendix A"):
            i += 1
        # Now append Vancouver references
        out.append("\n" + VANCOUVER + "\n")
        refs_replaced = True
        # After references, before Appendix A, insert Supporting information
        if not support_added:
            sup = ("\n## Supporting information\n\n"
                   "S1 File. STROBE/TRIPOD-AI-style reporting checklist. A completed checklist mapping each "
                   "STROBE and TRIPOD-AI item to the relevant section of this manuscript "
                   "(`reporting_checklist.md`).\n\n")
            out.append(sup)
            support_added = True
        # continue into Appendix A loop (i now at "## Appendix A")
        continue
    # ---- Heading exact rename ----
    if stripped in L1:
        if L1[stripped] is None:
            # remove whole "## 7. Author Statements" section up to next "## 8."
            i += 1
            while i < n and not lines[i].strip().startswith("## 8. Data and Code Availability"):
                i += 1
            removed_author_section = True
            continue
        else:
            out.append(L1[stripped] + "\n")
            i += 1
            continue
    if stripped in L2:
        out.append(L2[stripped] + "\n")
        i += 1
        continue
    # ---- Table caption reformat: **Table N: X** -> **Table N. X.** ----
    m = re.match(r"\*\*(Table \d+): (.+?)\*\*", stripped)
    if m:
        out.append(f"**{m.group(1)}. {m.group(2)}.**\n")
        i += 1
        continue
    out.append(raw)
    i += 1

with open(SRC, "w", encoding="utf-8") as f:
    f.writelines(out)

print("Transformed paper_final.md")
print(f"  author_section_removed={removed_author_section}")
print(f"  refs_replaced={refs_replaced} ack_added={ack_added} support_added={support_added}")
