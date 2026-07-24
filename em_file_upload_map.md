# PLOS ONE Editorial Manager — 文件上传映射表

> 用途：在 Editorial Manager (EM) 的 "Upload Files" 步骤中，系统会要求你为每个文件指定角色（Item Type）。本表列出你需要上传的全部文件及其在 EM 中的对应角色，照表拖拽/选择即可。

---

## 文件上传清单

| # | 本地文件路径 | EM Item Type (角色) | 说明 |
|---|---|---|---|
| 1 | `C:/Users/Administrator/Downloads/paper_en_submission_v2.docx` | **Main Manuscript** | 投稿正文（含 Abstract / Introduction / Methods / Results / Discussion / References）。PLOS ONE 要求 Main Manuscript **去标识**（不含作者名/单位/致谢），这些信息放在 Title Page |
| 2 | 需另存一份 **Title Page**（见下方说明） | **Title Page** | 含标题、全部作者、单位系所、通讯作者邮箱、ORCID、CRediT 贡献、Funding、COI 声明。PLOS ONE 要求 Title Page **单独上传**，不与 Main Manuscript 合并 |
| 3 | `H:/universal_bci_hypnosis/cover_letter.md` → 导出为 PDF | **Cover Letter** | 1 段式投稿信（可选但建议） |
| 4 | `H:/universal_bci_hypnosis/reporting_checklist_S1.pdf` | **Supplementary File** | Supplementary File S1（STROBE/TRIPOD-AI 报告清单） |
| 5 | — (如有独立图片文件) | **Figure** | 如论文有独立图片（位图 ≥300 dpi / 矢量），逐个上传；若图片已嵌入 docx 则可跳过，但 PLOS ONE 生产阶段会要求独立文件 |

---

## Title Page 制作说明

PLOS ONE 要求 Title Page 与 Main Manuscript 分离。你的 `paper_final.md` 的作者信息（§标题下署名 + §7 Author Statements）需要提取为一个独立的 Title Page 文件。

**最快方式**：从 `paper_en_submission_v2.docx` 另存一份，删除正文部分，只保留以下内容：

```
Title:
Multi-Source Domain Generalization with Few-Shot Calibration for
Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels

Authors:
Weng Zexiao¹, Jung Minpo¹*

Affiliations:
¹ Department of Computer Engineering, Youngsan University,
Yangsan, Republic of Korea

Corresponding Author:
Jung Minpo*
Email: minpo@ysu.ac.kr

Authors' Contributions (CRediT):
Weng Zexiao: Conceptualization, Data curation, Formal analysis,
Investigation, Methodology, Resources, Software, Validation,
Visualization, Writing – original draft, Writing – review & editing.
Jung Minpo: Supervision, Writing – review & editing.

Funding:
The authors received no specific funding for this work.

Competing Interests:
The authors have declared that no competing interests exist.
```

存为 `title_page.docx`，在 EM 中作为 **Title Page** 上传。

---

## Main Manuscript 去标识检查

上传前确认 `paper_en_submission_v2.docx` 的正文部分**不含**：
- ❌ 作者姓名（Weng Zexiao 等）
- ❌ 单位名称（Youngsan University 等）
- ❌ 致谢中的可识别信息
- ❌ 文件属性中的作者名（右键 → 属性 → 详细信息 → 清除作者）

正文中可以保留的内容：
- ✅ "the author" / "we"（第一人称泛指）
- ✅ IRB 豁免号（YSUIRB-202607-HR-219-02）
- ✅ 数据/代码链接（GitHub / Zenodo DOI）
- ✅ ORCID（放在 Title Page，不放正文）

> ⚠️ **当前 `paper_en_submission_v2.docx` 的正文仍含作者署名信息**。投稿前需要做一份去标识版本：另存为 `manuscript_blinded.docx`，删除作者块和 §7 Author Statements（把 Funding/COI/CRediT 移到 Title Page），保留正文其余部分。

---

## EM 上传后的合规问卷对应

上传文件后，EM 会弹出合规问卷（见 `em_compliance_answers.md` 逐题标准回复）。问卷完成后提交即进入技术审查（QC）。

---

## 文件上传后的流程

```
Upload Files → 合规问卷 → Build PDF for Review → 确认 PDF → Submit
```

EM 会自动生成一个合并 PDF 供你预览——检查标题页、正文、表格、S1 是否都正确拼接，确认后点 Submit。
