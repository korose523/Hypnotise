# Zenodo Archive Description Draft — `korose523/Hypnotise`

> **用途**：本文件为向 Zenodo 注册软件归档（软件 + 预处理中间结果）时所需的「描述（Description）」与元数据草稿。可直接粘贴到 Zenodo 网页表单，或用于 Zenodo REST API（`/api/deposit/depositions`）批量建仓。
> **状态**：✅ 已发布（2026-07-24）。DOI 已回填 `paper_final.md` §8.1。Concept DOI `10.5281/zenodo.21531272`，版本记录 `10.5281/zenodo.21531273`（https://zenodo.org/record/21531273）。
> **归档前必做**：仓库根目录目前**没有 LICENSE 文件**——Zenodo 开放获取要求声明许可证。建议先 `git add` 一个 MIT `LICENSE` 文件再归档（下方元数据已默认 `license: mit`，可按需改为 `cc-by-4.0` 等）。

---

## 1. Zenodo 网页表单填写速查

| 字段 | 填写值 |
|---|---|
| Upload type | **Software** |
| Title | Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels |
| Authors / Creators | **Weng, Zexiao** — Department of Computer Engineering, Youngsan University, Yangsan, Republic of Korea; ORCID: https://orcid.org/0009-0009-8600-8954<br>**Jung, Minpo** — Department of Computer Engineering, Youngsan University, Yangsan, Republic of Korea (Corresponding Author) |
| Description | 见下方 §2（可整段粘贴；支持 HTML，此处用纯文本/Markdown 亦可） |
| Keywords | EEG; domain generalization; proxy labels; cross-dataset; calibration; brain-computer interface; affective computing; machine learning benchmarking; few-shot learning; hypnotic depth |
| License | **MIT** （若选 CC，则 `cc-by-4.0`） |
| Access right | **Open** |
| Version | `v1.0.0` （或填归档时的 git commit 短哈希，如 `41115b0`） |
| Language | English |
| Publication date | 建议填投稿/接受日期，或留空默认归档日（2026-07-24） |
| Related identifiers | 见 §3 |
| Notes | 见 §4 |
| Communities | 可选：如 `plosone`、`eeg`；留空亦可 |
| Grants / Funding | 无（论文 §7 已声明 no specific funding） |

---

## 2. Description（归档描述正文，可整段粘贴）

**Repository.** This archive is the versioned software and results companion to the manuscript *"Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels"*. The continuously updated source of record is the public GitHub repository https://github.com/korose523/Hypnotise.

**Scope.** The deposit contains the complete source code, experiment scripts, configuration files, and result JSON files for a multi-source domain generalization study of **proxy-labeled** cross-dataset EEG state classification spanning 8 publicly available EEG datasets (712,832 total windows; 697,906 valid labeled). The framework aligns heterogeneous datasets to a common 14-channel EPOC+ representation with 63-dimensional spectral features, trains Random Forest classifiers on 7 diverse source domains, and evaluates zero-shot transfer versus few-shot target-domain calibration (20% subject samples via sample concatenation) across 8 held-out target domains.

**Contents.**
- `run_exp101_reproducible.py` — single reproducible runner regenerating the main LODO results (`multi_8ds.json`) from preprocessed data with fixed parameters (MAX_SRC=8000, MAX_TGT=8000, n_estimators=200, 20 seeds).
- `run_exp101_v2_mitigation.py` — label-collapse mitigation (FACED exclusion, SMOTE oversampling, class weighting).
- `run_exp104_eegnet_reproducible.py` and `run_exp104_v2_focal.py` — EEGNet-v4 baselines (incl. focal loss γ=2.0).
- `scripts/exp103_wfsc_dynamic_mahalanobis_vs_fixedw.py` — Mahalanobis-weighted WFSC benchmark.
- `analyze_shap_rf.py` — SHAP feature-importance diagnostics.
- `reprocess_ds006437_event_labels.py`, `reprocess_ds004572.py`, `fix_mahnob_labels.py`, `repair_subject_ids.py` — dataset preprocessing and label recovery (incl. MAHNOB `feltArsl` arousal recovery from `session.xml`).
- `shared/mahalanobis_wfsc.py` — Mahalanobis WFSC implementation (fits calibration only, no test leakage).
- `results/` — all experimental outputs (LODO/LOSO JSON, WFSC benchmark, EEGNet, SHAP summaries, v2 mitigation results).
- `reporting_checklist.md` — STROBE / TRIPOD-AI-style reporting checklist (Supplementary S1).
- `processed/` and `splits/` — derived, de-identified preprocessed feature/label matrices and per-participant train/calibration/test splits.

**Reproduction.** From the repository root, `python run_exp101_reproducible.py` regenerates the headline LODO results. All raw EEG recordings are obtained from the original public sources listed in §3; only the derived intermediate matrices are redistributed here, as the raw files are large (tens of GB) and governed by the original data-use licenses.

**Important caveats (proxy labels).** The three-class Awake/Light/Deep labels are **proxies** derived from task conditions, event-phase markers, or arousal self-reports, and are **not** validated clinical hypnosis-depth scores. Aggregate calibration gains are driven primarily by a single domain (ds006437) and reflect a majority-class flip rather than genuine cross-domain three-class discrimination (balanced accuracy remains at chance). The archive documents these limitations transparently; see the manuscript §4.4 and §9.

**Ethics.** This is a secondary analysis of de-identified, publicly available data. The authors' institutional IRB (Youngsan University IRB, 영산대학교 생명윤리위원회) granted exemption for this secondary analysis (exemption No. **YSUIRB-202607-HR-219-02**, 2026-07-22). No new participants were recruited and no identifiable information was accessed.

---

## 3. Related identifiers（Zenodo `related_identifiers`）

| Relation | Type | Identifier | 说明 |
|---|---|---|---|
| **isSupplementTo** (or isSourceOf) | URL | `https://github.com/korose523/Hypnotise` | 持续更新的代码源 |
| isDocumentedBy | DOI/URL | *[论文接受后填 DOI]* | 关联正式发表稿件 |
| references | DOI | `10.11922/sciencedb.01214` | FACED 数据集 (Science Data Bank) |
| references | URL | `https://openneuro.org/datasets/ds004572` | OpenNeuro ds004572 |
| references | URL | `https://openneuro.org/datasets/ds006437` | OpenNeuro ds006437 |
| references | URL | `https://mahnob.dump.unitn.it/` | MAHNOB-HCI |
| references | URL | `https://www.eecs.qmul.ac.uk/mmv/datasets/deap/` | DEAP |
| references | URL | `http://bcmi.sjtu.edu.cn/home/seed/` | SEED / SEED-IV |
| references | URL | `http://dreamer.ecs.soton.ac.uk/` | DREAMER |

> 注：OpenNeuro 数据集体也可改用其 DOI 形式（如 `10.18112/openneuro.ds004572`）。若 Zenodo 校验报错，改用上面 URL 形式（identifier_type: `url`）即可。

---

## 4. Notes（Zenodo `notes` 字段，纯文本）

```
IRB exemption No. YSUIRB-202607-HR-219-02 (Youngsan University IRB, 2026-07-22).
Secondary analysis of de-identified public EEG data; no new participants recruited.
Labels are PROXIES (task-condition / event-phase / arousal self-report), NOT validated
clinical hypnosis-depth scores. Balanced accuracy remains at chance; see manuscript §4.4/§9.
Raw EEG recordings are NOT included (large + original licenses); only derived, de-identified
preprocessed matrices (processed/, splits/) are redistributed. Obtain raw data from sources in Related identifiers.
Versioned archive; GitHub repo is the continuously updated source of record.
```

---

## 5. Zenodo REST API 元数据草稿（可选，用于脚本建仓）

```json
{
  "metadata": {
    "upload_type": "software",
    "title": "Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels",
    "creators": [
      {
        "name": "Weng, Zexiao",
        "affiliation": "Department of Computer Engineering, Youngsan University, Yangsan, Republic of Korea",
        "orcid": "0009-0009-8600-8954",
        "email": "wengzexiao@office.ysu.ac.kr"
      }
    ],
    "description": "<p>Repository companion to the manuscript on multi-source domain generalization for proxy-labeled cross-dataset EEG state classification across 8 public EEG datasets. Contains reproducible source code, experiment scripts, configuration, and result JSON files. The three-class Awake/Light/Deep labels are proxies (task-condition / event-phase / arousal self-report), NOT validated clinical hypnosis-depth scores; balanced accuracy remains at chance. Raw EEG is obtained from original sources; only derived de-identified preprocessed matrices are redistributed. IRB exemption YSUIRB-202607-HR-219-02.</p>",
    "keywords": [
      "EEG", "domain generalization", "proxy labels", "cross-dataset", "calibration",
      "brain-computer interface", "affective computing", "machine learning benchmarking",
      "few-shot learning", "hypnotic depth"
    ],
    "license": "mit",
    "access_right": "open",
    "version": "v1.0.0",
    "language": "eng",
    "publication_date": "2026-07-24",
    "notes": "IRB exemption YSUIRB-202607-HR-219-02 (Youngsan University IRB, 2026-07-22). Secondary analysis of de-identified public EEG data. Labels are PROXIES, not validated clinical hypnosis-depth scores; balanced accuracy at chance. Raw EEG not included; only derived preprocessed matrices redistributed.",
    "related_identifiers": [
      { "relation": "isSupplementTo", "identifier": "https://github.com/korose523/Hypnotise", "identifier_type": "url" },
      { "relation": "references", "identifier": "10.11922/sciencedb.01214", "identifier_type": "doi" },
      { "relation": "references", "identifier": "https://openneuro.org/datasets/ds004572", "identifier_type": "url" },
      { "relation": "references", "identifier": "https://openneuro.org/datasets/ds006437", "identifier_type": "url" }
    ]
  }
}
```

---

## 6. 建议引用格式（How to cite）

> Weng Zexiao, & Jung Minpo (2026). *Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21531272

**BibTeX**
```bibtex
@software{weng2026hypnotise,
  author = {Weng, Zexiao and Jung, Minpo},
  title  = {Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels},
  year   = {2026},
  version = {v1.0.0},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.21531272},
  url    = {https://github.com/korose523/Hypnotise}
}
```

---

## 7. 落地行动清单（归档前）

- [ ] 在仓库根目录添加 `LICENSE`（建议 MIT），`git add` + commit + push
- [ ] 在 Zenodo 建仓（网页或 API），粘贴 §2 描述与 §1 表单
- [ ] 回填真实 DOI 到 `paper_final.md` §8.1 占位符
- [ ] （可选）将 ORCID 填入本草稿与稿件署名
- [ ] （接受后）把论文正式 DOI 加为 `isDocumentedBy` related identifier
