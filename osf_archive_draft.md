# OSF Archive Draft — `korose523/Hypnotise`

> **用途**：本文件为向 OSF（Open Science Framework）注册项目归档（代码 + 预处理中间结果 + 结果 JSON）时所需的描述与元数据草稿。OSF 同样被 PLOS ONE 认可，注册后可生成 DataCite DOI。
> **状态**：草稿。DOI 占位符 `osf.io/XXXX` 在 OSF 注册后回填到 `paper_final.md` §8.1。
> **前置**：仓库根目录已有 MIT `LICENSE`（commit `2a7004d` 已加入）。OSF 注册时会要求选 License，选 MIT 即可。

---

## 1. OSF 网页表单填写速查

| 字段 | 填写值 |
|---|---|
| 项目标题 (Title) | Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels |
| 描述 (Description) | 见下方 §2（可整段粘贴；支持 Markdown） |
| 类别 (Category) | Software / Analysis (选 "Software" 或 "Data") |
| 标签 (Tags) | EEG; domain generalization; proxy labels; cross-dataset; calibration; brain-computer interface; affective computing; machine learning benchmarking; few-shot learning; hypnotic depth |
| 贡献者 (Contributors) | **Weng, Zexiao** — Department of Computer Engineering, Youngsan University; ORCID: 0009-0009-8600-8954 |
| License | **MIT** |
| 可见性 (Visibility) | **Public** |
| 关联 DOI (DOI) | 注册后由 OSF 自动生成（DataCite），回填 §8.1 |
| 关联链接 (Links) | 见 §3（GitHub 源 + 各数据集） |

> **OSF 建仓步骤提示**：
> 1. 新建 Project → 填标题/描述 → 设 Public。
> 2. 用 "Add component" 或直接在 Files 里 **拖拽上传**本仓库内容（代码 + results/ + reporting_checklist.md + processed/ + splits/）。
> 3. 在 "Metadata" 里填 Tags、Contributors、License。
> 4. 点 **"Register"**（或 "Create DOI"）生成可引用归档版本与 DOI。
> 5. 把 DOI（`osf.io/XXXX`）回填 `paper_final.md` §8.1。

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

## 3. Related links / identifiers

| Relation | Type | Identifier | 说明 |
|---|---|---|---|
| **isSupplementTo / source** | URL | `https://github.com/korose523/Hypnotise` | 持续更新的代码源 |
| isDocumentedBy | DOI/URL | *[论文接受后填 DOI]* | 关联正式发表稿件 |
| references | DOI | `10.11922/sciencedb.01214` | FACED 数据集 (Science Data Bank) |
| references | URL | `https://openneuro.org/datasets/ds004572` | OpenNeuro ds004572 |
| references | URL | `https://openneuro.org/datasets/ds006437` | OpenNeuro ds006437 |
| references | URL | `https://mahnob.dump.unitn.it/` | MAHNOB-HCI |
| references | URL | `https://www.eecs.qmul.ac.uk/mmv/datasets/deap/` | DEAP |
| references | URL | `http://bcmi.sjtu.edu.cn/home/seed/` | SEED / SEED-IV |
| references | URL | `http://dreamer.ecs.soton.ac.uk/` | DREAMER |

> OSF 的 "Links" 区可逐条添加上述 URL；论文正式 DOI 接受后作为 "isDocumentedBy" 关联。

---

## 4. Notes（归档备注，纯文本）

```
IRB exemption No. YSUIRB-202607-HR-219-02 (Youngsan University IRB, 2026-07-22).
Secondary analysis of de-identified public EEG data; no new participants recruited.
Labels are PROXIES (task-condition / event-phase / arousal self-report), NOT validated
clinical hypnosis-depth scores. Balanced accuracy remains at chance; see manuscript §4.4/§9.
Raw EEG recordings are NOT included (large + original licenses); only derived, de-identified
preprocessed matrices (processed/, splits/) are redistributed. Obtain raw data from sources in Related links.
Versioned archive; GitHub repo is the continuously updated source of record.
```

---

## 5. 建议引用格式（How to cite）

> Weng Zexiao (2026). *Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels* (v1.0.0) [Data set]. OSF. https://osf.io/XXXX

**BibTeX**
```bibtex
@misc{weng2026hypnotise,
  author = {Weng, Zexiao},
  title  = {Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels},
  year   = {2026},
  version = {v1.0.0},
  publisher = {OSF},
  url    = {https://osf.io/XXXX},
  note   = {Code: https://github.com/korose523/Hypnotise}
}
```

---

## 6. 落地行动清单（归档前）

- [x] 仓库根目录已有 `LICENSE`（MIT，commit `2a7004d`）
- [ ] 在 OSF 新建 Public 项目，上传仓库内容（代码 + results/ + processed/ + splits/ + reporting_checklist.md）
- [ ] 填 Tags / Contributors / License（MIT）
- [ ] Register 生成 DOI（`osf.io/XXXX`），回填 `paper_final.md` §8.1 占位符
- [ ] （接受后）把论文正式 DOI 加为 isDocumentedBy 关联
