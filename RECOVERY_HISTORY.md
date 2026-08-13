# RECOVERY_HISTORY — universal_bci_hypnosis 项目恢复历史

> 整理日期：2026-08-04
> 范围：`E:/universal_bci_hypnosis`（主项目）与 `E:/universal_bci_hypnosis_rerun`（重跑验证副本）
> 目的：系统重装后，从磁盘恢复本项目并梳理论文 / 版本 / 备份 / 投稿归档状态。

---

## 0. 关键恢复信号

- `universal_bci_hypnosis_rerun/rerun_deterministic.log` 内日志路径指向 **`H:\universal_bci_hypnosis_rerun\...`**，而当前目录位于 **E:**。
- 结论：重装后项目从原 **H: 盘**迁移 / 恢复至 **E:** 盘（与系统重装后本地项目丢失、需重新恢复的路径变更相符）。
- 云端对话历史检索（`conversation_search`，关键词 universal_bci_hypnosis / BCI催眠 / 重跑等）**未命中本项目专项对话**，仅返回 AIRI i18n、小说 v8.1 两条无关记录。故本文件基于磁盘文件还原。

---

## 1. 项目身份

| 项 | 内容 |
|---|---|
| 仓库 | `E:/universal_bci_hypnosis`（主项目）+ `E:/universal_bci_hypnosis_rerun`（重跑验证副本） |
| 主题 | 跨域 8 数据集 EEG “催眠深度” 三分类（**proxy label**，明确不主张临床催眠深度） |
| 稿件标题 | *Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels* |
| 投往期刊 | **PLOS ONE** |
| 作者 | Weng Zexiao（一作，ORCID 0009-0009-8600-8954）/ Jung Minpo（通讯，Youngsan University） |
| 代码源 | GitHub `korose523/Hypnotise` |

---

## 2. 版本与结果演进（README 版本表）

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v2.1 | 2026-05-14 | 锁定 63 维特征、LODO/LOSO/LOO、bootstrap CI |
| v5.0 | 2026-06-02 | 8 数据集多源 LODO + MAHNOB 真实标签 |
| v5.2 | 2026-06-03 | 单遍校验、ds006437 标签泄漏修复 |
| v6.4 | 2026-06-18 | 修复 MAHNOB/SEED/SEED_IV 真实被试级 split；ds006437 事件相位标签；单一可复现 runner |
| **v6.5** | **2026-06-19** | **定稿**：ds004572 全 52 被试、Mahalanobis WFSC(exp103)、EEGNet-v4(exp104)、SHAP |

**最终 headline 结果（v6.5，160/160 实验）**：Zero-Shot **40.02% ± 12.61** → Calibration **43.93% ± 13.53**（+3.91pp）。
增益几乎全部来自 ds006437（29.05% → 60.18%），其余目标 balanced accuracy 仍接近 chance —— 稿件在 §4.4 / §9 透明披露了该局限。

**核心配置**：14 通道 EPOC+、63 维谱特征（14×3 频段 log-bandpower + 7 不对称对×3 DASM）、8 数据集共 712,832 窗口（697,906 有效标签）、MAX_SRC=MAX_TGT=8000、n_estimators=200、20 seeds。

---

## 3. 磁盘上的恢复 / 备份痕迹（2026-07-23）

| 文件 | 说明 |
|---|---|
| `multi_8ds_master_oldbackup_20260723.json` | 主结果 JSON 的 7/23 前备份 |
| `paper_final_md_bak_20260723_mitig.md` | 稿件“缓解版”改作的 7/23 备份 |
| `paper_final_md_bak_20260723_reviewfix.md` | 稿件“审稿修订版”的 7/23 备份 |
| `paper_final_zh.md` | 中文版稿件 |
| `universal_bci_hypnosis_rerun/` | 7/23 当晚两次**确定性重跑**（约 19:47、20:16 起），验证 exp101 可复现 |

---

## 4. 投稿 / 归档状态（2026-07-24 收尾）

| 项 | 状态 |
|---|---|
| **Zenodo 归档** | ✅ 已发布：概念 DOI `10.5281/zenodo.21531272`，版本记录 `10.5281/zenodo.21531273`（v1.0.0） |
| **IRB 豁免** | `YSUIRB-202607-HR-219-02`（Youngsan Univ IRB，2026-07-22） |
| OSF 草稿 | 备用方案，**未使用**（保留为备选，`osf_archive_draft.md`） |
| Figshare 草稿 | 备用方案，**未使用**（保留为备选，`figshare_archive_draft.md`） |
| 投稿配套 | `cover_letter.md`、`pre_submission_checklist.md`、`em_compliance_answers.md`、`em_file_upload_map.md`、`reporting_checklist_S1.pdf`（STROBE / TRIPOD-AI） |
| 接稿后脚本 | `add_paper_doi_to_zenodo.py`（接受后回填论文正式 DOI + 撤销 token） |

---

## 5. 关键文件索引

| 文件 | 用途 |
|---|---|
| `README.md` | 项目说明 + 版本历史 + 最终实验结果表 |
| `paper_final.md` | 英文定稿稿件（v6.5） |
| `paper_final_zh.md` | 中文版稿件 |
| `config.yaml` | 实验统一配置 |
| `run_exp101_reproducible.py` | 单一可复现 runner，生成 `multi_8ds.json` |
| `run_exp101_v2_mitigation.py` | 标签崩塌缓解（FACED 排除 / SMOTE / 类别加权） |
| `run_exp104_eegnet_reproducible.py`、`run_exp104_v2_focal.py` | EEGNet-v4 基线（含 focal loss γ=2.0） |
| `scripts/exp103_wfsc_dynamic_mahalanobis_vs_fixedw.py` | Mahalanobis 加权 WFSC 基准 |
| `analyze_shap_rf.py` | SHAP 特征重要性诊断 |
| `reprocess_ds006437_event_labels.py`、`reprocess_ds004572.py`、`fix_mahnob_labels.py`、`repair_subject_ids.py` | 数据集预处理与标签恢复（含 MAHNOB `feltArsl` 从 `session.xml` 恢复） |
| `results/` | 全部实验输出（LODO/LOSO、WFSC、EEGNet、SHAP、v2 缓解） |
| `processed/`、`splits/` | 衍生、去标识化预处理特征 / 标签矩阵与逐被试 split |
| `multi_8ds_master.json` | 主结果 JSON（160/160 实验，事件相位 ds006437 + 全 52 被试 ds004572） |

---

## 6. 后续建议

1. 核对当前 `paper_final.md` 与 `multi_8ds_master.json` 是否和已发 Zenodo(v1.0.0) 一致；若稿件有改动需重新归档并升版。
2. 接稿后运行 `add_paper_doi_to_zenodo.py --paper-doi <正式DOI>`，并撤销本次 Zenodo token。
3. 用 `git log` 复核提交历史，确认恢复完整性（当前 `.git` 存在，提交明细尚未逐条核对）。
