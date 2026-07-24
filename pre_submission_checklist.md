# PLOS ONE 最终投稿前 Checklist — `korose523/Hypnotise`

> **稿件**：*Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification under Proxy Labels*
> **通讯作者**：Weng Zexiao — Department of Computer Engineering, Youngsan University｜wengzexiao@office.ysu.ac.kr｜ORCID 0009-0009-8600-8954
> **Zenodo 归档**：概念 DOI `10.5281/zenodo.21531272`（v1.0.0 记录 `10.5281/zenodo.21531273`，https://zenodo.org/record/21531273）
> **IRB 豁免**：YSUIRB-202607-HR-219-02（Youngsan University IRB，2026-07-22）
> 模拟评审结论：需小修 7.2/10，全部 blocker 与中低优先项已闭环。

勾选每一项后再点 **Submit Manuscript**。

---

## A. 稿件内容
- [ ] 标题含关键词与 "under Proxy Labels" 限定
- [ ] 摘要含 7 数据集 headline 增益 **+ 多数类学习 caveat**（不过度外推）
- [ ] IMRaD 结构完整（引言/方法/结果/讨论）
- [ ] §3.1 Sex and Gender Reporting 段已就位
- [ ] Table 2 表头为 `mean ± SD; 95% CI`，FACED 行标 ⚠️
- [ ] 统计段 Overall p 仅声明 ds006437 稳健 + ds004572 效应量，effective m=5 脚注
- [ ] §8.1 DAS 已回填 Zenodo 概念 DOI `10.5281/zenodo.21531272`
- [ ] 结论（§9）明确 proxy-label 局限、不主张临床催眠深度
- [ ] 全文已扫包容性语言（participants 而非 subjects）

## B. 作者与署名
- [ ] 作者块：姓名 + 完整机构（Department of Computer Engineering, Youngsan University）+ 邮箱 + ORCID
- [ ] CRediT 作者贡献（§7）已列
- [ ] 通讯作者 ORCID `0009-0009-8600-8954` 已填（PLOS ONE 强制）

## C. 伦理与合规声明（§7）
- [ ] Funding 声明（no specific funding）
- [ ] Competing Interests 声明（无亦须写 "none"）
- [ ] Authors' Contributions（CRediT）
- [ ] IRB 豁免号 YSUIRB-202607-HR-219-02（§6 逐数据集）

## D. 数据 / 代码可用性
- [ ] DAS 指向公开仓库：GitHub `korose523/Hypnotise` + Zenodo `10.5281/zenodo.21531272`
- [ ] 代码公开（MIT LICENSE 已就位）
- [ ] 原始数据 accession 列全（§8.2：FACED / OpenNeuro ds004572·ds006437 / MAHNOB / DEAP / SEED·SEED-IV / DREAMER）
- [ ] 清楚说明 raw EEG 不随包分发、仅衍生矩阵（避免误述）

## E. 报告指南
- [ ] STROBE / TRIPOD-AI 清单已导出 `reporting_checklist_S1.pdf`
- [ ] 投稿时作为 **Supplementary File S1** 上传

## F. 投稿信
- [ ] `cover_letter.md`（1 段式）已备，含合规声明与原创性确认

## G. Editorial Manager 上传文件分角色
- [ ] **Main Manuscript**（去标识，不含作者名）—— 用 docx
- [ ] **Title Page**（含作者/机构/贡献）
- [ ] **Figures**（独立，≥300 dpi / 矢量）
- [ ] **Supplementary File S1** = `reporting_checklist_S1.pdf`
- [ ] **Cover Letter**
- [ ] （可选）其他补充材料

## H. 系统合规问卷（必答）
- [ ] 伦理审查？是 → IRB 号 YSUIRB-202607-HR-219-02
- [ ] 临床试验注册？否（非 RCT）
- [ ] 数据可用性？是（公开仓库）→ 填 GitHub + Zenodo 链接
- [ ] 报告指南？勾 STROBE + TRIPOD-AI，上传 checklist
- [ ] ORCID / COI / Funding 逐项确认

## I. 归档后动作（接受后）
- [ ] 论文接受后，运行 `add_paper_doi_to_zenodo.py --paper-doi <正式DOI>` 把论文 DOI 加为 Zenodo `isDocumentedBy`
- [ ] **Revoke 本次使用的 Zenodo token** 并换新（安全）
- [ ] 最终通读校样（proofs），确认单位系所 / 通讯邮箱无误

---

## 快速复核命令
```bash
# 本地校验 Zenodo 记录仍可访问、DOI 正确
python - <<'PY'
import requests
r = requests.get("https://zenodo.org/api/records/21531273", timeout=20)
print(r.status_code, r.json().get("doi"))
PY

# 论文接受后补加正式 DOI（先 dry-run 看计划）
export ZENODO_TOKEN="<your_token>"
python add_paper_doi_to_zenodo.py --paper-doi 10.1371/journal.pone.XXXXXXXX --dry-run
python add_paper_doi_to_zenodo.py --paper-doi 10.1371/journal.pone.XXXXXXXX
```
