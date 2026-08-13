# PLOS ONE — Editorial Manager 合规问卷标准答案

> 用途：投稿（Editorial Manager）走到 **"Compliance with PLOS ONE's policies"** 问卷页时，逐题照抄下方「标准回复」即可。
> 所有内容均与 `paper_final.md` §6–§8、§7 声明一致，可直接粘贴。
> 占位符：`<PAPER_DOI>` = 论文正式接收后的 DOI（接受前留空，接受后用 `add_paper_doi_to_zenodo.py` 回填 Zenodo）。

---

## A. 伦理 / 人类被试（Ethics & Human Subjects）

**Q1. Did this study involve human participants / human data?**
> 标准回复：Yes. This study involved de-identified, publicly available human EEG data (no new participant recruitment, no identifiable information accessed, no intervention performed).

**Q2. Ethics approval / IRB — Was approval obtained?**
> 标准回复：This manuscript reports secondary analysis of de-identified, publicly available data only. The authors' institutional IRB (Youngsan University IRB / 영산대학교 생명윤리위원회) granted exemption for this secondary analysis under IRB exemption review (심의면제, confirmed 2026-07-22), exemption No. **YSUIRB-202607-HR-219-02**, on the basis that the study involves secondary use of publicly available, de-identified data with no identifiable private information obtained by the investigators (equivalent to 45 CFR 46.104(d)(4)). Each original dataset was collected under the ethical oversight of its creating institution with documented informed-consent procedures (MAHNOB-HCI: University of Trento; DEAP: Queen Mary University of London; SEED/SEED-IV: Shanghai Jiao Tong University; DREAMER: University of Malta / Imperial College London; FACED: Tsinghua University; OpenNeuro ds004572/ds006437: original institutions' IRB approvals under open-data licenses).

**Q3. Informed consent — Was it obtained?**
> 标准回复：Informed consent was obtained by the original data collectors at the time of primary data collection; we relied solely on those pre-approved, de-identified public releases. No new consent was required for this exempt secondary analysis.

**Q4. Animal research?**
> 标准回复：No.

---

## B. 临床试验注册（Clinical Trial Registration）

**Q5. Is this a clinical trial? Was it registered?**
> 标准回复：No — this is an observational/computational secondary analysis of public EEG data, not a clinical trial or interventional study, and is not prospectively registered (no registration applies).

---

## C. 数据 / 代码可用性（Data & Code Availability）

**Q6. Data availability — Is data available?**
> 标准回复：Yes — Available in a public repository.
> - Raw EEG datasets: publicly available from their original repositories (MAHNOB-HCI, DEAP, SEED/SEED-IV, DREAMER, FACED, OpenNeuro ds004572, OpenNeuro ds006437).
> - Derived, de-identified preprocessed matrices and per-participant splits: deposited in the public GitHub repository https://github.com/korose523/Hypnotise under `processed/` and `splits/`.
> - Permanent versioned archive with DOI: Zenodo software archive (v1.0.12), https://doi.org/10.5281/zenodo.21531272 (concept DOI; v1.0.12 versioned record https://doi.org/10.5281/zenodo.21922749).

**Q7. Code availability — Is code available?**
> 标准回复：Yes — Available in a public repository. Complete source code, experiment scripts, configuration files, and result JSONs are in the public GitHub repository https://github.com/korose523/Hypnotise (MIT license). A single script, `run_exp101_reproducible.py`, regenerates the main results with fixed parameters (MAX_SRC=8000, MAX_TGT=8000, n_estimators=200, 20 seeds).

---

## D. 报告指南（Reporting Guidelines）

**Q8. Did you use a reporting guideline? Which, and where is the checklist?**
> 标准回复：Yes. We followed the **STROBE** statement (observational reporting) and the **TRIPOD-AI** checklist (AI prediction-model reporting), adapted for a cross-dataset ML benchmarking design. The completed checklist is uploaded as **Supplementary File S1** (`reporting_checklist.md` / `reporting_checklist_S1.pdf`).

---

## E. 作者 / ORCID / 贡献（Authors, ORCID, Contributions）

**Q9. ORCID — Provide ORCID for corresponding author (required) and all authors.**
> 标准回复：Corresponding author: **Jung Minpo** (ORCID: to be added). First author ORCID: **0009-0009-8600-8954** (Weng Zexiao). All available ORCID provided in the manuscript.

**Q10. Authors' contributions — Confirm CRediT roles declared.**
> 标准回复：Yes. Weng Zexiao: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Resources, Software, Validation, Visualization, Writing – original draft, Writing – review & editing. Jung Minpo: Supervision, Writing – review & editing.

**Q11. Sex and gender reporting — How were sex/gender variables handled?**
> 标准回复：Sex/gender characteristics of the source datasets are reported where disclosed by the original collectors; we declare in §3.1 that the three-class state labels are *proxies* derived from task conditions / event-phase markers / arousal self-reports, and that no validated clinical hypnotic-depth score was available. Sex/gender were not used as predictive features in this benchmarking study and are discussed as a limitation of proxy-labeled secondary data.

---

## F. 基金 / 利益冲突（Funding & Competing Interests）

**Q12. Funding — Was funding received?**
> 标准回复：The authors received no specific funding for this work.

**Q13. Competing interests — Declared?**
> 标准回复：The authors have declared that no competing interests exist.

---

## G. 原创性 / 重叠发表（Originality & Overlap）

**Q14. Is the work original and not under consideration elsewhere?**
> 标准回复：Yes — this manuscript is original, has not been published previously, and is not under consideration at any other journal. All authors have approved the manuscript and agree with its submission to PLOS ONE.

---

## H. 接受后动作（Post-acceptance, not asked at submission）

**H1. Link the published paper to the Zenodo record (run after acceptance):**
```bash
export ZENODO_TOKEN="<your_token>"
python add_paper_doi_to_zenodo.py --paper-doi <PAPER_DOI>
```
**H2. Revoke the Zenodo token** used during archival (account → Settings → Applications/Tokens).

---

### 速查卡（可直接整段复制提交时备注）

```
Ethics: Exempt secondary analysis; Youngsan University IRB exemption No. YSUIRB-202607-HR-219-02 (2026-07-22).
Consent: Obtained by original collectors; relied on de-identified public releases.
Animals: No. Clinical trial: No (observational/computational secondary analysis).
Data: Public repositories — GitHub https://github.com/korose523/Hypnotise + Zenodo DOI 10.5281/zenodo.21531272.
Code: Public GitHub repository (above), MIT license, reproducible runner run_exp101_reproducible.py.
Reporting: STROBE + TRIPOD-AI checklist uploaded as Supplementary File S1.
ORCID (corresponding): Jung Minpo — to be added. First author ORCID: 0009-0009-8600-8954 (Weng Zexiao).
Funding: No specific funding received. Competing interests: None declared.
Originality: Original, not previously published, not under consideration elsewhere; all authors approve.
```
