# Supplementary File S1 — STROBE / TRIPOD+AI-Style Reporting Checklist

This checklist adapts the **STROBE** statement (observational studies) and the
**TRIPOD+AI** checklist (prediction-model reporting) to the cross-dataset
machine-learning benchmarking design of this study. Items are mapped to the
sections of `paper_final.md`.

| # | Reporting item | Addressed in | Status |
|---|----------------|--------------|--------|
| 1 | Study design explicitly described (LODO/LOSO benchmarking) | §1, §3.2 | ✅ |
| 2 | Data sources & eligibility, public repositories named | §3.1, §8.2 | ✅ |
| 3 | **Proxy (non-validated) labels** explicitly declared & derived | §3.1 | ✅ |
| 4 | Participant-level (not trial-level) partitioning to avoid leakage | §3.2, §4.2 | ✅ |
| 5 | Handling of missing / degenerate data disclosed | §3.5, §4.4 | ✅ |
| 6 | No selective omission of underperforming datasets | §3.5 | ✅ |
| 7 | Predictors / features described (EEG bands, windows) | §2, §3 | ✅ |
| 8 | Model specification (RF, EEGNet, calibration) | §3.3–§3.4 | ✅ |
| 9 | Model performance metrics (Acc, Macro-F1, BAcc, κ) with rationale | §3.2, Table 2 | ✅ |
| 10 | Uncertainty: 20-seed 95% CIs reported | §4.1 | ✅ |
| 11 | **Multiple-comparison control**: Holm–Bonferroni over 8 targets | §4.1 | ✅ |
| 12 | Significance testing (Wilcoxon signed-rank) described & corrected | §4.1 | ✅ |
| 13 | Model limitations (label collapse, class imbalance) discussed | §4.4–§4.5, §5 | ✅ |
| 14 | Reproducibility: single-script regeneration, fixed seeds | §3.5, §8.1 | ✅ |
| 15 | Code availability (public repo + DOI plan) | §8.1 | ✅ |
| 16 | Data availability (raw accessions + derived intermediates) | §8.2 | ✅ |
| 17 | Ethics / IRB exemption stated | §6 | ✅ |
| 18 | Funding / Competing interests / Author contributions declared | §7 (Author Statements) | ✅ |

**Notes**
- STROBE and TRIPOD+AI are reporting frameworks, not study-design mandates;
  this study is a computational secondary analysis, so items are adapted
  (e.g., "participants" = dataset contributors; "follow-up" = cross-domain
  generalization rather than longitudinal follow-up).
- All quantitative claims are reproducible from `run_exp101_reproducible.py`
  and the deposited result JSONs.
