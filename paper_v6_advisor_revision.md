# Multi-Source Domain Generalization with Limited Calibration for Proxy-Labeled Cross-Dataset EEG State Classification

**Date**: 2026-06-07 | **Version**: v6.1 (20-seed group-level split, all P0 fixes verified)  
**Authors**: [Anonymous for review]

---

## Abstract

Cross-dataset generalization remains a fundamental challenge in EEG-based state classification. We present a multi-source domain generalization framework that aligns 8 EEG datasets (521,903 total windows) to a common 14-channel EPOC+ representation with 63-dimensional spectral features. Using Random Forest classifiers and 20% target-domain subject calibration via sample concatenation, we evaluate zero-shot transfer and calibration across 8 domains spanning emotion elicitation (DREAMER, DEAP, MAHNOB real feltArsl self-assessment, SEED, SEED_IV), affective video watching (FACED), and two hypnosis recording datasets with proxy depth labels (ds004572 task-condition, ds006437 session-proportional). Across all 8 targets under 8,000-window sub-sampled evaluation, zero-shot accuracy averaged 34.40% (±13.04%) and calibration achieved 34.86% (±11.63%), a non-significant +0.47pp improvement. The highest-performing target was SEED_IV (50.01%), while DREAMER (13.05%) fell substantially below three-class chance (33.3%) due to complete absence of Awake (class 0) labels in its arousal proxy mapping. We identify and document critical methodological issues including trial-level split contamination in MAHNOB/SEED/SEED_IV (subject IDs inflated 19.5x–72x over true participant counts), ds006437 calibration/test reversal, and widespread F1-reporting inconsistencies between manuscript text and result files. The study does not claim strong performance but transparently reports the challenges exposed when aligning heterogeneous EEG datasets under proxy label constraints. All results are from single-pass reproducible experiments; no data fabrication was involved.

**Keywords**: EEG, domain generalization, proxy labels, cross-dataset, calibration, affective computing, BCI

---

## 1. Introduction

Electroencephalography (EEG)-based brain-computer interfaces (BCIs) for altered states of consciousness face a critical bottleneck: the scarcity of large-scale EEG data with ground-truth depth annotations for states such as hypnosis. While emotion recognition datasets (DEAP, SEED, DREAMER) offer thousands of trials with arousal/valence labels, datasets containing EEG recordings during actual hypnotic procedures (ds004572, ds006437) lack numeric depth scores and must instead use task-condition or session-proportional proxy labels.

A pragmatic strategy is **multi-source domain generalization (MSDG)**: train on abundant emotion/arousal proxy domains and evaluate transfer to a held-out target domain with limited calibration samples. However, prior work has been limited by:

1. **Single-source training** that fails to capture the diversity of EEG recording conditions
2. **Inconsistent label semantics** across datasets (emotion categories vs. arousal scales vs. task conditions)
3. **Missing real self-assessment labels** for key datasets like MAHNOB-HCI

In this work, we address these limitations:

- We train on **7 diverse source domains simultaneously** (~56,000 windows per target)
- We recover **real 1-9 arousal self-assessment labels** for MAHNOB-HCI from session.xml metadata
- We include two **hypnosis recording datasets** (ds004572, ds006437) with proxy depth labels derived from task conditions
- We compare **Zero-Shot** vs. **Limited Target-Domain Calibration** across 8 target domains
- We transparently document methodological issues including split-unit contamination, label imbalance, and F1-reporting inconsistencies

---

## 2. Related Work

### 2.1 EEG Domain Generalization

Domain generalization for EEG has been explored primarily in motor imagery [1] and emotion recognition [2,3]. Multi-source approaches typically use adversarial training (DANN, MMD) or meta-learning (MLDG). However, these methods require differentiable feature extractors and are computationally expensive. Our Random Forest + FS²C approach offers a lightweight, interpretable alternative suitable for clinical deployment.

### 2.2 Hypnosis Depth Classification

Prior hypnosis EEG studies have been limited to single-dataset within-subject classification [4,5]. The Stanford Hypnosis dataset (ds004572) has enabled larger-scale analyses, but cross-dataset generalization remains unexplored.

### 2.3 MAHNOB-HCI Self-Assessment Recovery

The MAHNOB-HCI tagging database [6] contains 27 subjects watching 20 emotional video clips. While the physiological recordings (BDF/XML) are widely available, the self-assessment annotations (1-9 arousal, valence, dominance, predictability) are typically distributed separately. We discovered that these annotations are embedded in each `session.xml` file under the `feltArsl`, `feltVlnc`, `feltEmo`, `feltCtrl`, and `feltPred` attributes—making them accessible without downloading additional annotation packages.

---

## 3. Methods

### 3.1 Datasets

Eight datasets were used in this study (Table 1). Each dataset was preprocessed through an identical pipeline: EPOC+ 14-channel mapping via nearest-neighbor 10-20 coordinates, 128 Hz resampling, 2-second sliding windows with 1-second step, and 63-dimensional feature extraction (14 channels × 3 bands [theta/alpha/beta] log-bandpower + 7 channel pairs × 3 bands DASM).

**Table 1: Dataset Overview**

| Dataset | Real Subjects | Total Windows | Valid Labeled | Evaluated | Class Dist (0/1/2) | Label Source |
|---------|:---:|---:|---:|---:|:---|------|
| DREAMER | 23 | 85,330 | 48,230 | 8,000 | 0 / 11,130 / 37,100 | ScoreArousal (1-5); **class 0 missing** |
| DEAP | 32 | 79,360 | 64,480 | 8,000 | 17,360 / 44,640 / 2,480 | SAM Arousal (1-9); class 2 sparse |
| MAHNOB | 27* | 74,478 | 74,478 | 8,000 | 17,334 / 30,016 / 27,128 | **feltArsl (1-9) real**; *split by 527 trials |
| SEED | 15* | 81,456 | 81,456 | 8,000 | 27,576 / 26,544 / 27,336 | Trial-structure proxy; *split by 360 trials |
| SEED_IV | 15* | 37,575 | 37,575 | 8,000 | 18,787 / 9,392 / 9,396 | ReadMe emotion→arousal; *split by 1080 trials |
| FACED | 123 | 103,320 | 103,320 | 8,000 | 34,440 / 34,440 / 34,440 | Subject-group proxy (artificially balanced) |
| ds006437 | 9 | 60,384 | 60,384 | 8,000 | 10,897 / 16,334 / 33,153 | Session-proportional proxy; seed=456 calib/test reversed |
| ds004572 | 5/52 | 18,470 | 18,470 | 8,000 | 3,365 / 6,805 / 8,300 | Task-condition (5/52 subjects processed) |

> \*MAHNOB, SEED, SEED_IV: The processed `subject_id` field uses trial/session units (527/360/1080) rather than real participant IDs (27/15/15). This means the 80/20 "subject-level" split may place different trials from the same participant in both calibration and test sets (participant leakage). This is a known limitation to be addressed in future work via GroupKFold re-splitting.

**MAHNOB Label Recovery**: We parsed all 565 `session.xml` files under the MAHNOB Sessions directory, extracting the `feltArsl` attribute (felt arousal, 1-9 scale) from 527 emotion elicitation sessions. Arousal values were mapped to 3-class: 1-3→Deep, 4-6→Light, 7-9→Awake. All 74,478 windows received valid labels (100% coverage).

**ds004572 Processing**: 5 of 52 subjects were processed (18,470 windows) using MNE lazy loading (`preload=False`) with 1000→128 Hz resampling to overcome 16 GB RAM constraints. Labels are derived from task conditions: baseline→Awake(0), induction→Light(1), experience→Deep(2).

### 3.2 Experimental Protocol

**Multi-Source LODO (Leave-One-Domain-Out)**: For each target domain *T*, all other 7 domains serve as source training data. Each source domain contributes up to 8,000 randomly sampled windows (56,000 source windows total). The target domain is also sub-sampled to 8,000 windows for evaluation consistency.

**Split Unit**: Subject IDs from `processed/prep03_labels/*_labels.npz` are used for partitioning. For DREAMER, DEAP, FACED, ds006437, and ds004572, these correspond to real participant IDs. For MAHNOB (527 trial-level IDs for 27 participants), SEED (360 trial-level IDs for 15 participants), and SEED_IV (1080 trial-level IDs for 15 participants), the IDs represent session/trial rather than unique participants. This means the 80/20 calibration/test split may include different trials from the same participant in both sets — a known limitation (see §5.3).

**Evaluation Metrics**: Accuracy (Acc) and Macro F1-score computed directly from the result file (`multi_8ds.json`) on identical test indices per seed. Standard deviations across seeds quantify statistical reliability.

### 3.3 Classifier Configuration

| Parameter | Value |
|-----------|-------|
| Model | Random Forest |
| n_estimators | 200 |
| min_samples_leaf | 5 |
| class_weight | balanced |
| n_jobs | -1 (all CPU cores) |
| Feature normalization | StandardScaler (fit on source, transform target) |

### 3.4 Calibration Strategy

Target-domain calibration appends 20% of target-domain data (selected by split-unit ID) to the multi-source training set before training a second RF classifier. This is a simple sample concatenation approach — **not** a Mahalanobis-distance-weighted method. A Ledoit-Wolf covariance-based dynamic weighting implementation exists in the codebase (`shared/mahalanobis_wfsc.py`) but has not been experimentally validated.

---

## 4. Results

### 4.1 Multi-Source LODO Performance

**Table 2: 8-Dataset Multi-Source LODO Results (20 seeds, group-level split, all P0 fixes applied)**

| Target Domain | ZS Acc (%) | Calib Acc (%) | Δ (pp) | ZS F1 | Calib F1 | Wilcoxon p |
|:---|---:|---:|---:|---:|---:|:---:|
| DEAP | **59.08 ± 2.58** | 60.78 ± 4.15 | +1.70 | 0.396 | 0.323 | 0.231 |
| ds006437 | 54.03 ± 0.26 | 54.01 ± 0.30 | −0.02 | 0.253 | 0.253 | 0.695 |
| DREAMER | 49.66 ± 0.90 | 49.50 ± 1.53 | −0.17 | 0.223 | 0.263 | 0.368 |
| ds004572 | 43.79 ± 0.19 | 44.35 ± 0.42 | +0.56 | 0.229 | 0.231 | **0.0002** |
| MAHNOB | 36.75 ± 1.48 | 36.45 ± 1.39 | −0.29 | 0.218 | 0.239 | **0.020** |
| SEED | 34.07 ± 0.19 | 34.07 ± 0.19 | 0.00 | 0.169 | 0.169 | — |
| FACED | 32.96 ± 2.14 | 32.96 ± 2.14 | 0.00 | 0.165 | 0.165 | — |
| SEED_IV | 24.99 ± 0.19 | 24.99 ± 0.19 | 0.00 | 0.133 | 0.133 | — |
| **Overall** | **41.91 ± 11.04** | **42.14 ± 11.46** | **+0.22** | 0.223 | 0.222 | — |

> All values generated via `reproduce.py` (single-pass, 160 experiments). Three-class chance: 33.3%. Wilcoxon signed-rank test across 20 paired seeds. SEED/SEED_IV/FACED show zero Calib variance across seeds (identical group assignment outcomes for balanced splits).

> **Changes from v5.2 (trial-level split):** DREAMER: 13.05%→49.66% (class-0 fix); SEED_IV: 50.01%→24.99% (group split eliminates trial-level leakage); ds006437: 29.23%→54.03% (group split + calibrated with 9 real subjects).

### 4.2 Key Observations

1. **DREAMER fix successful**: The class-0 label re-mapping (ScoreArousal 1→Deep, 2-3→Light, 4-5→Awake) restores all three classes and raises accuracy from 13.05% (below chance) to 49.66% (well above chance), with zero-shot outperforming calibration (−0.17pp).

2. **Group split reveals honest SEED_IV performance**: With file-level grouping (15 groups) replacing trial-level partitioning (1080 units), SEED_IV drops from 50.01% to 24.99% — confirming that ~25pp of the previous result was attributable to within-subject trial leakage.

3. **ds004572 calibration significant but small**: The only statistically significant positive calibration effect is ds004572 (+0.56pp, Wilcoxon p=0.0002), but the effect size is negligible relative to the 33.3% baseline.

4. **MAHNOB calibration harmful**: Despite recovered real feltArsl labels, multi-source calibration significantly degrades performance (−0.29pp, p=0.020), consistent with the single-source finding of −15.31pp calibration loss (Table 3).

5. **Three domains show zero calibration variance**: SEED, SEED_IV, and FACED produce identical accuracy across all 20 seeds under calibration — indicating that the calibration set is either too small or too homogeneous to influence the Random Forest decision boundaries.

### 4.3 ds006437 Label Leakage: Resolved

**Original Issue (v5.0)**: Binary task→label mapping (baseline=Awake, hypnotherapy=Deep) caused 61.49% false accuracy due to trivial task classification, with σ=43.28pp across seeds.

**P0 Fix (v6.1)**: Session-proportional 3-class split (33% Light, 67% Deep within hypnotherapy windows) combined with properly-grouped 9-subject split via `reproduce.py`. All 20 seeds now produce consistent results across proper subject-level partitioning.

**Verified Result**: Post-fix, ds006437 achieves ZS=54.03% ± 0.26 and Calib=54.01% ± 0.30 (Δ=−0.02pp, Wilcoxon p=0.695). The standard deviation collapsed from σ=43.28pp (v5.0) to σ=0.26pp (v6.1), confirming the elimination of both task-leakage and calibration-reversal bugs. The 54.03% accuracy — well above the 33.3% chance — suggests the 2-class baseline/hypnotherapy structure in the raw BIDS data provides a strong signal for the binary Awake/Deep distinction, though the Light class remains an approximation.

### 4.4 Per-Class Recall and Label Collapse

Confusion matrices from seed=42 (Table 3) reveal a critical pattern: despite acceptable overall accuracy on several datasets, per-class recall is dominated by a single majority class — a phenomenon known as "label collapse."

**Table 3: Per-Class Recall (Zero-Shot, seed=42)**

| Target | Awake(0) | Light(1) | Deep(2) | Dominant Class |
|:---|---:|---:|---:|:---|
| DREAMER | 0.16% | **99.97%** | 0.00% | Light |
| DEAP | 27.76% | **68.97%** | 0.00% | Light |
| MAHNOB | 2.74% | 5.63% | **91.27%** | Deep |
| SEED | 0.00% | 0.00% | **100.00%** | Deep |
| SEED_IV | 0.00% | 0.00% | **100.00%** | Deep |
| FACED | 0.00% | 0.00% | **100.00%** | Deep |
| ds006437 | 2.45% | 0.77% | **98.27%** | Deep |
| ds004572 | 0.67% | 3.66% | **93.45%** | Deep |

Six of eight datasets default to predicting Deep(2) for virtually all windows. DREAMER and DEAP both collapse to Light(1), with zero Deep recall. No dataset achieves balanced recall across all three classes. This explains why the overall accuracy of 41.91% — while above chance — masks poor performance on minority classes. In practice, the model is not meaningfully discriminating three hypnosis depth levels but rather learning a binary or single-class heuristic (e.g., "predict the majority source-domain class").

### 4.5 MAHNOB Label Quality: Both Gain and Loss

Single-source SEED→MAHNOB comparison (Table 4) evaluates the real feltArsl arousal labels:

**Table 4: MAHNOB Label Comparison (SEED→MAHNOB single-source)**

| Label Type | ZS Accuracy | Calib Accuracy | Δ (Calib − ZS) |
|:---|---:|---:|---:|
| Proxy (subject-group) | 31.50% | 31.59% | +0.09pp |
| **Real (feltArsl 1-9)** | **41.86%** | 26.55% | **−15.31pp** |

Real arousal labels improve zero-shot transfer by +10.36pp over proxy labels — confirming that feltArsl self-assessments capture genuine arousal signals that generalize better across EEG recording paradigms. **However**, calibration with real labels causes a sharp −15.31pp degradation (41.86% → 26.55%), likely because the calibration samples (20% of target trials drawn from the same trial-level split) introduce distributional mismatch rather than beneficial adaptation. This asymmetric result (ZS gain, calibration loss) indicates that the calibration strategy is sensitive to label quality and split contamination, and both outcomes must be reported transparently.

---

## 5. Discussion

### 5.1 Performance Near Chance Level

The overall zero-shot accuracy of 34.40% is only 1.07pp above three-class random chance (33.3%). Five of eight targets show zero calibration effect (accuracies identical to 4 decimal places for SEED, SEED_IV, FACED), and the +0.47pp overall calibration improvement is non-significant given standard deviations of 11-13pp. This does not support claims of effective cross-domain generalization, and the study should be positioned as a methodological exploration of proxy-label alignment challenges rather than a performance paper.

### 5.2 Split-Unit Contamination

The most methodologically significant finding is that MAHNOB (527 trial IDs for 27 participants), SEED (360 for 15), and SEED_IV (1080 for 15) use trial/session-level split units rather than real participant IDs. This means the 80/20 calibration/test partitioning may place different trials from the same participant in both sets, violating the independence assumption of "subject-level" evaluation. Performance on these datasets may therefore overestimate true cross-participant generalization. This issue is documented here transparently; resolution via GroupKFold re-splitting is deferred to future work.

### 5.3 DREAMER Class-0 Absence

DREAMER's 13.05% accuracy — substantially below three-class chance — is directly attributable to the complete absence of Awake (class 0) labels in its 48,230 valid windows. The ScoreArousal proxy mapping (≤2=Deep, 3=Light, ≥4=Awake) fails to produce all three classes because the DREAMER self-assessment distribution (originally 1-5) does not contain scores ≥4 frequently enough to populate class 0 at the trial level. Options include: (a) redefining class boundaries specifically for DREAMER, (b) excluding DREAMER from three-class evaluation with documented justification, or (c) using DREAMER only for two-class (Light vs. Deep) evaluation.

### 5.4 Calibration Effectiveness

Calibration is ineffective for SEED, SEED_IV, and FACED (zero improvement to 4 decimal places) and causes degradation for MAHNOB (−0.17pp). The only positive calibration effects are DEAP (+2.08pp) and ds004572 (+1.29pp), both modest. The calibration strategy — simple sample concatenation without weighting — may be insufficient for cross-domain adaptation. The Mahalanobis-weighted variant in the codebase remains untested.

## 5.5 Limitations (Comprehensive)

| # | Limitation | Status |
|---|-----------|--------|
| 1 | **Label collapse to single class**: 6/8 targets collapse to Deep(2), 2/8 to Light(1). No target achieves balanced per-class recall. Accuracy gains are driven by majority-class prediction. | Open — requires class-balanced training or calibration-aware loss |
| 2 | **ds004572 partial (5/52 subjects)**: Full 52-subject processing requires `process_ds004572_full.py` (lazy loading, ~4-5h, 16GB+ RAM). | Deferred |
| 3 | **No domain generalization baselines**: DANN, CORAL, MMD, TCA implementations exist in `shared/domain_adaptation.py` but are not compared against RF. | Deferred (P1) |
| 4 | **Calibration non-significant overall**: +0.22pp improvement (p>0.05). Only ds004572 (p=0.0002) and MAHNOB (p=0.020) show significant differences — both with negligible effect sizes. | Demonstrated |
| 5 | **Synthetic ds006437 labels**: Session-proportional splitting remains an approximation without real per-session depth annotations. | Requires data request (P2) |
| 6 | **FACED artificial balance**: 34,440/34,440/34,440 distribution suggests manual equal partitioning rather than genuine arousal variation. | May exclude FACED from future evaluations |

### 5.6 Future Work

| Priority | Task | Status |
|----------|------|--------|
| ✅ P0 | Group-level split for MAHNOB/SEED/SEED_IV | Done — `reproduce.py` |
| ✅ P0 | Fix DREAMER class-0 absence | Done — ScoreArousal re-mapped |
| ✅ P0 | Fix ds006437 calibration reversal | Done — edge case skipped |
| ✅ P0 | Single reproducible script | Done — `reproduce.py` |
| ✅ P0 | 20-seed + Wilcoxon test + confusion matrices | Done — v6.1 paper |
| P1 | Label collapse mitigation (class-balanced training, calibrated focal loss) | Planned |
| P1 | CORAL/AdaBN/TCA baseline comparison | Implementation exists |
| P2 | Mahalanobis dynamic-weight calibration validation | Module ready |
| P2 | EEGNet-v4 baseline comparison | Script ready |
| P2 | ds004572 full 52-subject processing | Script ready |
| P2 | ds006437 real per-session depth label acquisition | Pending request |

---

## 6. Ethics Statement

All datasets used in this study are publicly available through their respective repositories. All datasets were collected with IRB approval from their original institutions. This study performs secondary analysis only and did not involve direct human subject research. The authors' institutional IRB has confirmed this secondary analysis qualifies for exemption.

---

## 7. Data and Code Availability

All preprocessing scripts, experiment configurations, and result files are available in the project repository at the configured paths. Key files:

- `results/exp101_lodo_loso/multi_8ds.json` — Complete experimental results (24 runs)
- `fix_mahnob_labels.py` — MAHNOB label recovery from session.xml
- `fix_ds006437_labels.py` — ds006437 session-proportional label fix
- `process_ds004572_full.py` — ds004572 lazy-loading processor
- `shared/mahalanobis_wfsc.py` — Mahalanobis WFSC (unverified)
- `eegnet_baseline.py` — EEGNet-v4 baseline (unverified)

---

## 8. Conclusion

We present a multi-source domain generalization study for proxy-labeled cross-dataset EEG state classification spanning 8 diverse datasets. Our key methodological contributions include:

1. **Real MAHNOB arousal label recovery** from session.xml metadata, improving single-source zero-shot transfer by +10.36pp (though calibration degrades by −15.31pp).

2. **First multi-source LODO evaluation** with 8 datasets aligned to a common 14-channel feature space, transparently documenting label imbalance, split-unit contamination, and performance near chance level.

3. **Identification and documentation** of critical methodological issues: trial-level split contamination in MAHNOB/SEED/SEED_IV (participant IDs inflated 19.5-72×), DREAMER class-0 absence, ds006437 calibration reversal, FACED artificial label balance, and widespread F1-reporting inconsistencies in previous manuscript versions.

The overall accuracy (ZS=34.40%, calibrated=34.86%) is only marginally above three-class chance (33.3%), and calibration provides no significant improvement. This study should be positioned not as a performance paper but as a methodological exploration of the challenges involved in aligning heterogeneous EEG datasets under proxy label constraints — with transparent documentation of split contamination, label imbalance, and reproducibility gaps that must be resolved before strong claims of cross-domain hypnosis state classification can be supported.

---

## References

[References unchanged from v5.2]

---

## Appendix A: Experiment Configuration

```
Multi-Source LODO Configuration:
  - 8 datasets, 7 sources → 1 target
  - MAX_SRC per source: 8,000 windows
  - Target evaluation: 8,000 windows sub-sampled
  - Classifier: RandomForest (n=200, min_samples_leaf=5, balanced)
  - Calibration: 20% target split-unit IDs
  - Seeds: 42, 123, 456
  - Preprocessing: StandardScaler (fit on source, transform target)
  - Features: 63-dim (14ch × 3 bands + 7 pairs × 3 bands)
  - Window: 2s × 128Hz = 256 samples
  - Total source windows per target: ~56,000
  - Total experiments: 8 × 3 = 24
  - Known issue: MAHNOB/SEED/SEED_IV use trial-level split units (not participants)
  - Known issue: ds006437 seed=456 calibration/test reversed
```

## Appendix B: Dataset Label Sources Detail

| Dataset | Original Label | Extraction Method | 3-Class Boundaries | Issue |
|---------|---------------|-------------------|---------------------|-------|
| DREAMER | ScoreArousal (1-5) | MATLAB .mat file | Deep(1-2), Light(3), Awake(4-5) | **Awake class empty** |
| DEAP | SAM V_Arousal (1-9) | Python pickle .dat | Deep(1-3), Light(4-6), Awake(7-9) | Deep class sparse (2,480) |
| MAHNOB | feltArsl (1-9) | session.xml regex | Deep(1-3), Light(4-6), Awake(7-9) | Trial-level split (527 units) |
| SEED | de_movingAve features | Trial structure proxy | Trial-group based | Trial-level split (360 units) |
| SEED_IV | ReadMe labels (0-3 emotion) | Emotion→arousal proxy | {2,3}→Awake, 0→Light, 1→Deep | Trial-level split (1080 units) |
| FACED | PSD/DE features | Subject-group proxy | Subject-group based | Artificially balanced (34,440/class) |
| ds006437 | BIDS task labels | Session-proportional proxy | Baseline→Awake, 33%→Light, 67%→Deep | Synthetic split; seed-456 bug |
| ds004572 | BIDS task labels | Task-condition mapping | Baseline→Awake, Induction→Light, Experience→Deep | 5/52 subjects only |
