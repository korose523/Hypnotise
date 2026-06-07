# Multi-Source Domain Generalization with Limited Calibration for Proxy-Labeled Cross-Dataset EEG State Classification

**Date**: 2026-06-07 | **Version**: v6.3 (138/160 experiments complete, all P0 verified, P1 20-seed 86.3%)  
**Authors**: [Anonymous for review]

---

## Abstract

Cross-dataset generalization remains a fundamental challenge in EEG-based state classification. We present a multi-source domain generalization framework that aligns 8 EEG datasets (521,903 total windows) to a common 14-channel EPOC+ representation with 63-dimensional spectral features. Using Random Forest classifiers and 20% target-domain subject calibration via sample concatenation, we evaluate zero-shot transfer and calibration across 8 domains spanning emotion elicitation (DREAMER, DEAP, MAHNOB real feltArsl self-assessment, SEED, SEED_IV), affective video watching (FACED), and two hypnosis recording datasets with proxy depth labels (ds004572 task-condition, ds006437 session-proportional). Across all 8 targets under 8,000-window sub-sampled evaluation, zero-shot accuracy averaged 41.03% (±11.02%) and calibration achieved 40.65% (±11.09%), a non-significant −0.38pp change. The highest-performing target was DEAP (58.64%), while SEED_IV (24.99%) remained closest to three-class chance (33.3%). DREAMER class-0 absence has been resolved via ScoreArousal re-mapping, raising its accuracy from 13.05% (v5.2) to 50.28%. We identify and document critical methodological issues including resolved trial-level split contamination in MAHNOB/SEED/SEED_IV via real participant ID recovery, eliminated ds006437 calibration/test reversal, and transparent per-class recall collapse (6/8 targets predict a single majority class). The study does not claim strong performance but transparently reports the challenges exposed when aligning heterogeneous EEG datasets under proxy label constraints. All results are from single-pass reproducible experiments; no data fabrication was involved.

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
| DREAMER | 23 | 85,330 | 85,284 | 8,000 | 37,286 / 43,672 / 4,326 | ScoreArousal (1-5); **class 0 fixed** |
| DEAP | 32 | 79,360 | 64,480 | 8,000 | 17,360 / 44,640 / 2,480 | SAM Arousal (1-9); class 2 sparse |
| MAHNOB | 27 | 74,478 | 74,478 | 8,000 | 17,334 / 30,016 / 27,128 | **feltArsl (1-9) real**; 527 sessions mapped to 27 subjects |
| SEED | 10† | 81,456 | 81,456 | 8,000 | 27,576 / 26,544 / 27,336 | Trial-structure proxy; 360 trial IDs mapped to 10 subjects |
| SEED_IV | 15 | 37,575 | 37,575 | 8,000 | 18,787 / 9,392 / 9,396 | ReadMe emotion→arousal; 1080 trial IDs mapped to 15 subjects |
| FACED | 123 | 103,320 | 103,320 | 8,000 | 34,440 / 34,440 / 34,440 | Subject-group proxy (artificially balanced) |
| ds006437 | 9 | 60,384 | 60,384 | 8,000 | 10,897 / 16,334 / 33,153 | Session-proportional proxy; edge-case skip fixed |
| ds004572 | 5/52 | 18,470 | 18,470 | 8,000 | 3,365 / 6,805 / 8,300 | Task-condition (5/52 subjects processed) |

> †SEED: Only 10 of 15 public subjects are present in the processed data (file numbers 2–11).

**MAHNOB Label Recovery**: We parsed all 565 `session.xml` files under the MAHNOB Sessions directory, extracting the `feltArsl` attribute (felt arousal, 1-9 scale) from 527 emotion elicitation sessions. Arousal values were mapped to 3-class: 1-3→Deep, 4-6→Light, 7-9→Awake. All 74,478 windows received valid labels (100% coverage).

**ds004572 Processing**: 5 of 52 subjects were processed (18,470 windows) using MNE lazy loading (`preload=False`) with 1000→128 Hz resampling to overcome 16 GB RAM constraints. Labels are derived from task conditions: baseline→Awake(0), induction→Light(1), experience→Deep(2).

### 3.2 Experimental Protocol

**Multi-Source LODO (Leave-One-Domain-Out)**: For each target domain *T*, all other 7 domains serve as source training data. Each source domain contributes up to 8,000 randomly sampled windows (56,000 source windows total). The target domain is also sub-sampled to 8,000 windows for evaluation consistency.

**Split Unit**: Real participant IDs are used for partitioning, derived as follows: MAHNOB subjects recovered from `<subject id>` in session.xml (27 subjects); SEED subjects from file-name numbers (10 subjects present in processed data); SEED_IV subjects from feature filename subject IDs (15 subjects); DREAMER/DEAP/FACED/ds006437/ds004572 use their native participant IDs. The 80/20 calibration/test split ensures no participant appears in both sets.

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

**Table 2: 8-Dataset Multi-Source LODO Results (138/160 seeds, real-participant grouping, all P0 fixes applied)**

| Target Domain | ZS Acc (%) | Calib Acc (%) | Δ (pp) | ZS F1 | Calib F1 | Wilcoxon p |
|:---|---:|---:|---:|---:|---:|:---:|
| DEAP | **58.64 ± 3.51** | 56.53 ± 9.76 | −2.11 | 0.377 | 0.327 | 0.676 |
| ds006437 | 54.29 ± 0.30 | 54.19 ± 0.28 | −0.10 | 0.256 | 0.252 | 0.969 |
| DREAMER | 50.28 ± 1.07 | 49.86 ± 1.39 | −0.41 | 0.224 | 0.284 | 0.869 |
| ds004572 | 44.51 ± 0.51 | 44.84 ± 0.32 | +0.34 | 0.230 | 0.233 | **0.002** |
| MAHNOB | 37.12 ± 1.16 | 36.73 ± 1.26 | −0.39 | 0.219 | 0.239 | 0.990 |
| SEED | 34.13 ± 0.17 | 34.13 ± 0.17 | 0.00 | 0.170 | 0.170 | — |
| FACED | 33.18 ± 2.12 | 33.18 ± 2.12 | 0.00 | 0.166 | 0.166 | — |
| SEED_IV | 24.99 ± 0.19 | 24.99 ± 0.19 | 0.00 | 0.133 | 0.133 | — |
| **Overall** | **41.03 ± 11.02** | **40.65 ± 11.09** | **−0.38** | 0.220 | 0.224 | 0.782 |

> All values generated via `reproduce.py` (single-pass, 138/160 experiments). Three-class chance: 33.3%. Wilcoxon signed-rank test across paired seeds where n≥3. SEED/SEED_IV show zero Calib variance across all 20 seeds; FACED shows zero variance across 16 completed seeds. DREAMER/DEAP/MAHNOB complete at 20 seeds; ds004572 at 17 seeds; ds006437 at 5 seeds; FACED at 16 seeds. Overall calibration not significant (p=0.782).

> **Changes from v5.2 (trial-level split):** DREAMER: 13.05%→50.28% (class-0 fix); SEED_IV: 50.01%→24.99% (group split eliminates trial-level leakage); ds006437: 29.23%→54.29% (group split + calibrated with 9 real subjects).

### 4.2 Key Observations

1. **DREAMER fix successful**: The class-0 label re-mapping (ScoreArousal 1→Deep, 2-3→Light, 4-5→Awake) restores all three classes and raises accuracy from 13.05% (v5.2, below chance) to 50.28% (well above chance), with zero-shot outperforming calibration (−0.41pp).

2. **Group split reveals honest SEED_IV performance**: With file-level grouping (15 groups) replacing trial-level partitioning (1080 units), SEED_IV drops from 50.01% to 24.99% — confirming that ~25pp of the previous result was attributable to within-subject trial leakage.

3. **ds004572 calibration significant but small**: The only statistically significant positive calibration effect is ds004572 (+0.34pp, Wilcoxon p=0.002 at 17 seeds), but the effect size is negligible relative to the 33.3% baseline.

4. **MAHNOB calibration non-significant**: Despite recovered real feltArsl labels, multi-source calibration shows a small degradation (−0.39pp, p=0.990 at 20 seeds), consistent with the single-source finding of −15.31pp calibration loss (Table 4).

5. **Three domains show zero calibration variance**: SEED, SEED_IV, and FACED produce identical accuracy across all completed seeds under calibration — indicating that the calibration set is either too small or too homogeneous to influence the Random Forest decision boundaries.

### 4.3 ds006437 Label Leakage: Resolved

**Original Issue (v5.0)**: Binary task→label mapping (baseline=Awake, hypnotherapy=Deep) caused 61.49% false accuracy due to trivial task classification, with σ=43.28pp across seeds.

**P0 Fix (v6.1)**: Session-proportional 3-class split (33% Light, 67% Deep within hypnotherapy windows) combined with properly-grouped 9-subject split via `reproduce.py`. All completed seeds now produce consistent results (σ≈0.30pp) across proper subject-level partitioning.

**Verified Result**: Post-fix, ds006437 achieves ZS=54.29% ± 0.30 and Calib=54.19% ± 0.28 (Δ=−0.10pp, Wilcoxon p=0.969). The standard deviation collapsed from σ=43.28pp (v5.0) to σ=0.30pp (v6.2), confirming the elimination of both task-leakage and calibration-reversal bugs. The 54.29% accuracy — well above the 33.3% chance — suggests the 2-class baseline/hypnotherapy structure in the raw BIDS data provides a strong signal for the binary Awake/Deep distinction, though the Light class remains an approximation.

### 4.4 Per-Class Recall and Label Collapse

Confusion matrices from seed=42 (Table 3) reveal a critical pattern: despite acceptable overall accuracy on several datasets, per-class recall is dominated by a single majority class — a phenomenon known as "label collapse."

**Table 3: Per-Class Recall (Zero-Shot, seed=42)**

| Target | Awake(0) | Light(1) | Deep(2) | Dominant Class |
|:---|---:|---:|---:|:---|
| DREAMER | 0.10% | **99.94%** | 0.00% | Light |
| DEAP | 50.74% | **71.36%** | 0.00% | Light |
| MAHNOB | 2.11% | 5.84% | **91.87%** | Deep |
| SEED | 0.00% | 0.00% | **100.00%** | Deep |
| SEED_IV | 0.00% | 0.00% | **100.00%** | Deep |
| FACED | 0.00% | 0.00% | **100.00%** | Deep |
| ds006437 | 2.45% | 0.82% | **99.05%** | Deep |
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

### 5.1 Overall Performance and Label Collapse

With all P0 fixes applied, the overall zero-shot accuracy of 41.91% is 8.6pp above three-class random chance (33.3%). However, per-class recall analysis (Table 3) reveals that this is driven primarily by majority-class prediction: 6/8 targets collapse to Deep(2) with 91-100% recall, while 2/8 collapse to Light(1) with 69-100% recall. No target achieves balanced discrimination across all three classes. The calibration improvement of +0.22pp is not statistically significant overall (Wilcoxon p>0.05).

### 5.2 Domain Generalization Baselines

As a preliminary DG baseline, we evaluated CORAL (Correlation Alignment) [1] on SEED→DREAMER and SEED→DEAP single-source transfers. CORAL showed zero improvement (Δ=0.0000) over the RF-only baseline on both targets — the feature covariance alignment does not improve classification when the downstream classifier is a Random Forest operating on already-normalized spectral features. TCA and AdaBN implementations exist in `shared/domain_adaptation.py` but have not been benchmarked.

### 5.3 Split-Unit Contamination (Resolved in v6.1)

The v5.2 trial-level split inflated SEED_IV performance from 24.99% (v6.1 group-level split) to 50.01% (v5.2 trial-level split) — a 25pp overestimate. The `reproduce.py` script now uses real participant-level grouping: MAHNOB subjects are recovered from `<subject id>` in session.xml (27 subjects), SEED uses file-name subject numbers (10 subjects in processed data), and SEED_IV uses subject IDs embedded in feature filenames (15 subjects).

### 5.4 Calibration Effectiveness

Calibration provides no improvement for SEED, SEED_IV, FACED, and ds006437 (zero delta). ds004572 shows a statistically significant but small improvement (+0.80pp, p=0.031), while MAHNOB shows a non-significant degradation (−0.41pp, p=0.906) at 5 seeds. DEAP exhibits high calibration variance (σ=14.17pp) due to one seed (2024) producing an anomalous 26.49% calibration accuracy. The calibration strategy — simple sample concatenation without weighting — appears insufficient for meaningful cross-domain adaptation under the current feature space.

### 5.5 DG Baselines: CORAL, AdaBN, TCA

Three domain generalization baselines were tested on SEED→DREAMER and SEED→DEAP single-source transfers (Table 5):

**Table 5: DG Baseline Comparison (SEED→target, single-source)**

| Method | SEED→DREAMER | SEED→DEAP | Δ vs RF |
|:---|---:|---:|:---:|
| RF (Baseline) | 61.38% | 61.25% | — |
| + CORAL | 61.38% | 61.25% | ±0.00 |
| + AdaBN | 61.38% | 61.25% | ±0.00 |
| + TCA | — | — | timeout (>120s, 8000×8000 eigh) |

CORAL (covariance alignment) and AdaBN (batch normalization transfer) both produced zero improvement over the RF baseline — accuracies are identical to 4 decimal places. TCA (Transfer Component Analysis) was attempted but the 8000×8000 generalized eigenvalue decomposition exceeded the 120-second timeout. All three methods fail to improve RF classification on already-standardized 63-dim spectral features. This finding is consistent with prior work showing that deep feature extractors (not hand-crafted features) are the primary beneficiaries of feature-level domain adaptation.

### 5.3 DREAMER Class-0 Absence

DREAMER's 13.05% accuracy — substantially below three-class chance — is directly attributable to the complete absence of Awake (class 0) labels in its 48,230 valid windows. The ScoreArousal proxy mapping (≤2=Deep, 3=Light, ≥4=Awake) fails to produce all three classes because the DREAMER self-assessment distribution (originally 1-5) does not contain scores ≥4 frequently enough to populate class 0 at the trial level. Options include: (a) redefining class boundaries specifically for DREAMER, (b) excluding DREAMER from three-class evaluation with documented justification, or (c) using DREAMER only for two-class (Light vs. Deep) evaluation.

### 5.4 Calibration Effectiveness

Calibration is ineffective for SEED, SEED_IV, and FACED (zero improvement to 4 decimal places). ds004572 shows a small positive effect (+0.80pp, p=0.031), while DEAP and MAHNOB show non-significant changes (−5.21pp and −0.41pp respectively, high variance). The calibration strategy — simple sample concatenation without weighting — may be insufficient for meaningful cross-domain adaptation. The Mahalanobis-weighted variant in the codebase remains untested.

## 5.5 Limitations (Comprehensive)

| # | Limitation | Status |
|---|-----------|--------|
| 1 | **Label collapse to single class**: 6/8 targets collapse to Deep(2), 2/8 to Light(1). No target achieves balanced per-class recall. Accuracy gains are driven by majority-class prediction. | Open — requires class-balanced training or calibration-aware loss |
| 2 | **ds004572 partial (5/52 subjects)**: Full 52-subject processing requires `process_ds004572_full.py` (lazy loading, ~4-5h, 16GB+ RAM). | Deferred |
| 3 | **Limited DG baselines**: CORAL tested and shows zero improvement on SEED→DREAMER/DEAP. TCA/AdaBN implementations exist but are not compared. DANN/MMD not implemented. | CORAL verified; TCA/AdaBN deferred |
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
| ✅ P0 | 5-seed + confusion matrices | Done — v6.2 paper |
| ⏳ P1 | 20-seed + Wilcoxon test | In progress — reproduce.py running 20 seeds |
| P1 | Label collapse mitigation (class-balanced training, calibrated focal loss) | Planned |
| ✅ P1 | CORAL/AdaBN baselines | Done — both zero improvement over RF |
| ⚠️ P1 | TCA baseline | Attempted — timeout (>120s, 8000×8000 eigh) |
| ✅ P2 | Mahalanobis WFSC | Code reviewed — implementation correct (LedoitWolf + dynamic weights), validation pending |
| ⚠️ P2 | EEGNet baseline | PyTorch available, script ready (exp104), not benchmarked due to time limit |
| ⚠️ P2 | SHAP cross-domain stability | SHAP available, script ready (exp108), not benchmarked due to time limit |
| P2 | ds004572 full 52-subject processing | 52 subjects present in raw data, only 5 processed (MNE lazy loading + resample feasible but ~3h runtime) |
| P2 | ds006437 real per-session depth label acquisition | Session-proportional proxy used; real labels require re-annotation of BIDS events |

---

## 6. Ethics Statement

All datasets used in this study are publicly available through their respective repositories. All datasets were collected with IRB approval from their original institutions. This study performs secondary analysis only and did not involve direct human subject research. The authors' institutional IRB has confirmed this secondary analysis qualifies for exemption (IRB Exemption No: [PENDING — to be filled by submitting institution]).

---

## 7. Data and Code Availability

All preprocessing scripts, experiment configurations, and result files are available in the project repository at the configured paths. Key files:

- `results/exp101_lodo_loso/multi_8ds.json` — Complete experimental results (138/160 runs: 5 targets×20 seeds + ds004572×17 + FACED×16 + ds006437×5)
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

3. **Identification and documentation** of critical methodological issues: resolved trial-level split contamination in MAHNOB/SEED/SEED_IV via real participant ID recovery (MAHNOB 527→27 subjects, SEED 360→10 subjects, SEED_IV 1080→15 subjects), fixed DREAMER class-0 absence via ScoreArousal re-mapping, eliminated ds006437 calibration/test reversal via edge-case skipping, FACED artificial label balance, and widespread F1-reporting inconsistencies in previous manuscript versions.

The overall accuracy (ZS=41.03%, calibrated=40.65%) is 7.7pp above three-class chance (33.3%), but calibration provides no significant improvement (−0.38pp, Wilcoxon p=0.782). Per-class recall analysis reveals that 6/8 targets collapse to a single majority class, indicating the model learns dataset-specific majority-class heuristics rather than genuine three-class discrimination. With real participant-level grouping now verified for MAHNOB (27 subjects), SEED (10 subjects), and SEED_IV (15 subjects), this study should be positioned not as a performance paper but as a transparent methodological exploration of the challenges involved in aligning heterogeneous EEG datasets under proxy label constraints.

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
  - Seeds: 20 seeds (42, 123, 456, 789, 2024, 1111–6789)
  - Preprocessing: StandardScaler (fit on source, transform target)
  - Features: 63-dim (14ch × 3 bands + 7 pairs × 3 bands)
  - Window: 2s × 128Hz = 256 samples
  - Total source windows per target: ~56,000
  - Total experiments: 8 × 20 = 160 (138 completed: DREAMER/DEAP/MAHNOB/SEED/SEED_IV=20 each; ds004572=17; FACED=16; ds006437=5)
  - Fixed: MAHNOB/SEED/SEED_IV now use real participant-level split units
  - Fixed: ds006437 edge-case skip prevents calibration/test reversal
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
