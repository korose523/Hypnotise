# Universal BCI Hypnosis Depth Classification (通用EEG催眠深度分类系统)

> **Multi-Source Domain Generalization with WFSC Calibration for Cross-Dataset Three-Level Hypnosis Depth EEG Classification**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements a complete pipeline for **cross-dataset three-level hypnosis depth classification** using EEG signals. The system unifies **8 public EEG datasets** (including 2 real hypnosis datasets) into a common 63-dimensional feature space, trains multi-source domain generalization models, and applies **Weighted Feature Space Calibration (WFSC)** with Mahalanobis dynamic weighting for target domain adaptation.

### Two Papers

| Paper | Focus | Key Contribution |
|-------|-------|-----------------|
| **Paper 1** (Offline) | Multi-source domain generalization framework | MSDG-WFSC with LODO+LOSO evaluation, 8 datasets, 20 seeds |
| **Paper 2** (Real-time) | EPOC+ real-time BCI system | <200ms latency, SSS protocol, online WFSC calibration |

### Key Features

- **63-dimensional feature space**: 14 channels x 3 bands Log-Bandpower + 7 asymmetry pairs x 3 bands DASM
- **8 datasets unified**: DREAMER, DEAP, MAHNOB-HCI, SEED, SEED-IV, FACED, ds004572, ds006437
- **LODO + LOSO evaluation**: Leave-One-Domain-Out cross-validation with inner Leave-One-Subject-Out splits
- **WFSC calibration**: Mahalanobis distance-based dynamic sample weighting for cross-domain adaptation
- **Real-time pipeline**: EPOC+ compatible, <200ms end-to-end latency, prediction smoothing
- **Full reproducibility**: 20 random seeds, non-parametric statistics, bootstrap confidence intervals

---

## Directory Structure

```
universal_bci_hypnosis/
|-- config.yaml                          # Global configuration (single source of truth)
|-- requirements.txt                     # Python dependencies
|-- .gitignore                           # Git ignore rules
|
|-- shared/                              # Shared utility modules
|   |-- __init__.py                      # Package exports
|   |-- config_loader.py                 # Config validation & directory creation
|   |-- seed_manager.py                  # Central random seed management
|   |-- logger.py                        # Unified logging (console + file)
|   |-- split_manager.py                 # LODO/LOSO/LOO split management
|   |-- feature_extraction.py            # 63-dim feature extraction (14ch mapping, BP, DASM)
|   |-- label_mapping.py                 # Dataset-specific -> 3-class label mapping
|   |-- domain_adaptation.py             # CORAL, TCA, AdaBN implementations
|   |-- wfsc.py                          # WFSC (Mahalanobis + Fixed weight variants)
|   |-- metrics.py                       # Metrics & statistical tests
|
|-- scripts/                             # Offline experiment scripts (Paper 1)
|   |-- prep01_build_63feat_all_datasets.py  # Step 1: Data loading, 14ch mapping, windowing
|   |-- prep02_make_3class_hypnosis_labels.py # Step 2: 63-dim feature extraction
|   |-- prep03_generate_splits_lodo_loso.py    # Step 3: 3-class label generation & alignment
|   |-- prep04_generate_splits_lodo_loso.py    # Step 4: Train/calib/test split generation
|   |-- exp101_rf_lodo_loso_zero_shot_vs_wfsc.py  # Core: Zero-shot vs WFSC (LODO x 20 seeds)
|   |-- exp102_rf_calibration_ratio_sweep_lodo.py  # Calibration ratio sweep
|   |-- exp103_wfsc_dynamic_mahalanobis_vs_fixedw.py  # WFSC ablation study
|   |-- exp104_eegnet_lodo_loso_baseline.py  # EEGNet deep learning baseline
|   |-- exp105_real_da_baselines_coral_tca_adabn.py  # Domain adaptation baselines
|   |-- exp106_legacy_setting_dreamer_mahnob_to_deap.py  # Legacy 2-dataset reproduction
|   |-- exp107_stats_tests_bootstrap_wilcoxon.py  # Statistical tests & publication tables
|   |-- exp108_shap_cross_domain_stability.py  # SHAP interpretability analysis
|
|-- realtime/                            # Real-time experiment scripts (Paper 2)
|   |-- rt201_epoc_protocol_segment_and_rating.py    # Protocol design & segmentation
|   |-- rt202_epoc_stream_to_63feat.py               # Real-time stream -> 63-dim features
|   |-- rt203_realtime_inference_public_pretrained.py # Real-time inference engine
|   |-- rt204_realtime_wfsc_calibration_fixed_vs_mahal.py  # Online calibration study
|   |-- rt205_online_eval_metrics_and_latency.py     # Latency profiling & evaluation
|
|-- data/                                # Raw EEG datasets (not in git, see below)
|-- processed/                           # Preprocessed features & labels (auto-generated)
|-- splits/                              # Train/calib/test splits (auto-generated)
|-- models/                              # Saved models (auto-generated)
|-- results/                             # Experiment results (auto-generated)
|-- logs/                                # Log files (auto-generated)
```

---

## Environment Setup

### Requirements

- Python >= 3.8
- NumPy >= 1.21
- SciPy >= 1.7
- scikit-learn >= 1.0
- MNE-Python >= 1.0 (for BIDS dataset loading)
- PyTorch >= 1.9 (optional, for EEGNet)
- PyYAML, h5py, mat4py, pandas, matplotlib, seaborn, tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/korose523/BCI_Full_Length.git
cd BCI_Full_Length

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: install PyTorch for EEGNet experiments
pip install torch torchvision
```

---

## Dataset Download

Place each dataset in the `data/` directory as follows:

| Dataset | Path | Source | Type |
|---------|------|--------|------|
| DREAMER | `data/DREAMER/DREAMER.mat` | [IEEE DataPort](https://ieee-dataport.org/) | Emotion (proxy) |
| DEAP | `data/DEAP/s01.dat ... s32.dat` | [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) | Emotion (proxy) |
| MAHNOB-HCI | `data/MAHNOB/Sessions/` | [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/) | Emotion (proxy) |
| SEED | `data/SEED/` | [SEED](https://bcmi.sjtu.edu.cn/~seed/) | Emotion (proxy) |
| SEED-IV | `data/SEED_IV/` | [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) | Emotion (proxy) |
| FACED | `data/FACED/EEG_Features/` | [FACED](https://github.com/FACED-Dataset/FACED) | Emotion (proxy) |
| ds004572 | `data/ds004572/` (BIDS) | [OpenNeuro](https://openneuro.org/datasets/ds004572) | **Real hypnosis** |
| ds006437 | `data/ds006437/` (BIDS) | [OpenNeuro](https://openneuro.org/datasets/ds006437) | **Real hypnosis** |

### Quick Setup

```bash
mkdir -p data/DREAMER data/DEAP data/MAHNOB data/SEED data/SEED_IV data/FACED data/ds004572 data/ds006437

# Download and extract each dataset into its respective directory
# See individual dataset READMEs for download instructions
```

---

## Run Order

**CRITICAL**: Scripts must be run in the order shown below. Each stage depends on the output of the previous stage.

### Stage 1: Preprocessing (prep01 -> prep02 -> prep03 -> prep04)

```bash
# Step 1: Load raw EEG, map to 14 EPOC+ channels, sliding window segmentation
python scripts/prep01_build_63feat_all_datasets.py

# Step 2: Extract 63-dimensional features per window
python scripts/prep02_make_3class_hypnosis_labels.py

# Step 3: Generate 3-class hypnosis depth labels (Awake/Light/Deep)
python scripts/prep03_generate_splits_lodo_loso.py

# Step 4: Generate LODO/LOSO train/calib/test splits
python scripts/prep04_generate_splits_lodo_loso.py
```

### Stage 2: Offline Experiments — Paper 1 (exp101 -> exp108)

```bash
# Core experiment: Zero-shot RF vs WFSC-Mahalanobis (LODO x 8 targets x 20 seeds)
python scripts/exp101_rf_lodo_loso_zero_shot_vs_wfsc.py

# Calibration ratio sweep: [0, 0.05, 0.10, 0.20, 0.30, 0.50]
python scripts/exp102_rf_calibration_ratio_sweep_lodo.py

# WFSC ablation: Mahalanobis dynamic vs Fixed weight
python scripts/exp103_wfsc_dynamic_mahalanobis_vs_fixedw.py

# Deep learning baseline: EEGNet-v4
python scripts/exp104_eegnet_lodo_loso_baseline.py

# Domain adaptation baselines: CORAL, TCA, AdaBN
python scripts/exp105_real_da_baselines_coral_tca_adabn.py

# Legacy reproduction: DREAMER+MAHNOB -> DEAP (2-dataset setting)
python scripts/exp106_legacy_setting_dreamer_mahnob_to_deap.py

# Statistical tests: Bootstrap CI, Wilcoxon, paired t-test, publication tables
python scripts/exp107_stats_tests_bootstrap_wilcoxon.py

# SHAP interpretability and cross-domain feature importance stability
python scripts/exp108_shap_cross_domain_stability.py
```

### Stage 3: Real-time Experiments — Paper 2 (rt201 -> rt205)

```bash
# Protocol design: 4-phase segmentation + subjective rating
python realtime/rt201_epoc_protocol_segment_and_rating.py

# Real-time stream: EPOC+ EEG -> 63-dim features (circular buffer)
python realtime/rt202_epoc_stream_to_63feat.py

# Real-time inference: Pretrained model + prediction smoothing
python realtime/rt203_realtime_inference_public_pretrained.py

# Online calibration study: WFSC-Mahalanobis vs WFSC-Fixed convergence
python realtime/rt204_realtime_wfsc_calibration_fixed_vs_mahal.py

# Evaluation: Latency profiling, phase-level accuracy, prediction stability
python realtime/rt205_online_eval_metrics_and_latency.py
```

---

## Feature Description

### 63-Dimensional Feature Vector

| Range | Dimensions | Description |
|-------|-----------|-------------|
| [0:42] | 42 | 14 channels x 3 bands Log-Bandpower (Theta, Alpha, Beta) |
| [42:63] | 21 | 7 asymmetry pairs x 3 bands DASM (left - right) |

### Feature Computation (Paper Section 3.2)

1. **Channel mapping**: All datasets mapped to 14 EPOC+ channels via nearest-neighbor on 10-20 coordinates
2. **Resampling**: All data resampled to 128 Hz via integer-ratio polyphase resampling
3. **Sliding window**: Window W = 256 samples (2s), Step S = 128 (1s), 50% overlap
4. **Log-Bandpower**: Welch PSD estimation per channel per band, `log10(trapz(PSD) + 1e-10)`
5. **DASM**: `f_dasm = logBP(left) - logBP(right)` for 7 symmetric electrode pairs
6. **Normalization**: Subject-level z-score per feature dimension

### Frequency Bands

| Band | Range | Neural Relevance |
|------|-------|-----------------|
| Theta | 4-8 Hz | Hypnosis depth marker (frontal enhancement) |
| Alpha | 8-13 Hz | Relaxation & focused attention |
| Beta | 13-30 Hz | Arousal & active processing |

### 14-Channel EPOC+ Montage

```
AF3  F7   F3   FC5  T7   P7   O1
AF4  F8   F4   FC6  T8   P8   O2
```

### 7 Asymmetry Pairs (Left - Right)

1. AF3 - AF4 (prefrontal)
2. F7 - F8 (frontal)
3. F3 - F4 (mid-frontal)
4. FC5 - FC6 (fronto-central)
5. T7 - T8 (temporal)
6. P7 - P8 (parietal)
7. O1 - O2 (occipital)

---

## Label Transparency

### 3-Class Hypnosis Depth

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Awake (清醒) | Normal waking consciousness |
| 1 | Light Hypnosis (浅催眠) | Relaxation, heightened suggestibility |
| 2 | Deep Hypnosis (深催眠) | Profound relaxation, altered perception |

### IMPORTANT: Label Types

| Dataset | Label Type | Source | Notes |
|---------|-----------|--------|-------|
| DREAMER | **Proxy** (arousal) | Self-assessment (1-5 scale) | Arousal <= 2 = Deep, 3 = Light, >= 4 = Awake |
| DEAP | **Proxy** (arousal) | SAM arousal (1-9 scale) | Arousal <= 3 = Deep, 4-6 = Light, >= 7 = Awake |
| MAHNOB-HCI | **Proxy** (arousal) | Self-assessment | Same as DEAP |
| SEED | **Proxy** (emotion) | 3-class emotion | Positive=Awake, Neutral=Light, Negative=Deep |
| SEED-IV | **Proxy** (emotion) | 4-class emotion | Mapped to 3-class |
| FACED | **Proxy** (arousal) | Continuous arousal | Discretized to 3 levels |
| **ds004572** | **True hypnosis** | Therapist depth score (0-10) | Direct hypnosis depth measurement |
| **ds006437** | **True hypnosis** | Protocol phases | Pre/during/post phases |

> **Note**: Proxy labels are derived from arousal/emotion dimensions as approximations of hypnosis-related states. They are clearly marked in all outputs. Only ds004572 and ds006437 contain true hypnosis labels.

---

## Evaluation Protocol

### LODO (Leave-One-Domain-Out) — Outer Layer

- 8-fold cross-dataset validation
- Each dataset takes turns as target domain
- Remaining 7 datasets merged as source training set
- No target domain labels used during training (Zero-shot)

### LOSO (Leave-One-Subject-Out) — Inner Layer

- Within each target domain, subjects are split into calibration and test sets
- Calibration ratio: 20% (default), with sweep from 0% to 50%
- Stratified by class distribution

### Statistical Testing

- **20 random seeds** per experiment
- **Wilcoxon signed-rank test** (non-parametric, n=20)
- **Paired t-test** with Cohen's d effect size
- **Bootstrap 95% CI** (B=10,000 iterations)
- Significance level: alpha = 0.05

---

## Configuration

All parameters are centralized in `config.yaml`. Key settings:

```yaml
# Feature extraction
features:
  fs_target: 128        # Target sampling rate (Hz)
  window_sec: 2.0       # Window duration (s)
  step_sec: 1.0         # Step between windows (s)
  nperseg: 128          # Welch PSD segment length

# Random Forest model
model:
  rf:
    n_estimators: 500
    min_samples_leaf: 5
    class_weight: "balanced"

# Experiment
experiment:
  n_seeds: 20
  calib_ratios: [0, 0.05, 0.10, 0.20, 0.30, 0.50]
```

---

## FAQ & Design Decisions

### Q: Why hand-crafted features instead of deep learning?

A: In cross-dataset zero-shot scenarios, the domain gap (different devices, electrode layouts, sampling rates) severely degrades deep model performance. Hand-crafted features with explicit physical meaning (bandpower in specific frequency ranges) transfer more robustly across domains. Our ablation study (exp104) quantifies this gap.

### Q: Why 14 channels instead of 32/64?

A: The EMOTIV EPOC+ consumer-grade headset has exactly 14 channels. By mapping all datasets to this layout, we ensure **direct deployability** on the real-time system (Paper 2). This is a deliberate engineering constraint, not a limitation.

### Q: Why is DASM computed for all 3 bands, not just Alpha?

A: While the original paper highlights Alpha-band asymmetry for hypnosis, computing DASM for all 3 bands (Theta, Alpha, Beta) provides additional discriminative power at minimal computational cost. The 7 pairs x 3 bands = 21 DASM dimensions are included in the 63-dim feature vector.

### Q: How are proxy labels justified?

A: Proxy labels are used because no large-scale, multi-subject, real-hypnosis EEG dataset exists with standardized annotations. We leverage the well-established relationship between arousal reduction and hypnotic depth (Theta enhancement, Beta suppression) to approximate hypnosis-like states from emotion datasets. **All proxy labels are clearly marked** in outputs and the paper explicitly discusses this limitation.

### Q: What is WFSC and why Mahalanobis?

A: WFSC (Weighted Feature Space Calibration) re-weights source domain training samples based on their similarity to the target domain distribution. The Mahalanobis distance accounts for feature correlations (via the covariance matrix), providing more principled weighting than Euclidean distance. LedoitWolf robust covariance estimation ensures numerical stability in high dimensions (63-dim).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bci_hypnosis_2025,
  title={Multi-Source Domain Generalization with WFSC Calibration for Cross-Dataset Three-Level Hypnosis Depth EEG Classification},
  author={},
  journal={},
  year={2025}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

Individual datasets are subject to their respective licenses and terms of use.
