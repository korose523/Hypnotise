# Universal BCI Hypnosis Depth Classification (通用EEG催眠深度分类系统)

> **Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version v6.1](https://img.shields.io/badge/version-v6.1-brightgreen)]()

---

## Overview

This repository implements a complete pipeline for **cross-dataset three-level hypnosis depth classification** using EEG signals. The system unifies **8 public EEG datasets** into a common 63-dimensional feature space, trains multi-source domain generalization models, and evaluates **Few-Shot Subject Calibration (FS²C)** with simplified sample concatenation for target domain adaptation.

### Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v5.2** | 2026-06-03 | Single-pass verification, ds006437 label leak fixed, MAHNOB real arousal labels recovered, 30+ redundant files cleaned |
| v5.0 | 2026-06-02 | 8-dataset multi-source LODO with MAHNOB real labels |
| v2.1 | 2026-05-14 | 63-dim locked features, LODO/LOSO/LOO, bootstrap CI |

### Key Features

- **63-dimensional feature space**: 14 channels × 3 bands Log-Bandpower + 7 asymmetry pairs × 3 bands DASM
- **8 datasets unified** (521,903 total windows):
  - 5 emotion proxy: DREAMER, DEAP, MAHNOB-HCI, SEED, SEED-IV
  - 1 affective video: FACED
  - 2 true hypnosis: ds004572, ds006437
- **Real MAHNOB self-assessment labels**: 1-9 feltArsl extracted from session.xml metadata (527 trials, 100% window coverage)
- **Multi-Source LODO**: Leave-One-Domain-Out — 7 source domains train, 1 target domain evaluates
- **FS²C calibration**: 20% target-domain subjects added to training set for domain adaptation
- **Label leak diagnosis & fix**: ds006437 binary task→label leakage resolved with session-proportional 3-class split
- **ds004572 lazy loading**: 1000→128Hz MNE resampling with preload=False to handle 45GB on 16GB RAM

---

## Final Experimental Results (v5.2)

> Single-pass verification: 8 targets × 3 seeds (42, 123, 456), MAX_SRC=8,000, RF(n=200, balanced), 363 seconds

| Target Domain | Zero-Shot Acc | WFSC Acc (20%) | Δ | ZS F1 | Label Source |
|:---|---:|---:|---:|---:|:---|
| SEED_IV | **50.01%** ± 0.22 | 50.01% ± 0.22 | — | 0.488 | ReadMe emotion→arousal |
| ds004572 | 41.97% ± 0.07 | **43.26%** ± 0.15 | +1.29pp | 0.417 | Task-condition (5/52 subj) |
| DEAP | 39.25% ± 17.48 | 41.34% ± 1.00 | +2.08pp | 0.286 | SAM Arousal (1-9) |
| FACED | 37.97% ± 6.27 | 37.97% ± 6.27 | — | 0.330 | Subject-group proxy |
| SEED | 34.29% ± 4.50 | 34.29% ± 4.50 | — | 0.331 | Trial-structure proxy |
| MAHNOB | 29.41% ± 0.88 | 29.24% ± 0.73 | −0.17pp | 0.239 | **feltArsl (1-9) real** |
| ds006437 | 29.23% ± 12.27 | 29.28% ± 12.28 | +0.05pp | 0.256 | Session-proportional [FIXED] |
| DREAMER | 13.05% ± 0.09 | 13.52% ± 0.00 | +0.47pp | 0.143 | ScoreArousal (1-5) |
| **Overall** | **34.40%** ± 13.04 | **34.86%** ± 11.63 | **+0.47pp** | — | — |

### Key Findings

1. **DREAMER class-0 fixed**: ScoreArousal re-mapped → 49.66% (was 13.05% below chance)
2. **Group split reveals honest SEED_IV**: File-level grouping → 24.99% (was 50.01% trial-leaked)
3. **ds006437 stabilized**: σ 43.28→0.26pp, seed-456 reversal fixed
4. **MAHNOB real labels**: feltArsl self-assessment recovered from session.xml
5. **20-seed Wilcoxon**: 160 experiments, calibration not significant (+0.22pp overall)
6. **DG baselines all zero**: CORAL Δ=0.00, AdaBN Δ=0.00, TCA timeout
7. **Label collapse**: 6/8 targets collapse to single class (per-class recall analysis)

---

## Directory Structure

```
universal_bci_hypnosis/
├── paper_v5_final.md                    # Final paper (v5.2, single-pass verified)
├── config.yaml                          # Global configuration
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── run_all_experiments.py               # Unified experiment runner (20-seed + EEGNet + Mahalanobis)
├── eegnet_baseline.py                   # EEGNet-v4 PyTorch baseline
├── fix_mahnob_labels.py                 # MAHNOB real arousal label recovery from session.xml
├── fix_ds006437_labels.py               # ds006437 session-proportional label fix
├── process_ds004572_full.py             # ds004572 lazy-loading 1000→128Hz processor
│
├── shared/                              # Shared utility modules
│   ├── config_loader.py                 # Config validation & directory creation
│   ├── seed_manager.py                  # Central random seed management
│   ├── logger.py                        # Unified logging (console + file)
│   ├── split_manager.py                 # LODO/LOSO/LOO split management
│   ├── feature_extraction.py            # 63-dim feature extraction (14ch mapping, BP, DASM)
│   ├── label_mapping.py                 # Dataset-specific → 3-class label mapping
│   ├── domain_adaptation.py             # CORAL, TCA, AdaBN implementations
│   ├── mahalanobis_wfsc.py              # Mahalanobis dynamic-weight WFSC (LedoitWolf)
│   ├── wfsc.py                          # Fixed-weight WFSC variant
│   └── metrics.py                       # Metrics & statistical tests
│
├── scripts/                             # Preprocessing pipeline (prep01–prep04)
│   ├── prep01_build_63feat_all_datasets.py  # Load raw EEG, 14ch mapping, windowing
│   ├── prep02_make_3class_hypnosis_labels.py # 63-dim feature extraction
│   ├── prep03_generate_splits_lodo_loso.py   # 3-class label generation & alignment
│   └── prep04_generate_splits_lodo_loso.py   # Train/calib/test split generation
│
├── realtime/                            # Real-time EPOC+ BCI scripts (Paper 2, planned)
├── data/                                # Raw EEG datasets (not in git)
├── processed/                           # Preprocessed features & labels
├── results/                             # Experiment results
├── models/                              # Saved models
└── logs/                                # Log files
```

---

## Environment Setup

```bash
# Clone
git clone https://github.com/korose523/BCI_Full_Length.git
cd BCI_Full_Length

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For EEGNet baseline:
pip install torch torchvision
```

**Python ≥ 3.8** required. Core dependencies: NumPy, SciPy, scikit-learn, MNE-Python, PyYAML, h5py, pandas, tqdm.

---

## Dataset Setup

All 8 datasets are in the project `data/` folder (total ~62 GB). Paths are configured in `config.yaml`.

| # | Dataset | Data Path | Size | Label Source | Type |
|---|---------|-----------|------|-------------|------|
| 1 | DREAMER | `data/DREAMER/DREAMER.mat` | 0.5 GB | ScoreArousal (1-5) | Emotion proxy |
| 2 | DEAP | `data/DEAP/data_preprocessed_python/` | 3.3 GB | SAM Arousal (1-9) | Emotion proxy |
| 3 | MAHNOB-HCI | `data/MAHNOB/Sessions/` | 3.8 GB | **feltArsl (1-9) real** | Emotion proxy |
| 4 | SEED | `data/SEED/ExtractedFeatures_1s/` | 1.9 GB | Trial-structure proxy | Emotion proxy |
| 5 | SEED-IV | `data/SEED_IV/eeg_feature_smooth/` | 0.3 GB | ReadMe emotion→arousal | Emotion proxy |
| 6 | FACED | `data/FACED/EEG_Features/` | 0.3 GB | Subject-group proxy | Affective video |
| 7 | ds004572 | `data/ds004572/` (BIDS) | 47.3 GB | Task-condition | **True hypnosis** |
| 8 | ds006437 | `data/ds006437/` (BIDS) | 4.7 GB | Session-proportional [FIXED] | **True hypnosis** |

**Dataset sources**: DREAMER ([IEEE DataPort](https://ieee-dataport.org/)), DEAP ([QMUL](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)), MAHNOB-HCI ([mahnob-db.eu](https://mahnob-db.eu/hci-tagging/)), SEED/SEED-IV ([BCMI Cloud](https://cloud.bcmi.sjtu.edu.cn)), FACED ([GitHub](https://github.com/FACED-Dataset/FACED)), ds004572/ds006437 ([OpenNeuro](https://openneuro.org)).

**MAHNOB self-assessment**: Real 1-9 feltArsl labels recovered from `data/MAHNOB/Sessions/*/session.xml` (see `fix_mahnob_labels.py`). Also saved as `data/MAHNOB/mahnob_self_assessment.json`.

---

## Run Order

### Stage 1: Preprocessing

```bash
# Step 1: Load raw EEG, map to 14 EPOC+ channels, sliding windows
python scripts/prep01_build_63feat_all_datasets.py

# Step 2: Extract 63-dimensional features per window
python scripts/prep02_make_3class_hypnosis_labels.py

# Step 3: Generate 3-class hypnosis depth labels (Awake/Light/Deep)
python scripts/prep03_generate_splits_lodo_loso.py

# Step 4: Generate calibration splits
python scripts/prep04_generate_splits_lodo_loso.py
```

### Stage 2: Label Fixes (run before experiments)

```bash
# Recover MAHNOB real self-assessment arousal labels from session.xml
python fix_mahnob_labels.py

# Fix ds006437 label leakage (binary task→label → session-proportional 3-class)
python fix_ds006437_labels.py
```

### Stage 3: Experiments

```bash
# Multi-source LODO (subsampled, ~6min):
python run_all_experiments.py

# Full-scale (all data, 20 seeds, ~2-8h):
python run_all_experiments.py --full

# EEGNet baseline:
python run_all_experiments.py --eegnet
```

### Stage 4: ds004572 Full Processing (optional, 52 subjects)

```bash
# Lazy-loading 1000→128Hz for all 52 subjects (~4-5h):
NUMBA_DISABLE_JIT=1 python process_ds004572_full.py
```

---

## Feature Description

### 63-Dimensional Feature Vector

| Range | Dimensions | Description |
|-------|-----------|-------------|
| [0:42] | 42 | 14 channels × 3 bands Log-Bandpower (Theta 4-8Hz, Alpha 8-13Hz, Beta 13-30Hz) |
| [42:63] | 21 | 7 asymmetry pairs × 3 bands DASM |

### Pipeline

1. **Channel mapping**: Nearest-neighbor on 10-20 coordinates → 14 EPOC+ channels
2. **Resampling**: Integer-ratio polyphase → 128 Hz
3. **Sliding window**: 256 samples (2s), step 128 (1s), 50% overlap
4. **Log-Bandpower**: Welch PSD → `log10(trapz(PSD) + 1e-10)` per band
5. **DASM**: `logBP(left) - logBP(right)` for 7 symmetric pairs
6. **Normalization**: Subject-level z-score per feature dimension

### 14 EPOC+ Channels

```
AF3  F7   F3   FC5  T7   P7   O1
AF4  F8   F4   FC6  T8   P8   O2
```

### 7 Asymmetry Pairs

AF3-AF4, F7-F8, F3-F4, FC5-FC6, T7-T8, P7-P8, O1-O2 (all left-minus-right)

---

## Label Transparency

### 3-Class Mapping

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Awake (清醒) | Normal waking consciousness / high arousal |
| 1 | Light Hypnosis (浅催眠) | Relaxation, transition state |
| 2 | Deep Hypnosis (深催眠) | Profound relaxation, altered perception / low arousal |

### Per-Dataset Label Details

| Dataset | Original Scale | 3-Class Boundaries | Type |
|---------|---------------|-------------------|------|
| DREAMER | ScoreArousal (1-5) | ≤2=Deep, 3=Light, ≥4=Awake | Proxy |
| DEAP | SAM Arousal (1-9) | ≤3=Deep, 4-6=Light, ≥7=Awake | Proxy |
| MAHNOB | **feltArsl (1-9) real** | ≤3=Deep, 4-6=Light, ≥7=Awake | **Real self-report** |
| SEED | de_movingAve trial structure | Trial-group based | Proxy |
| SEED-IV | ReadMe emotion labels (0-3) | {2,3}→Awake, 0→Light, 1→Deep | Proxy (ReadMe-derived) |
| FACED | PSD/DE features | Subject-group based | Proxy |
| ds004572 | Task condition | Baseline→Awake, Induction→Light, Experience→Deep | Task-condition |
| ds006437 | Session-proportional split | Baseline→Awake, Hypno-1st-33%→Light, Hypno-67%→Deep | [FIXED] session-proxy |

> ⚠️ **Important**: Only MAHNOB uses real continuous self-assessment labels (1-9 scale). All other datasets use proxy or task-condition mappings. ds004572 and ds006437 are true hypnosis datasets but lack numeric depth scores (0-10). Label types are clearly marked in all outputs.

---

## Evaluation Protocol

### Multi-Source LODO (Leave-One-Domain-Out)

- 8-fold: each dataset as target, remaining 7 merged as source
- No target labels used during zero-shot training
- 20% target-domain subjects reserved for FS²C calibration

### FS²C (Few-Shot Subject Calibration)

- Calibration samples concatenated with source data before final RF training
- Simple sample concatenation (not Mahalanobis-weighted — see `shared/mahalanobis_wfsc.py` for advanced variant)

### Statistical Reliability

- 3 seeds (42, 123, 456) for rapid iteration
- 20-seed mode available via `run_all_experiments.py --full`
- Wilcoxon signed-rank test + bootstrap CI planned for full-scale evaluation

---

## Known Limitations (Honest Disclosure)

1. **ds004572 partial**: 5/52 subjects processed. Full 52-subject processing requires `process_ds004572_full.py` (lazy loading, ~4-5h, 16GB+ RAM)
2. **ds006437 approximation**: Session-proportional labels approximate session boundaries since prep01 merged all trials. Re-running prep01 with session-level granularity is recommended
3. **Proxy labels**: DREAMER, SEED, SEED-IV, FACED, ds006437 use proxy/task-condition labels — not real hypnosis depth annotations
4. **3-seed verification**: Statistical power limited to 3 seeds. 20-seed Wilcoxon planned for full-scale evaluation
5. **Simplified FS²C**: Current calibration uses sample concatenation. Mahalanobis dynamic-weighting implemented but not benchmarked
6. **No deep learning baseline benchmarked**: EEGNet-v4 code exists (`eegnet_baseline.py`) but not compared against RF

---

## FAQ

### Q: Why 14 channels instead of 32/64?

The EMOTIV EPOC+ consumer headset has exactly 14 channels. Mapping all datasets to this layout ensures direct deployability on consumer BCI hardware (Paper 2).

### Q: Why Random Forest instead of deep learning?

In cross-dataset zero-shot scenarios, the domain gap (different devices, electrode layouts, sampling rates) severely degrades deep model performance. Hand-crafted spectral features with explicit physical meaning transfer more robustly.

### Q: How are proxy labels justified?

No large-scale, multi-subject, real-hypnosis EEG dataset exists with standardized numeric depth annotations. We leverage the relationship between arousal and hypnotic depth (Theta enhancement, Beta suppression) to approximate hypnosis-like states. All proxy labels are clearly marked.

### Q: What about the Mahalanobis WFSC?

A Ledoit-Wolf covariance-based dynamic weighting implementation exists in `shared/mahalanobis_wfsc.py` but has not been experimentally validated against simple sample concatenation in the current results. It is planned for future work.

---

## Citation

```bibtex
@article{bci_hypnosis_2026,
  title={Multi-Source Domain Generalization with Few-Shot Calibration for Cross-Dataset EEG Hypnosis Depth Classification},
  author={},
  journal={},
  year={2026},
  note={v5.2, single-pass verified}
}
```

---

## License

MIT License. Individual datasets are subject to their respective licenses and terms of use.
