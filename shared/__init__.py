"""
shared — Shared utility modules for Universal BCI Hypnosis Classification.

Provides centralized configuration, seed management, logging, data splitting,
feature extraction, label mapping, domain adaptation, WFSC fusion, and metrics.
"""

from .config_loader import load_config
from .seed_manager import SeedManager
from .logger import setup_logger
from .split_manager import SplitManager, ALL_DATASETS
from .feature_extraction import (
    FeatureExtractor, map_channels_to_14, EPOC_CHANNELS,
    resample_to_target, resample_eeg, FEATURE_ORDER, FEAT_NAMES,
    ASYM_PAIRS, BAND_DEFS, BANDS, BAND_NAMES,
    BP_CHANNELS, BP_BANDS,
    extract_features_window, extract_features_sliding, subject_zscore,
)
from .label_mapping import LabelMapper
from .domain_adaptation import CORAL, TCA, AdaBN
from .wfsc import WFSC_Mahalanobis, WFSC_Fixed, make_wfsc, RF_PARAMS, make_rf
from .metrics import (
    compute_all_metrics, print_metrics, aggregate_seeds,
    paired_ttest, wilcoxon_test, bootstrap_ci,
    CLASS_NAMES, CLASS_NAMES_CN,
)

__all__ = [
    'load_config',
    'SeedManager',
    'setup_logger',
    'SplitManager',
    'ALL_DATASETS',
    'FeatureExtractor',
    'map_channels_to_14',
    'EPOC_CHANNELS',
    'resample_to_target',
    'resample_eeg',
    'FEATURE_ORDER',
    'FEAT_NAMES',
    'ASYM_PAIRS',
    'BAND_DEFS',
    'BANDS',
    'BAND_NAMES',
    'extract_features_window',
    'extract_features_sliding',
    'subject_zscore',
    'LabelMapper',
    'CORAL',
    'TCA',
    'AdaBN',
    'WFSC_Mahalanobis',
    'WFSC_Fixed',
    'make_wfsc',
    'RF_PARAMS',
    'make_rf',
    'compute_all_metrics',
    'print_metrics',
    'aggregate_seeds',
    'paired_ttest',
    'wilcoxon_test',
    'bootstrap_ci',
    'CLASS_NAMES',
    'CLASS_NAMES_CN',
]
