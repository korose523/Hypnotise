"""
shared — Shared utility modules for Universal BCI Hypnosis Classification.

Provides centralized configuration, seed management, logging, data splitting,
feature extraction, label mapping, domain adaptation, WFSC fusion, and metrics.
"""

from .config_loader import load_config
from .seed_manager import SeedManager
from .logger import setup_logger
from .split_manager import SplitManager
from .feature_extraction import FeatureExtractor, map_channels_to_14
from .label_mapping import LabelMapper
from .domain_adaptation import CORAL, TCA, AdaBN
from .wfsc import WFSC_Mahalanobis, WFSC_Fixed
from .metrics import compute_all_metrics, print_metrics, aggregate_seeds, paired_ttest

__all__ = [
    'load_config',
    'SeedManager',
    'setup_logger',
    'SplitManager',
    'FeatureExtractor',
    'map_channels_to_14',
    'LabelMapper',
    'CORAL',
    'TCA',
    'AdaBN',
    'WFSC_Mahalanobis',
    'WFSC_Fixed',
    'compute_all_metrics',
    'print_metrics',
    'aggregate_seeds',
    'paired_ttest',
]
