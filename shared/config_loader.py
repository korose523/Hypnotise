"""
config_loader.py — Load and validate the global configuration file (config.yaml).
"""

import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load and validate the global configuration file.

    Args:
        config_path: str or Path, path to config.yaml

    Returns:
        dict: parsed configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Validate required top-level keys
    required_keys = ['output_dir', 'splits_dir', 'models_dir', 'logs_dir', 'processed_dir']
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required config key: '{key}'")

    # Create output directories
    for d in required_keys:
        Path(cfg[d]).mkdir(parents=True, exist_ok=True)

    # Create experiment result sub-directories
    result_subdirs = [
        'exp10_stats', 'exp11_lodo', 'exp12_baselines',
        'exp13_wfsc_ablation', 'exp14_transfer_matrix',
        'rt00_pipeline_check', 'rt01_realtime', 'rt02_online_eval',
        'rt03_realtime_ablation',
    ]
    for subdir in result_subdirs:
        (Path(cfg['output_dir']) / subdir).mkdir(parents=True, exist_ok=True)

    return cfg
