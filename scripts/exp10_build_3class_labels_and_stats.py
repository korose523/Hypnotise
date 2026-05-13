#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp10_build_3class_labels_and_stats.py — Experiment 10: Build 3-class labels & statistics.

For each of the 8 datasets:
  1. Load raw data
  2. Map channels to 14 EPOC+ montage
  3. Extract 63-dim features
  4. Map native labels → 3-class (Awake/Light/Deep)
  5. Save processed data and statistics

Output:
  processed/{dataset}_14ch_63feat.npz
  results/exp10_stats/{dataset}_stats.json
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.feature_extraction import FeatureExtractor, map_channels_to_14
from shared.label_mapping import LabelMapper
from shared.metrics import CLASS_NAMES


def load_dreamer(cfg):
    """Load DREAMER dataset (14 Emotiv channels, arousal 1-5)."""
    data_path = Path(cfg['data_paths']['DREAMER'])
    # DREAMER uses .mat format with 14 Emotiv channels natively
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(data_path))
        # Adjust keys based on actual DREAMER structure
        eeg_data = mat['DREAMER'][0, 0]['EEG'][0, 0]['data']
        arousal = mat['DREAMER'][0, 0]['EEG'][0, 0]['arousal']
        valence = mat['DREAMER'][0, 0]['EEG'][0, 0]['valence']
        return eeg_data, arousal, valence
    except Exception as e:
        print(f"  [WARNING] Could not load DREAMER: {e}")
        return None, None, None


def load_deap(cfg):
    """Load DEAP dataset (32 BioSemi channels, arousal 1-9)."""
    data_path = Path(cfg['data_paths']['DEAP'])
    try:
        import pickle
        all_eeg = []
        all_arousal = []
        for i in range(1, 33):
            fname = data_path / f"s{i:02d}.dat"
            if fname.exists():
                with open(fname, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                all_eeg.append(data['data'][:32, :])  # 32 channels
                all_arousal.append(data['labels'][:, 0])  # arousal
        return np.array(all_eeg), np.array(all_arousal), None
    except Exception as e:
        print(f"  [WARNING] Could not load DEAP: {e}")
        return None, None, None


def load_mahnob(cfg):
    """Load MAHNOB dataset."""
    data_path = Path(cfg['data_paths']['MAHNOB'])
    print(f"  [INFO] MAHNOB data path: {data_path}")
    print(f"  [INFO] Place MAHNOB data at {data_path} and re-run")
    return None, None, None


def load_seed(cfg):
    """Load SEED dataset (62 channels, emotion -1/0/1)."""
    data_path = Path(cfg['data_paths']['SEED'])
    print(f"  [INFO] SEED data path: {data_path}")
    print(f"  [INFO] Place SEED data at {data_path} and re-run")
    return None, None, None


def load_seed_iv(cfg):
    """Load SEED-IV dataset."""
    data_path = Path(cfg['data_paths']['SEED_IV'])
    print(f"  [INFO] SEED-IV data path: {data_path}")
    print(f"  [INFO] Place SEED-IV data at {data_path} and re-run")
    return None, None, None


def load_faced(cfg):
    """Load FACED dataset."""
    data_path = Path(cfg['data_paths']['FACED'])
    print(f"  [INFO] FACED data path: {data_path}")
    print(f"  [INFO] Place FACED data at {data_path} and re-run")
    return None, None, None


def load_ds004572(cfg):
    """Load ds004572 (OpenNeuro, hypnosis depth 0-10)."""
    data_path = Path(cfg['data_paths']['ds004572'])
    print(f"  [INFO] ds004572 data path: {data_path}")
    print(f"  [INFO] Place ds004572 data at {data_path} and re-run")
    return None, None, None


def load_ds006437(cfg):
    """Load ds006437 (OpenNeuro, phase pre/during/post)."""
    data_path = Path(cfg['data_paths']['ds006437'])
    print(f"  [INFO] ds006437 data path: {data_path}")
    print(f"  [INFO] Place ds006437 data at {data_path} and re-run")
    return None, None, None


DATASET_LOADERS = {
    'DREAMER': load_dreamer,
    'DEAP': load_deap,
    'MAHNOB': load_mahnob,
    'SEED': load_seed,
    'SEED_IV': load_seed_iv,
    'FACED': load_faced,
    'ds004572': load_ds004572,
    'ds006437': load_ds006437,
}


def process_dataset(dataset_name, cfg, logger):
    """
    Load, map channels, extract features, and map labels for one dataset.

    Returns:
        features: ndarray or None
        labels: ndarray or None
        stats: dict
    """
    logger.info(f"Processing {dataset_name}...")

    stats = {
        'dataset': dataset_name,
        'status': 'pending',
        'n_subjects': 0,
        'n_samples': 0,
        'n_features': 63,
        'class_distribution': {},
    }

    # Load raw data
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        logger.error(f"No loader for dataset: {dataset_name}")
        stats['status'] = 'no_loader'
        return None, None, stats

    eeg_data, raw_labels, extra = loader(cfg)

    if eeg_data is None:
        stats['status'] = 'data_not_available'
        logger.warning(f"Data not available for {dataset_name} — skipping")
        return None, None, stats

    # Map channels to 14 EPOC+ montage
    ch_mapping_cfg = cfg.get('channel_mapping', {})
    dataset_ch_cfg = ch_mapping_cfg.get(dataset_name, {})

    if dataset_ch_cfg.get('mapping_type') == 'direct':
        # DREAMER: already 14 Emotiv channels
        logger.info(f"  {dataset_name}: direct channel mapping (14 channels)")
        mapped_data = eeg_data[:, :14] if eeg_data.ndim == 2 else eeg_data
    else:
        # Nearest-neighbor mapping
        original_channels = dataset_ch_cfg.get('original_channels')
        if original_channels:
            logger.info(f"  {dataset_name}: mapping {len(original_channels)} -> 14 channels")
            mapped_data, mapping_info = map_channels_to_14(
                eeg_data.reshape(-1, len(original_channels)),
                original_channels
            )
            stats['channel_mapping'] = mapping_info
        else:
            logger.warning(f"  {dataset_name}: no channel list in config, assuming 14ch")
            mapped_data = eeg_data

    # Extract 63-dim features
    feat_cfg = cfg.get('features', {})
    fs = feat_cfg.get('fs_target', 128)
    window_sec = feat_cfg.get('window_sec', 4.0)
    overlap = feat_cfg.get('overlap', 0.5)

    extractor = FeatureExtractor(fs=fs)

    all_features = []
    all_labels = []

    # Process each subject/trial
    n_subjects = eeg_data.shape[0] if eeg_data.ndim >= 2 else 1
    for subj in range(n_subjects):
        try:
            subj_eeg = mapped_data[subj] if mapped_data.ndim >= 3 else mapped_data
            features = extractor.extract_windows(subj_eeg, window_sec, overlap)
            if len(features) > 0:
                all_features.append(features)
                if raw_labels is not None:
                    label = raw_labels[subj] if hasattr(raw_labels, '__len__') else raw_labels
                    all_labels.extend([label] * len(features))
        except Exception as e:
            logger.warning(f"  Subject {subj} failed: {e}")
            continue

    if not all_features:
        stats['status'] = 'feature_extraction_failed'
        return None, None, stats

    features = np.vstack(all_features)
    stats['n_samples'] = features.shape[0]
    stats['n_subjects'] = n_subjects

    # Map labels to 3-class
    if all_labels:
        mapper = LabelMapper(cfg)
        labels, mask = mapper.map_labels(dataset_name, all_labels)
        stats['class_distribution'] = mapper.get_class_distribution(labels)
        stats['n_mapped'] = int(mask.sum())
        stats['n_unmapped'] = int((~mask).sum())
        valid = mask
        features = features[valid]
        labels = labels[valid]
    else:
        labels = None
        stats['class_distribution'] = 'no_labels'

    stats['status'] = 'success'
    logger.info(f"  {dataset_name}: {features.shape[0]} samples, "
                f"features: {features.shape[1] if features.ndim == 2 else 'N/A'}")

    return features, labels, stats


def main():
    """Main entry point for Experiment 10."""
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("exp10_build_3class", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("Experiment 10: Build 3-class labels & statistics")
    logger.info("=" * 60)

    processed_dir = Path(cfg['processed_dir'])
    stats_dir = Path(cfg['output_dir']) / 'exp10_stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for dataset_name in ['DREAMER', 'DEAP', 'MAHNOB', 'SEED',
                          'SEED_IV', 'FACED', 'ds004572', 'ds006437']:
        features, labels, stats = process_dataset(dataset_name, cfg, logger)

        all_stats[dataset_name] = stats

        # Save processed data
        if features is not None:
            save_path = processed_dir / f"{dataset_name}_14ch_63feat.npz"
            if labels is not None:
                np.savez_compressed(save_path, features=features, labels=labels)
            else:
                np.savez_compressed(save_path, features=features)
            logger.info(f"  Saved: {save_path}")

        # Save stats
        stats_path = stats_dir / f"{dataset_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

    # Save summary
    summary_path = stats_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Experiment 10 complete!")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
