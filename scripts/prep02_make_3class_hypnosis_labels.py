#!/usr/bin/env python3
"""
prep02_extract_63feat_all.py — 63-dimensional feature extraction for all 8 datasets.

This script:
  1. Loads windowed EEG data from prep01 output
  2. Extracts 63-dimensional features per window:
     - [0:42]  = 14 channels x 3 bands Log-Bandpower (Theta, Alpha, Beta)
     - [42:63] = 7 asymmetry pairs x 3 bands DASM
  3. Saves features as .npz files to processed/prep02_features/

Input:  processed/prep01_windows/{dataset}_windows.npz
Output: processed/prep02_features/{dataset}_features.npz
        Each .npz contains: features (n_windows, 63), subject_ids, trial_ids
"""

import sys
import numpy as np
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.feature_extraction import (
    extract_features_window, FEAT_NAMES, subject_zscore,
    EPOC_CHANNELS, ASYM_PAIRS
)
from shared.logger import setup_logger


# Asymmetry pair indices in EPOC_CHANNELS order (used for DASM from precomputed DE)
_ch_idx = {ch: i for i, ch in enumerate(EPOC_CHANNELS)}
ASYM_INDICES = [(i, _ch_idx[l], _ch_idx[r]) for i, (l, r) in enumerate(ASYM_PAIRS)]


def build_63feat_from_precomputed_de(de_14_3):
    """
    Build 63-dim feature vector from pre-computed DE (Differential Entropy) features.

    Args:
        de_14_3: ndarray (14, 3) — DE values for 14 EPOC channels × 3 bands
                 (bands order: Theta, Alpha, Beta)
                 DE = 0.5 * log(2πe * variance); treated as log-bandpower proxy.

    Returns:
        feat: ndarray (63,) — matches FEAT_NAMES order
    """
    # FACED DE uses natural-log scale; convert to log10 scale to match other datasets.
    # log10(x) = log(x) / log(10), but DE = 0.5*ln(2πe*σ²) ≠ log10(power).
    # We apply: log_bp_proxy = de / ln(10) — keeps relative differences consistent.
    log_bp = de_14_3 / np.log(10)   # (14, 3)  log10-scale proxy

    feat_bp = log_bp.flatten()       # (42,)

    feat_dasm = np.zeros(len(ASYM_PAIRS) * 3)
    for pi, li, ri in ASYM_INDICES:
        for bi in range(3):
            feat_dasm[pi * 3 + bi] = log_bp[li, bi] - log_bp[ri, bi]

    feat = np.concatenate([feat_bp, feat_dasm])
    assert feat.shape[0] == 63
    return feat


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep02', str(PROJECT_ROOT / config['logs_dir']))

    prep01_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = config['features']['fs_target']
    all_datasets = ['DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV',
                    'FACED', 'ds004572', 'ds006437']

    summary = {}

    for dataset_name in all_datasets:
        windows_path = prep01_dir / f'{dataset_name}_windows.npz'

        if not windows_path.exists():
            logger.warning(f"{dataset_name}: No windows file found at {windows_path}. Skipping.")
            summary[dataset_name] = {'status': 'missing_prep01'}
            continue

        logger.info(f"Extracting features for {dataset_name}...")
        t0 = time.time()

        data = np.load(windows_path, allow_pickle=True)
        windows = data['windows']       # (n_windows, n_samples, 14)
        subj_ids = data['subject_ids']
        trial_ids = data['trial_ids']

        # Check for precomputed DE features (FACED dataset)
        precomputed_de = data['precomputed_de'] if 'precomputed_de' in data else None

        n_windows = windows.shape[0]
        logger.info(f"  {dataset_name}: {n_windows} windows, shape={windows.shape[1:]}")
        if precomputed_de is not None:
            logger.info(f"  {dataset_name}: using precomputed DE features, shape={precomputed_de.shape}")

        # Extract 63-dim features for each window
        all_features = []
        valid_indices = []

        for i in range(n_windows):
            try:
                # ── Precomputed DE fast path (FACED) ────────────────────────────
                if precomputed_de is not None:
                    de_14_3 = precomputed_de[i]  # (14, 3)
                    feat = build_63feat_from_precomputed_de(de_14_3)
                else:
                    # ── Standard Welch-based extraction ─────────────────────────
                    window = windows[i]  # (n_samples, 14)
                    feat = extract_features_window(window, fs=fs)

                all_features.append(feat)
                valid_indices.append(i)
            except Exception as e:
                if i < 5:  # Only log first few errors
                    logger.warning(f"  Window {i} extraction failed: {e}")
                continue

        if len(all_features) == 0:
            logger.error(f"  {dataset_name}: No features extracted!")
            summary[dataset_name] = {'status': 'error', 'error': 'no_features'}
            continue

        features = np.array(all_features)  # (n_valid_windows, 63)
        valid_subj_ids = np.array([subj_ids[i] for i in valid_indices])
        valid_trial_ids = np.array([trial_ids[i] for i in valid_indices])

        # Verify feature dimensions
        assert features.shape[1] == 63, f"Expected 63 features, got {features.shape[1]}"

        # Save
        save_path = out_dir / f'{dataset_name}_features.npz'
        np.savez_compressed(
            save_path,
            features=features,
            subject_ids=valid_subj_ids,
            trial_ids=valid_trial_ids,
            feature_names=np.array(FEAT_NAMES),
        )

        unique_subjs = list(set(str(s) for s in valid_subj_ids))
        elapsed = time.time() - t0

        # Basic feature statistics
        feat_stats = {
            'mean': float(features.mean()),
            'std': float(features.std()),
            'min': float(features.min()),
            'max': float(features.max()),
            'has_nan': bool(np.any(np.isnan(features))),
            'has_inf': bool(np.any(np.isinf(features))),
        }

        summary[dataset_name] = {
            'status': 'ok',
            'n_windows': int(features.shape[0]),
            'n_features': int(features.shape[1]),
            'n_subjects': len(unique_subjs),
            'feature_stats': feat_stats,
            'save_path': str(save_path),
            'elapsed_sec': round(elapsed, 1),
        }

        logger.info(f"  {dataset_name}: {features.shape} features extracted ({elapsed:.1f}s)")
        logger.info(f"    Stats: mean={feat_stats['mean']:.4f}, std={feat_stats['std']:.4f}")

        # Handle NaN/Inf warning
        if feat_stats['has_nan']:
            logger.warning(f"  {dataset_name}: Features contain NaN values!")
        if feat_stats['has_inf']:
            logger.warning(f"  {dataset_name}: Features contain Inf values!")

    # Save summary
    summary_path = out_dir / 'prep02_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save feature names for verification
    feat_names_path = out_dir / 'feat_names.json'
    with open(feat_names_path, 'w') as f:
        json.dump(FEAT_NAMES, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("prep02 complete. Summary:")
    for ds, info in summary.items():
        if isinstance(info, dict) and 'n_windows' in info:
            logger.info(f"  {ds}: {info['n_windows']} windows x {info.get('n_features', '?')} dims, "
                        f"{info.get('n_subjects', 0)} subjects")
    logger.info(f"Feature names saved to: {feat_names_path}")


if __name__ == '__main__':
    main()
