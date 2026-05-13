#!/usr/bin/env python3
"""
prep04_generate_splits_lodo_loso.py — Generate all train/calib/test splits.

This script:
  1. Generates LODO (Leave-One-Domain-Out) splits — deterministic, 8 folds
  2. For each LODO target domain, generates inner LOSO/LOO splits
     using multiple seeds for statistical robustness
  3. Supports calibration ratio sweep [0, 0.05, 0.10, 0.20, 0.30, 0.50]
  4. All splits are saved to splits/ directory

CRITICAL: No experiment script may perform its own random split.
          All splits must come from this centralized module.

Output:
  splits/lodo_splits.json              — LODO fold definitions
  splits/subj_{dataset}_seed{n}.json   — Inner LOSO splits per target domain per seed
  splits/prep04_summary.json           — Generation summary
"""

import sys
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager, ALL_DATASETS
from shared.logger import setup_logger


def get_dataset_subject_info(dataset_name, prep_dir):
    """
    Extract unique subject IDs and their majority labels from prep03 data.
    """
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'

    if not label_path.exists():
        return [], {}, False

    data = np.load(label_path, allow_pickle=True)
    labels = data['labels']
    subj_ids = data['subject_ids']

    unique_subjs = sorted(set(str(s) for s in subj_ids))

    if len(unique_subjs) <= 1:
        return [], {}, False

    # Compute majority label per subject
    labels_per_subject = {}
    for s in unique_subjs:
        mask = np.array([str(x) == s for x in subj_ids])
        subj_labels = labels[mask]
        valid_labels = subj_labels[subj_labels >= 0]
        if len(valid_labels) > 0:
            unique, counts = np.unique(valid_labels, return_counts=True)
            labels_per_subject[s] = int(unique[np.argmax(counts)])
        else:
            labels_per_subject[s] = 0

    return unique_subjs, labels_per_subject, True


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep04', str(PROJECT_ROOT / config['logs_dir']))

    splits_dir = Path(PROJECT_ROOT / config['splits_dir'])
    splits_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    sm = SplitManager(str(splits_dir))

    seeds = config['experiment']['seeds']
    calib_ratios = config['experiment']['calib_ratios']

    summary = {}

    # Step 1: Generate LODO splits (deterministic)
    logger.info("Generating LODO splits (Leave-One-Domain-Out)...")
    lodo_splits = sm.generate_lodo_splits()
    summary['lodo'] = {
        'n_folds': len(lodo_splits),
        'target_domains': list(lodo_splits.keys()),
    }
    logger.info(f"  Generated {len(lodo_splits)} LODO folds")

    # Step 2: Generate inner splits for each target domain x seed
    logger.info(f"\nGenerating inner splits ({len(seeds)} seeds x {len(ALL_DATASETS)} datasets)...")

    for dataset_name in ALL_DATASETS:
        logger.info(f"\n--- {dataset_name} ---")

        subject_ids, labels_per_subj, has_subjects = \
            get_dataset_subject_info(dataset_name, prep_dir)

        if has_subjects:
            logger.info(f"  Using LOSO: {len(subject_ids)} subjects")
            for seed in seeds:
                sm.generate_loso_split(
                    dataset_name, subject_ids, labels_per_subj, seed,
                    calib_ratio=config['experiment']['default_calib_ratio']
                )
            summary[dataset_name] = {
                'split_type': 'LOSO',
                'n_subjects': len(subject_ids),
                'n_seeds': len(seeds),
            }
        else:
            logger.info(f"  No subject info — skipping inner splits")
            summary[dataset_name] = {'split_type': 'none'}

    # Step 3: Generate calibration ratio sweep splits
    logger.info("\nGenerating calibration ratio sweep splits...")
    for dataset_name in ALL_DATASETS:
        subject_ids, labels_per_subj, has_subjects = \
            get_dataset_subject_info(dataset_name, prep_dir)

        if not has_subjects:
            continue

        for seed in seeds[:3]:
            all_ratio_splits = sm.generate_calib_ratio_splits(
                dataset_name, subject_ids, labels_per_subj, seed, calib_ratios
            )

    # Save summary
    summary_path = splits_dir / 'prep04_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("prep04 complete.")
    logger.info(f"All splits saved to: {splits_dir}")


if __name__ == '__main__':
    main()
