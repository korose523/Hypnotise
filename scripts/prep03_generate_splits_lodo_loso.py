#!/usr/bin/env python3
"""
prep03_generate_splits_lodo_loso.py — Generate all train/calib/test splits for experiments.

This script:
  1. Generates LODO (Leave-One-Domain-Out) splits — deterministic, 8 folds
  2. For each LODO target domain, generates inner LOSO/LOO splits
     using multiple seeds for statistical robustness
  3. Supports calibration ratio sweep [0, 0.01, 0.02, 0.05, 0.10, 0.20]
  4. All splits are saved to splits/ directory and loaded by experiment scripts

CRITICAL: No experiment script may perform its own random split.
          All splits must come from this centralized module.

Output:
  splits/lodo_splits.json              — LODO fold definitions
  splits/loso_{dataset}_seed{n}.json   — Inner LOSO splits per target domain per seed
  splits/loo_{dataset}_seed{n}.json    — Inner LOO splits (fallback)
  splits/prep03_summary.json           — Generation summary
"""

import sys
import numpy as np
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager, ALL_DATASETS
from shared.logger import setup_logger


def get_dataset_subject_info(dataset_name, prep_dir):
    """
    Extract unique subject IDs and their majority labels from prep data.

    Args:
        dataset_name: str
        prep_dir: Path, directory containing prep01/prep02 output

    Returns:
        subject_ids: list of str
        labels_per_subject: dict {subject_id: majority_label}
        has_subject_info: bool, True if dataset has meaningful subject info
    """
    label_path = prep_dir / 'prep02_labels' / f'{dataset_name}_labels.npz'

    if not label_path.exists():
        print(f"  Warning: No labels for {dataset_name}. Using placeholder.")
        return [], {}, False

    data = np.load(label_path, allow_pickle=True)
    labels = data['labels']
    subj_ids = data['subject_ids']

    # Get unique subjects
    unique_subjs = sorted(set(str(s) for s in subj_ids))

    if len(unique_subjs) <= 1:
        # Not enough subjects for LOSO — use LOO (trial-level)
        return [], {}, False

    # Compute majority label per subject
    labels_per_subject = {}
    for s in unique_subjs:
        mask = np.array([str(x) == s for x in subj_ids])
        subj_labels = labels[mask]
        valid_labels = subj_labels[subj_labels >= 0]
        if len(valid_labels) > 0:
            # Majority class
            unique, counts = np.unique(valid_labels, return_counts=True)
            labels_per_subject[s] = int(unique[np.argmax(counts)])
        else:
            labels_per_subject[s] = 0  # Default to Awake

    return unique_subjs, labels_per_subject, True


def get_dataset_trial_info(dataset_name, prep_dir):
    """
    Extract trial-level info for LOO split (fallback when no subject info).

    Returns:
        n_trials: int
        labels: ndarray
        has_trial_info: bool
    """
    label_path = prep_dir / 'prep02_labels' / f'{dataset_name}_labels.npz'

    if not label_path.exists():
        return 0, np.array([]), False

    data = np.load(label_path, allow_pickle=True)
    labels = data['labels']
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    return len(valid_labels), valid_labels, len(valid_labels) > 0


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep03', str(PROJECT_ROOT / config['logs_dir']))

    splits_dir = Path(PROJECT_ROOT / config['splits_dir'])
    splits_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    sm = SplitManager(str(splits_dir))

    seeds = config['experiment']['seeds']
    calib_ratios = config['experiment']['calib_ratios']

    summary = {}

    # ------------------------------------------------------------------
    # Step 1: Generate LODO splits (deterministic)
    # ------------------------------------------------------------------
    logger.info("Generating LODO splits (Leave-One-Domain-Out)...")
    lodo_splits = sm.generate_lodo_splits()
    summary['lodo'] = {
        'n_folds': len(lodo_splits),
        'target_domains': list(lodo_splits.keys()),
    }
    logger.info(f"  Generated {len(lodo_splits)} LODO folds")
    for target, info in lodo_splits.items():
        logger.info(f"    Target: {target} <- Sources: {info['source_domains']}")

    # ------------------------------------------------------------------
    # Step 2: Generate inner splits for each target domain x seed
    # ------------------------------------------------------------------
    logger.info(f"\nGenerating inner splits ({len(seeds)} seeds x {len(ALL_DATASETS)} datasets)...")

    for dataset_name in ALL_DATASETS:
        logger.info(f"\n--- {dataset_name} ---")

        # Try subject-level (LOSO) first
        subject_ids, labels_per_subj, has_subjects = \
            get_dataset_subject_info(dataset_name, prep_dir)

        if has_subjects:
            logger.info(f"  Using LOSO: {len(subject_ids)} subjects")
            for seed in seeds:
                for cr in calib_ratios:
                    if cr == 0:
                        # Zero-shot split
                        sm.generate_calib_ratio_splits(
                            dataset_name, subject_ids, labels_per_subj, seed, [cr]
                        )
                    else:
                        sm.generate_loso_split(
                            dataset_name, subject_ids, labels_per_subj, seed, cr
                        )
            summary[dataset_name] = {
                'split_type': 'LOSO',
                'n_subjects': len(subject_ids),
                'n_seeds': len(seeds),
                'calib_ratios': calib_ratios,
            }
        else:
            # Fallback to trial-level (LOO)
            n_trials, trial_labels, has_trials = \
                get_dataset_trial_info(dataset_name, prep_dir)

            if has_trials:
                logger.info(f"  Using LOO: {n_trials} trials")
                for seed in seeds:
                    sm.generate_loo_split(
                        dataset_name, n_trials, trial_labels, seed,
                        calib_ratio=config['experiment']['default_calib_ratio']
                    )
                summary[dataset_name] = {
                    'split_type': 'LOO',
                    'n_trials': n_trials,
                    'n_seeds': len(seeds),
                }
            else:
                logger.warning(f"  No labels found for {dataset_name}. Skipping inner splits.")
                summary[dataset_name] = {
                    'split_type': 'none',
                    'reason': 'No label data available',
                }

    # ------------------------------------------------------------------
    # Step 3: Generate calibration ratio sweep splits (for exp102)
    # ------------------------------------------------------------------
    logger.info("\nGenerating calibration ratio sweep splits...")
    for dataset_name in ALL_DATASETS:
        subject_ids, labels_per_subj, has_subjects = \
            get_dataset_subject_info(dataset_name, prep_dir)

        if not has_subjects:
            continue

        for seed in seeds[:3]:  # Only first 3 seeds for ratio sweep preview
            all_ratio_splits = sm.generate_calib_ratio_splits(
                dataset_name, subject_ids, labels_per_subj, seed, calib_ratios
            )
            logger.info(f"  {dataset_name} seed={seed}: "
                        f"{len(all_ratio_splits)} ratio splits generated")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary_path = splits_dir / 'prep03_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("prep03 complete. Summary:")
    for ds, info in summary.items():
        if isinstance(info, dict):
            logger.info(f"  {ds}: type={info.get('split_type', '?')}, "
                        f"subjects/trials={info.get('n_subjects', info.get('n_trials', '?'))}")
    logger.info(f"All splits saved to: {splits_dir}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
