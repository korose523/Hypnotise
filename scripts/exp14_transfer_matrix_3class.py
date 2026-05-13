#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp14_transfer_matrix_3class.py — Experiment 14: Cross-Dataset Transfer Matrix 3-Class.

Builds a comprehensive transfer matrix showing classification accuracy
for every source→target pair (8x8 matrix).

Also computes:
  - Transfer gain/loss compared to within-domain performance
  - Transfer difficulty ranking
  - Optimal source selection for each target

Output:
  results/exp14_transfer_matrix/transfer_matrix.json
  results/exp14_transfer_matrix/transfer_matrix.csv
  results/exp14_transfer_matrix/difficulty_ranking.json
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.split_manager import SplitManager
from shared.domain_adaptation import CORAL
from shared.metrics import compute_all_metrics, aggregate_seeds


ALL_DATASETS = [
    'DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV',
    'FACED', 'ds004572', 'ds006437'
]


def load_processed_dataset(processed_dir, dataset_name):
    """Load preprocessed data."""
    path = Path(processed_dir) / f"{dataset_name}_14ch_63feat.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    return data['features'], data.get('labels', None)


def run_transfer_matrix(cfg, logger):
    """Build the full 8x8 transfer matrix."""
    sm = SeedManager(cfg['experiment']['seeds'])
    split_mgr = SplitManager(cfg['splits_dir'])
    processed_dir = Path(cfg['processed_dir'])
    results_dir = Path(cfg['output_dir']) / 'exp14_transfer_matrix'
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_params = cfg['model']['rf']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Load all available datasets
    datasets = {}
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            datasets[d] = (X, y)
            logger.info(f"  Loaded {d}: {X.shape[0]} samples")
        else:
            logger.warning(f"  {d} not available")

    available = list(datasets.keys())
    if len(available) < 2:
        logger.error("Need at least 2 datasets for transfer matrix")
        return {}

    # Transfer matrix: {source} x {target} -> metrics
    transfer_matrix = {}
    within_domain = {}

    n_seeds_to_use = min(5, len(sm))  # Use 5 seeds for efficiency (8x8x5 = 320 runs)

    for i, source in enumerate(available):
        for j, target in enumerate(available):
            key = f"{source}_to_{target}"

            logger.info(f"\nTransfer: {source} → {target}")

            X_source, y_source = datasets[source]
            X_target, y_target = datasets[target]

            if source == target:
                # Within-domain evaluation
                seed_results = []
                for si, seed in enumerate(sm):
                    if si >= n_seeds_to_use:
                        break
                    sm.set_seed(seed)

                    # Simple train/test split
                    n = len(X_source)
                    indices = np.random.permutation(n)
                    n_train = int(0.8 * n)
                    train_idx, test_idx = indices[:n_train], indices[n_train:]

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_source[train_idx])
                    X_te = scaler.transform(X_source[test_idx])
                    y_tr = y_source[train_idx]
                    y_te = y_source[test_idx]

                    clf = RandomForestClassifier(**rf_params, random_state=seed)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    m = compute_all_metrics(y_te, y_pred, clf.predict_proba(X_te))
                    seed_results.append(m)

                agg = aggregate_seeds(seed_results)
                agg['source'] = source
                agg['target'] = target
                agg['type'] = 'within_domain'
                transfer_matrix[key] = agg
                within_domain[source] = agg.get('balanced_accuracy_mean', 0)

                logger.info(f"  Within-domain BA: {within_domain[source]:.4f}")

            else:
                # Cross-domain transfer
                seed_results = []

                # Generate subject splits for target
                samples_per_subject = max(1, X_target.shape[0] // max(5, X_target.shape[0] // 10))
                n_subjects = X_target.shape[0] // samples_per_subject
                unique_subjects = np.arange(max(n_subjects, 5))
                labels_per_sub = {
                    int(s): int(np.median(y_target[s * samples_per_subject:(s + 1) * samples_per_subject]))
                    for s in unique_subjects if s * samples_per_subject < len(y_target)
                }

                for si, seed in enumerate(sm):
                    if si >= n_seeds_to_use:
                        break
                    sm.set_seed(seed)

                    try:
                        split = split_mgr.generate_subject_splits(
                            target, unique_subjects, labels_per_sub, seed,
                            cfg['experiment']['calib_ratio']
                        )
                    except Exception:
                        continue

                    def _get_idx(subject_list):
                        idx = []
                        for s in subject_list:
                            start = int(s) * samples_per_subject
                            end = min(start + samples_per_subject, X_target.shape[0])
                            idx.extend(range(start, end))
                        return [i for i in idx if i < X_target.shape[0]]

                    test_idx = _get_idx(split['test_subjects'])
                    if not test_idx:
                        continue

                    X_test = X_target[test_idx]
                    y_test = y_target[test_idx]

                    # Train on source with CORAL adaptation
                    try:
                        calib_idx = _get_idx(split['calib_subjects'])
                        X_calib = X_target[calib_idx] if calib_idx else None

                        if X_calib is not None:
                            coral = CORAL()
                            X_src_adapted = coral.fit_transform(X_source, X_calib)
                            X_test_adapted = coral.transform(X_test)
                        else:
                            X_src_adapted = X_source
                            X_test_adapted = X_test

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_src_adapted)
                        X_te = scaler.transform(X_test_adapted)

                        clf = RandomForestClassifier(**rf_params, random_state=seed)
                        clf.fit(X_tr, y_source)
                        y_pred = clf.predict(X_te)
                        m = compute_all_metrics(y_test, y_pred, clf.predict_proba(X_te))
                        seed_results.append(m)
                    except Exception as e:
                        logger.error(f"  Transfer failed (seed {seed}): {e}")

                if seed_results:
                    agg = aggregate_seeds(seed_results)
                    agg['source'] = source
                    agg['target'] = target
                    agg['type'] = 'cross_domain'

                    # Compute transfer gain
                    if source in within_domain:
                        agg['transfer_gain'] = agg.get('balanced_accuracy_mean', 0) - within_domain[source]
                    else:
                        agg['transfer_gain'] = None

                    transfer_matrix[key] = agg
                    logger.info(f"  Cross-domain BA: {agg.get('balanced_accuracy_mean', 0):.4f}")
                else:
                    transfer_matrix[key] = {'source': source, 'target': target,
                                            'type': 'cross_domain', 'error': 'no_results'}

    # Build difficulty ranking
    difficulty = {}
    for target in available:
        cross_results = {
            k: v for k, v in transfer_matrix.items()
            if v.get('target') == target and v.get('type') == 'cross_domain'
            and 'balanced_accuracy_mean' in v
        }
        if cross_results:
            mean_ba = np.mean([v['balanced_accuracy_mean'] for v in cross_results.values()])
            difficulty[target] = {
                'mean_cross_domain_ba': float(mean_ba),
                'n_sources': len(cross_results),
                'ranking_note': 'Higher BA = easier to transfer into'
            }

    # Sort by difficulty (easiest first)
    difficulty_sorted = dict(sorted(
        difficulty.items(), key=lambda x: x[1]['mean_cross_domain_ba'], reverse=True
    ))

    # Save results
    matrix_path = results_dir / "transfer_matrix.json"
    with open(matrix_path, 'w') as f:
        json.dump(transfer_matrix, f, indent=2, default=str)

    # Save as CSV
    csv_path = results_dir / "transfer_matrix.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['source', 'target', 'type', 'BA_mean', 'BA_std', 'Acc_mean', 'F1_mean']
        writer.writerow(header)
        for key, val in transfer_matrix.items():
            writer.writerow([
                val.get('source', ''),
                val.get('target', ''),
                val.get('type', ''),
                f"{val.get('balanced_accuracy_mean', 0):.4f}",
                f"{val.get('balanced_accuracy_std', 0):.4f}",
                f"{val.get('accuracy_mean', 0):.4f}",
                f"{val.get('macro_f1_mean', 0):.4f}",
            ])

    # Save difficulty ranking
    ranking_path = results_dir / "difficulty_ranking.json"
    with open(ranking_path, 'w') as f:
        json.dump(difficulty_sorted, f, indent=2, default=str)

    # Print summary table
    logger.info("\nTransfer Matrix (Balanced Accuracy):")
    logger.info(f"  {'Source→Target':<20}", end='')
    for t in available:
        logger.info(f"  {t[:8]:>8}", end='')
    logger.info("")

    for source in available:
        logger.info(f"  {source:<20}", end='')
        for target in available:
            key = f"{source}_to_{target}"
            if key in transfer_matrix and 'balanced_accuracy_mean' in transfer_matrix[key]:
                ba = transfer_matrix[key]['balanced_accuracy_mean']
                logger.info(f"  {ba:>8.4f}", end='')
            else:
                logger.info(f"  {'N/A':>8}", end='')
        logger.info("")

    return transfer_matrix


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("exp14_transfer_matrix", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("Experiment 14: Cross-Dataset Transfer Matrix 3-Class")
    logger.info("=" * 60)

    transfer_matrix = run_transfer_matrix(cfg, logger)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 14 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
