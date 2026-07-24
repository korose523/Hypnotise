#!/usr/bin/env python3
"""
exp101_rf_lodo_loso_zero_shot_vs_wfsc.py — LODO/LOSO RF baseline: zero-shot vs WFSC.

Experiment 101 (Paper 1, Core):
  - LODO: 8-fold, each dataset as target
  - Inner: LOSO/LOO subject/trial-level split
  - Compare: (a) Zero-shot RF ensemble, (b) WFSC-Mahalanobis, (c) WFSC-Fixed
  - Metrics: Accuracy, Balanced Accuracy, Macro-F1, Cohen's Kappa
  - 20 seeds for statistical robustness

Output: results/exp101_lodo_loso/exp101_results.json
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager, ALL_DATASETS
from shared.seed_manager import SeedManager
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
from shared.metrics import compute_all_metrics, print_metrics, aggregate_seeds
from shared.logger import setup_logger


def load_prep_data(dataset_name, prep_dir):
    """Load features and labels from prep02/prep03 output."""
    feat_path = prep_dir / 'prep02_features' / f'{dataset_name}_features.npz'
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'

    if not feat_path.exists() or not label_path.exists():
        return None, None, None, None

    feat_data = np.load(feat_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)

    features = feat_data['features']
    labels = label_data['labels']
    subj_ids = label_data['subject_ids']
    trial_ids = label_data['trial_ids']

    # Only keep valid labels
    valid = labels >= 0
    return features[valid], labels[valid], subj_ids[valid], trial_ids[valid]


def run_single_lodo_fold(target_domain, source_domains, config, seed, logger):
    """
    Run a single LODO fold: train on all sources, evaluate on target.

    Args:
        target_domain: str
        source_domains: list of str
        config: dict
        seed: int
        logger: logging.Logger

    Returns:
        dict with results for zero_shot, wfsc_mahalanobis, wfsc_fixed
    """
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))

    # Load target data
    X_target, y_target, target_subj_ids, target_trial_ids = \
        load_prep_data(target_domain, prep_dir)

    if X_target is None or len(X_target) == 0:
        logger.warning(f"  [{seed}] {target_domain}: No data available. Skipping.")
        return None

    # Load source data
    source_data = {}
    for src in source_domains:
        X_src, y_src, _, _ = load_prep_data(src, prep_dir)
        if X_src is not None and len(X_src) > 0:
            source_data[src] = (X_src, y_src)

    if len(source_data) == 0:
        logger.warning(f"  [{seed}] {target_domain}: No source data. Skipping.")
        return None

    # Get inner split for target domain
    try:
        split = sm.load_subject_split(target_domain, seed)
    except FileNotFoundError:
        logger.warning(f"  [{seed}] {target_domain}: No inner split found. Using all as test.")
        split = {
            'split_type': 'none',
            'calib_subjects': [],
            'test_subjects': list(set(str(s) for s in target_subj_ids)),
        }

    # Split target into calibration and test
    calib_subjs = split.get('calib_subjects', [])
    test_subjs = split.get('test_subjects', split.get('test_subjects', []))
    calib_idx = split.get('calib_indices', [])
    test_idx = split.get('test_indices', [])

    if split['split_type'] == 'LOSO' and len(calib_subjs) > 0:
        calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
        test_mask = ~calib_mask
    elif split['split_type'] == 'LOO' and len(calib_idx) > 0:
        calib_mask = np.zeros(len(X_target), dtype=bool)
        test_mask = np.ones(len(X_target), dtype=bool)
        calib_mask[calib_idx] = True
        test_mask[calib_idx] = False
    else:
        # Zero-shot: no calibration
        calib_mask = np.zeros(len(X_target), dtype=bool)
        test_mask = np.ones(len(X_target), dtype=bool)

    X_calib = X_target[calib_mask]
    y_calib = y_target[calib_mask]
    X_test = X_target[test_mask]
    y_test = y_target[test_mask]

    if len(X_test) == 0:
        logger.warning(f"  [{seed}] {target_domain}: No test samples. Skipping.")
        return None

    rf_params = {
        'n_estimators': config['model']['rf']['n_estimators'],
        'max_depth': config['model']['rf']['max_depth'],
        'min_samples_leaf': config['model']['rf']['min_samples_leaf'],
        'class_weight': config['model']['rf']['class_weight'],
        'n_jobs': config['model']['rf']['n_jobs'],
    }

    results = {
        'target': target_domain,
        'seed': seed,
        'n_calib': len(X_calib),
        'n_test': len(X_test),
        'n_sources': len(source_data),
        'split_type': split['split_type'],
    }

    # Method 1: WFSC-Mahalanobis
    try:
        wfsc_m = WFSC_Mahalanobis(random_state=seed)
        wfsc_m.fit(source_data, X_calib, y_calib)
        y_pred_m = wfsc_m.predict(X_test)
        y_proba_m = wfsc_m.predict_proba(X_test)
        metrics_m = compute_all_metrics(y_test, y_pred_m, y_proba_m)
        metrics_m['weights'] = wfsc_m.get_weights()
        results['wfsc_mahalanobis'] = metrics_m
    except Exception as e:
        logger.error(f"  [{seed}] {target_domain} WFSC-Mahalanobis failed: {e}")
        results['wfsc_mahalanobis'] = {'error': str(e)}

    # Method 2: WFSC-Fixed (uniform weights)
    try:
        wfsc_f = WFSC_Fixed(random_state=seed)
        wfsc_f.fit(source_data, X_calib, y_calib)
        y_pred_f = wfsc_f.predict(X_test)
        y_proba_f = wfsc_f.predict_proba(X_test)
        metrics_f = compute_all_metrics(y_test, y_pred_f, y_proba_f)
        results['wfsc_fixed'] = metrics_f
    except Exception as e:
        logger.error(f"  [{seed}] {target_domain} WFSC-Fixed failed: {e}")
        results['wfsc_fixed'] = {'error': str(e)}

    # Method 3: Zero-shot (no calibration)
    try:
        wfsc_z = WFSC_Mahalanobis(random_state=seed)
        wfsc_z.fit(source_data)  # No calibration data
        y_pred_z = wfsc_z.predict(X_test)
        y_proba_z = wfsc_z.predict_proba(X_test)
        metrics_z = compute_all_metrics(y_test, y_pred_z, y_proba_z)
        results['zero_shot'] = metrics_z
    except Exception as e:
        logger.error(f"  [{seed}] {target_domain} Zero-shot failed: {e}")
        results['zero_shot'] = {'error': str(e)}

    logger.info(f"  [{seed}] {target_domain}: "
                f"Zero-shot={results.get('zero_shot', {}).get('balanced_accuracy', 0):.4f}, "
                f"WFSC-Fixed={results.get('wfsc_fixed', {}).get('balanced_accuracy', 0):.4f}, "
                f"WFSC-Mahal={results.get('wfsc_mahalanobis', {}).get('balanced_accuracy', 0):.4f}")

    return results


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp101', str(PROJECT_ROOT / config['logs_dir']))

    seed_mgr = SeedManager(config['experiment']['seeds'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp101_lodo_loso')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {target: {seed: results}}

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n{'='*60}")
        logger.info(f"LODO Fold: Target={target_domain}, Sources={source_domains}")
        logger.info(f"{'='*60}")

        all_results[target_domain] = {}

        for seed in seed_mgr:
            seed_mgr.set_seed(seed)
            result = run_single_lodo_fold(
                target_domain, source_domains, config, seed, logger
            )
            if result is not None:
                all_results[target_domain][seed] = result

    # Aggregate across seeds
    summary = {}
    for target, seed_results in all_results.items():
        if not seed_results:
            continue

        for method in ['zero_shot', 'wfsc_fixed', 'wfsc_mahalanobis']:
            method_results = [
                r[method] for r in seed_results.values()
                if method in r and 'error' not in r[method]
            ]
            if method_results:
                agg = aggregate_seeds(method_results)
                summary[f'{target}_{method}'] = agg
                logger.info(f"{target} [{method}]: "
                            f"BAcc={agg.get('balanced_accuracy_mean', 0):.4f} "
                            f"+/- {agg.get('balanced_accuracy_std', 0):.4f}")

    # Save results
    results_path = out_dir / 'exp101_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_fold_per_seed': {
                k: {str(sk): sv for sk, sv in v.items()}
                for k, v in all_results.items()
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("exp101 complete.")


if __name__ == '__main__':
    main()
