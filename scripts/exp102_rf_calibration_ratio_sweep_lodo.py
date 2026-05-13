#!/usr/bin/env python3
"""
exp102_rf_calibration_ratio_sweep_lodo.py — Calibration ratio sweep under LODO.

Experiment 102 (Paper 1, Calibration Study):
  - Sweep calibration ratios: [0, 0.01, 0.02, 0.05, 0.10, 0.20]
  - For each ratio, run LODO evaluation with WFSC-Mahalanobis
  - Goal: Determine minimum calibration needed for reliable cross-dataset transfer
  - 20 seeds x 6 ratios x 8 targets = 960 evaluations

Output: results/exp102_calib_sweep/exp102_results.json
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
from shared.wfsc import WFSC_Mahalanobis
from shared.metrics import compute_all_metrics, aggregate_seeds
from shared.logger import setup_logger


def load_prep_data(dataset_name, prep_dir):
    """Load features and labels from prep02/prep03 output."""
    feat_path = prep_dir / 'prep02_features' / f'{dataset_name}_features.npz'
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'

    if not feat_path.exists() or not label_path.exists():
        return None, None, None

    feat_data = np.load(feat_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)

    features = feat_data['features']
    labels = label_data['labels']
    subj_ids = label_data['subject_ids']
    valid = labels >= 0
    return features[valid], labels[valid], subj_ids[valid]


def get_calib_test_split(X, y, subj_ids, calib_ratio, seed):
    """Split target data by calibration ratio using stratified subject sampling."""
    np.random.seed(seed)

    unique_subjs = list(set(str(s) for s in subj_ids))
    subj_majority = {}
    for s in unique_subjs:
        mask = np.array([str(x) == s for x in subj_ids])
        s_labels = y[mask]
        s_labels = s_labels[s_labels >= 0]
        if len(s_labels) > 0:
            unique, counts = np.unique(s_labels, return_counts=True)
            subj_majority[s] = int(unique[np.argmax(counts)])
        else:
            subj_majority[s] = 0

    # Stratified split
    unique_labels = sorted(set(subj_majority.values()))
    calib_subjs = []
    test_subjs = []

    for label in unique_labels:
        label_subjs = [s for s in unique_subjs if subj_majority[s] == label]
        np.random.shuffle(label_subjs)
        n_calib = max(1, int(len(label_subjs) * calib_ratio)) if calib_ratio > 0 else 0
        calib_subjs.extend(label_subjs[:n_calib])
        test_subjs.extend(label_subjs[n_calib:])

    if calib_ratio == 0:
        test_subjs = list(unique_subjs)

    calib_mask = np.array([str(s) in calib_subjs for s in subj_ids])
    test_mask = np.array([str(s) in test_subjs for s in subj_ids])

    return X[calib_mask], y[calib_mask], X[test_mask], y[test_mask]


def run_single_eval(target_domain, source_domains, config, seed, calib_ratio, logger):
    """Run a single evaluation with specific calibration ratio."""
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    X_target, y_target, target_subj_ids = load_prep_data(target_domain, prep_dir)
    if X_target is None or len(X_target) < 10:
        return None

    # Load source data
    source_data = {}
    for src in source_domains:
        X_src, y_src, _ = load_prep_data(src, prep_dir)
        if X_src is not None and len(X_src) > 0:
            source_data[src] = (X_src, y_src)

    if len(source_data) == 0:
        return None

    # Split target
    X_calib, y_calib, X_test, y_test = get_calib_test_split(
        X_target, y_target, target_subj_ids, calib_ratio, seed
    )

    if len(X_test) == 0:
        return None

    # WFSC-Mahalanobis
    wfsc = WFSC_Mahalanobis(random_state=seed, n_jobs=-1)
    wfsc.fit(source_data, X_calib if len(X_calib) > 0 else None,
             y_calib if len(y_calib) > 0 else None)
    y_pred = wfsc.predict(X_test)
    y_proba = wfsc.predict_proba(X_test)

    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    metrics['n_calib'] = len(X_calib)
    metrics['n_test'] = len(X_test)
    metrics['n_sources'] = len(source_data)
    metrics['weights'] = wfsc.get_weights()

    return metrics


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp102', str(PROJECT_ROOT / config['logs_dir']))

    seed_mgr = SeedManager(config['experiment']['seeds'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()
    calib_ratios = config['experiment']['calib_ratios']

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp102_calib_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {target: {ratio: {seed: metrics}}}

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n--- LODO Target: {target_domain} ---")

        all_results[target_domain] = {}

        for cr in calib_ratios:
            all_results[target_domain][cr] = {}
            logger.info(f"  Calib ratio: {cr}")

            for seed in seed_mgr:
                seed_mgr.set_seed(seed)
                metrics = run_single_eval(
                    target_domain, source_domains, config, seed, cr, logger
                )
                if metrics is not None:
                    all_results[target_domain][cr][seed] = metrics

    # Aggregate: mean BA per target per ratio
    summary = {}
    for target, ratio_results in all_results.items():
        for cr, seed_results in ratio_results.items():
            if seed_results:
                agg = aggregate_seeds(list(seed_results.values()))
                key = f'{target}_cr{cr}'
                summary[key] = {
                    'calib_ratio': cr,
                    'target': target,
                    **agg,
                }
                logger.info(f"  {target} cr={cr}: "
                            f"BAcc={agg.get('balanced_accuracy_mean', 0):.4f} "
                            f"+/- {agg.get('balanced_accuracy_std', 0):.4f}")

    results_path = out_dir / 'exp102_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_seed': {
                k: {str(cr): {str(sk): sv for sk, sv in v.items()}
                    for cr, v in rv.items()}
                for k, rv in all_results.items()
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("exp102 complete.")


if __name__ == '__main__':
    main()
