#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp13_wfsc_ablation_3class.py — Experiment 13: WFSC Ablation Study 3-Class.

Ablation experiments to validate each component of WFSC:
  1. WFSC-Mahalanobis (full model)
  2. WFSC-Fixed (uniform weights)
  3. Single-Best-Source (oracle: pick best single source per target)
  4. Majority Voting (unweighted ensemble voting)
  5. Stacking (meta-learner on top of source models)

Uses LODO evaluation with 20 seeds.

Output:
  results/exp13_wfsc_ablation/{target}_{variant}_seed{N}.json
  results/exp13_wfsc_ablation/summary.json
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.split_manager import SplitManager
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
from shared.metrics import compute_all_metrics, aggregate_seeds, paired_ttest


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


def run_ablation(cfg, logger):
    """Run ablation experiments."""
    sm = SeedManager(cfg['experiment']['seeds'])
    split_mgr = SplitManager(cfg['splits_dir'])
    lodo_splits = split_mgr.load_lodo_splits()
    processed_dir = Path(cfg['processed_dir'])
    results_dir = Path(cfg['output_dir']) / 'exp13_wfsc_ablation'
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_params = cfg['model']['rf']
    all_results = {}

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    ABLATION_VARIANTS = [
        'wfsc_mahalanobis',   # Full model
        'wfsc_fixed',         # Uniform weights
        'single_best_source', # Oracle: best single source
        'majority_voting',    # Unweighted ensemble voting
        'stacking',           # Meta-learner
    ]

    for target in ALL_DATASETS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Target: {target}")
        logger.info(f"{'='*50}")

        X_target, y_target = load_processed_dataset(processed_dir, target)
        if X_target is None:
            continue

        samples_per_subject = max(1, X_target.shape[0] // max(5, X_target.shape[0] // 10))
        n_subjects = X_target.shape[0] // samples_per_subject
        unique_subjects = np.arange(max(n_subjects, 5))
        labels_per_subject = {
            int(s): int(np.median(y_target[s * samples_per_subject:(s + 1) * samples_per_subject]))
            for s in unique_subjects if s * samples_per_subject < len(y_target)
        }

        source_domains = lodo_splits[target]['source_domains']

        for seed in sm:
            sm.set_seed(seed)

            try:
                split = split_mgr.generate_subject_splits(
                    target, unique_subjects, labels_per_subject, seed,
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

            calib_idx = _get_idx(split['calib_subjects'])
            test_idx = _get_idx(split['test_subjects'])
            if not calib_idx or not test_idx:
                continue

            X_calib, y_calib = X_target[calib_idx], y_target[calib_idx]
            X_test, y_test = X_target[test_idx], y_target[test_idx]

            # Load source data
            source_models = {}
            source_scalers = {}
            for src in source_domains:
                X_s, y_s = load_processed_dataset(processed_dir, src)
                if X_s is not None and y_s is not None:
                    scaler = StandardScaler()
                    X_s_sc = scaler.fit_transform(X_s)
                    clf = RandomForestClassifier(**rf_params, random_state=seed)
                    clf.fit(X_s_sc, y_s)
                    source_models[src] = clf
                    source_scalers[src] = scaler

            if not source_models:
                continue

            # ---- Ablation 1: WFSC-Mahalanobis (full) ----
            try:
                source_data = {s: (np.vstack(
                    [load_processed_dataset(processed_dir, s)[0]]),
                    load_processed_dataset(processed_dir, s)[1])
                    for s in source_domains
                    if load_processed_dataset(processed_dir, s)[0] is not None}

                wfsc_full = WFSC_Mahalanobis(n_jobs=rf_params['n_jobs'], random_state=seed)
                wfsc_full.fit(source_data, {s: X_calib for s in source_data})
                y_pred = wfsc_full.predict(X_test)
                y_proba = wfsc_full.predict_proba(X_test)
                m = compute_all_metrics(y_test, y_pred, y_proba)
                m.update({'variant': 'wfsc_mahalanobis', 'target': target, 'seed': seed})
                all_results.setdefault('wfsc_mahalanobis', {}).setdefault(target, {})[seed] = m
            except Exception as e:
                logger.error(f"  WFSC-Mahalanobis failed: {e}")

            # ---- Ablation 2: WFSC-Fixed (uniform) ----
            try:
                wfsc_fixed = WFSC_Fixed(n_jobs=rf_params['n_jobs'], random_state=seed)
                wfsc_fixed.fit(source_data)
                y_pred_f = wfsc_fixed.predict(X_test)
                m_f = compute_all_metrics(y_test, y_pred_f)
                m_f.update({'variant': 'wfsc_fixed', 'target': target, 'seed': seed})
                all_results.setdefault('wfsc_fixed', {}).setdefault(target, {})[seed] = m_f
            except Exception as e:
                logger.error(f"  WFSC-Fixed failed: {e}")

            # ---- Ablation 3: Single-Best-Source (oracle) ----
            best_acc = -1
            best_m = None
            for src, clf in source_models.items():
                scaler = source_scalers[src]
                X_t_sc = scaler.transform(X_test)
                y_pred_s = clf.predict(X_t_sc)
                m_s = compute_all_metrics(y_test, y_pred_s)
                if m_s['accuracy'] > best_acc:
                    best_acc = m_s['accuracy']
                    m_s.update({'variant': 'single_best_source', 'target': target, 'seed': seed})
                    best_m = m_s
            if best_m:
                all_results.setdefault('single_best_source', {}).setdefault(target, {})[seed] = best_m

            # ---- Ablation 4: Majority Voting ----
            proba_sum = None
            for src, clf in source_models.items():
                scaler = source_scalers[src]
                X_t_sc = scaler.transform(X_test)
                p = clf.predict_proba(X_t_sc)
                if proba_sum is None:
                    proba_sum = np.zeros_like(p)
                proba_sum += p
            if proba_sum is not None:
                proba_avg = proba_sum / len(source_models)
                y_pred_mv = np.argmax(proba_avg, axis=1)
                m_mv = compute_all_metrics(y_test, y_pred_mv, proba_avg)
                m_mv.update({'variant': 'majority_voting', 'target': target, 'seed': seed})
                all_results.setdefault('majority_voting', {}).setdefault(target, {})[seed] = m_mv

            # ---- Ablation 5: Stacking ----
            try:
                # Use calibration set predictions as meta-features
                meta_features = []
                for src, clf in source_models.items():
                    scaler = source_scalers[src]
                    X_c_sc = scaler.transform(X_calib)
                    proba_calib = clf.predict_proba(X_c_sc)
                    meta_features.append(proba_calib)
                meta_train = np.hstack(meta_features)

                meta_features_test = []
                for src, clf in source_models.items():
                    scaler = source_scalers[src]
                    X_t_sc = scaler.transform(X_test)
                    proba_test = clf.predict_proba(X_t_sc)
                    meta_features_test.append(proba_test)
                meta_test = np.hstack(meta_features_test)

                meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                              multi_class='multinomial', random_state=seed)
                meta_clf.fit(meta_train, y_calib)
                y_pred_stack = meta_clf.predict(meta_test)
                m_stack = compute_all_metrics(y_test, y_pred_stack, meta_clf.predict_proba(meta_test))
                m_stack.update({'variant': 'stacking', 'target': target, 'seed': seed})
                all_results.setdefault('stacking', {}).setdefault(target, {})[seed] = m_stack
            except Exception as e:
                logger.error(f"  Stacking failed: {e}")

    # Save summary
    summary = {}
    for variant in ABLATION_VARIANTS:
        if variant in all_results:
            summary[variant] = {}
            for target in all_results[variant]:
                summary[variant][target] = aggregate_seeds(
                    list(all_results[variant][target].values())
                )

    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return all_results


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("exp13_wfsc_ablation", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("Experiment 13: WFSC Ablation Study 3-Class")
    logger.info("=" * 60)

    all_results = run_ablation(cfg, logger)

    # Print ablation comparison table
    logger.info("\nAblation Comparison (Balanced Accuracy):")
    for target in ALL_DATASETS:
        row = f"  {target:<12}"
        for variant in ['wfsc_mahalanobis', 'wfsc_fixed', 'single_best_source',
                         'majority_voting', 'stacking']:
            if (variant in all_results and target in all_results[variant] and
                    all_results[variant][target]):
                agg = aggregate_seeds(list(all_results[variant][target].values()))
                ba = agg.get('balanced_accuracy_mean', 0)
                row += f"  {ba:.4f}"
            else:
                row += f"  N/A"
        logger.info(row)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 13 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
