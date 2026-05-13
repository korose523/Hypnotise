#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp12_real_baselines_3class.py — Experiment 12: Real Baselines 3-Class Classification.

Compare WFSC against standard baselines:
  - Source-Only RF
  - Source-Only SVM
  - Source-Only LR (Logistic Regression)
  - MDA (Maximum Mean Discrepancy + RF)
  - CORAL + RF
  - TCA + RF
  - AdaBN + RF
  - WFSC-Mahalanobis (proposed)

Uses the same LODO splits as exp11 for fair comparison.

Output:
  results/exp12_baselines/{target}_{method}_seed{N}.json
  results/exp12_baselines/summary.json
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
from shared.domain_adaptation import CORAL, TCA, AdaBN
from shared.wfsc import WFSC_Mahalanobis
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


def run_baselines(cfg, logger):
    """Run all baseline methods with LODO evaluation."""
    sm = SeedManager(cfg['experiment']['seeds'])
    split_mgr = SplitManager(cfg['splits_dir'])
    lodo_splits = split_mgr.load_lodo_splits()
    processed_dir = Path(cfg['processed_dir'])
    results_dir = Path(cfg['output_dir']) / 'exp12_baselines'
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_params = cfg['model']['rf']
    all_results = {}

    BASELINE_METHODS = {
        'rf': 'Random Forest',
        'svm': 'SVM (RBF)',
        'lr': 'Logistic Regression',
        'coral_rf': 'CORAL + RF',
        'tca_rf': 'TCA + RF',
        'adabn_rf': 'AdaBN + RF',
        'wfsc': 'WFSC-Mahalanobis (Ours)',
    }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for target in ALL_DATASETS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Target: {target}")
        logger.info(f"{'='*50}")

        X_target, y_target = load_processed_dataset(processed_dir, target)
        if X_target is None:
            logger.warning(f"Target {target} not available — skipping")
            continue

        # Approximate subject structure
        samples_per_subject = max(1, X_target.shape[0] // max(5, X_target.shape[0] // 10))
        n_subjects = X_target.shape[0] // samples_per_subject
        unique_subjects = np.arange(max(n_subjects, 5))
        labels_per_subject = {
            int(s): int(np.median(y_target[s * samples_per_subject:(s + 1) * samples_per_subject]))
            for s in unique_subjects
            if s * samples_per_subject < len(y_target)
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

            def _get_indices(subject_list):
                indices = []
                for s in subject_list:
                    start = int(s) * samples_per_subject
                    end = min(start + samples_per_subject, X_target.shape[0])
                    indices.extend(range(start, end))
                return [i for i in indices if i < X_target.shape[0]]

            calib_idx = _get_indices(split['calib_subjects'])
            test_idx = _get_indices(split['test_subjects'])

            if not calib_idx or not test_idx:
                continue

            X_calib, y_calib = X_target[calib_idx], y_target[calib_idx]
            X_test, y_test = X_target[test_idx], y_target[test_idx]

            # Load source data
            X_src_list, y_src_list = [], []
            for src in source_domains:
                X_s, y_s = load_processed_dataset(processed_dir, src)
                if X_s is not None and y_s is not None:
                    X_src_list.append(X_s)
                    y_src_list.append(y_s)

            if not X_src_list:
                continue

            X_source = np.vstack(X_src_list)
            y_source = np.concatenate(y_src_list)

            # --- RF Baseline ---
            scaler = StandardScaler()
            X_s_sc = scaler.fit_transform(X_source)
            X_t_sc = scaler.transform(X_test)
            clf = RandomForestClassifier(**rf_params, random_state=seed)
            clf.fit(X_s_sc, y_source)
            y_pred = clf.predict(X_t_sc)
            m = compute_all_metrics(y_test, y_pred, clf.predict_proba(X_t_sc))
            m.update({'method': 'rf', 'target': target, 'seed': seed})
            all_results.setdefault('rf', {}).setdefault(target, {})[seed] = m

            # --- SVM Baseline ---
            svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                      probability=True, random_state=seed)
            svm.fit(X_s_sc, y_source)
            y_pred_svm = svm.predict(X_t_sc)
            m_svm = compute_all_metrics(y_test, y_pred_svm, svm.predict_proba(X_t_sc))
            m_svm.update({'method': 'svm', 'target': target, 'seed': seed})
            all_results.setdefault('svm', {}).setdefault(target, {})[seed] = m_svm

            # --- LR Baseline ---
            lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                                    multi_class='multinomial', random_state=seed)
            lr.fit(X_s_sc, y_source)
            y_pred_lr = lr.predict(X_t_sc)
            m_lr = compute_all_metrics(y_test, y_pred_lr, lr.predict_proba(X_t_sc))
            m_lr.update({'method': 'lr', 'target': target, 'seed': seed})
            all_results.setdefault('lr', {}).setdefault(target, {})[seed] = m_lr

            # --- CORAL + RF ---
            try:
                coral = CORAL()
                X_s_coral = coral.fit_transform(X_source, X_calib)
                X_t_coral = coral.transform(X_test)
                sc_c = StandardScaler()
                clf_c = RandomForestClassifier(**rf_params, random_state=seed)
                clf_c.fit(sc_c.fit_transform(X_s_coral), y_source)
                y_pred_c = clf_c.predict(sc_c.transform(X_t_coral))
                m_c = compute_all_metrics(y_test, y_pred_c, clf_c.predict_proba(sc_c.transform(X_t_coral)))
                m_c.update({'method': 'coral_rf', 'target': target, 'seed': seed})
                all_results.setdefault('coral_rf', {}).setdefault(target, {})[seed] = m_c
            except Exception as e:
                logger.error(f"  CORAL+RF failed: {e}")

            # --- TCA + RF ---
            try:
                tca = TCA(n_components=10)
                X_s_tca = tca.fit_transform(X_source, X_calib)
                X_t_tca = tca.transform(X_test)
                clf_t = RandomForestClassifier(**rf_params, random_state=seed)
                clf_t.fit(X_s_tca, y_source)
                y_pred_t = clf_t.predict(X_t_tca)
                m_t = compute_all_metrics(y_test, y_pred_t)
                m_t.update({'method': 'tca_rf', 'target': target, 'seed': seed})
                all_results.setdefault('tca_rf', {}).setdefault(target, {})[seed] = m_t
            except Exception as e:
                logger.error(f"  TCA+RF failed: {e}")

            # --- AdaBN + RF ---
            try:
                adabn = AdaBN()
                X_s_ada = adabn.fit_transform(X_source, X_calib)
                X_t_ada = adabn.transform(X_test)
                clf_a = RandomForestClassifier(**rf_params, random_state=seed)
                clf_a.fit(X_s_ada, y_source)
                y_pred_a = clf_a.predict(X_t_ada)
                m_a = compute_all_metrics(y_test, y_pred_a)
                m_a.update({'method': 'adabn_rf', 'target': target, 'seed': seed})
                all_results.setdefault('adabn_rf', {}).setdefault(target, {})[seed] = m_a
            except Exception as e:
                logger.error(f"  AdaBN+RF failed: {e}")

            # --- WFSC-Mahalanobis (Ours) ---
            try:
                wfsc = WFSC_Mahalanobis(n_jobs=rf_params['n_jobs'], random_state=seed)
                wfsc.fit({'src': (X_source, y_source)}, {'src': X_calib})
                y_pred_w = wfsc.predict(X_test)
                y_prob_w = wfsc.predict_proba(X_test)
                m_w = compute_all_metrics(y_test, y_pred_w, y_prob_w)
                m_w.update({'method': 'wfsc', 'target': target, 'seed': seed})
                all_results.setdefault('wfsc', {}).setdefault(target, {})[seed] = m_w
            except Exception as e:
                logger.error(f"  WFSC failed: {e}")

    # Aggregate and save
    summary = {}
    for method, desc in BASELINE_METHODS.items():
        if method in all_results:
            summary[method] = {'description': desc}
            for target in all_results[method]:
                seed_results = list(all_results[method][target].values())
                agg = aggregate_seeds(seed_results)
                summary[method][target] = agg

    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return all_results


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("exp12_baselines", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("Experiment 12: Real Baselines 3-Class Classification")
    logger.info("=" * 60)

    all_results = run_baselines(cfg, logger)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 12 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
