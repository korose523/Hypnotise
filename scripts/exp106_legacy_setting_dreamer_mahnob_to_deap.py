#!/usr/bin/env python3
"""
exp106_legacy_setting_dreamer_mahnob_to_deap.py — Legacy setting reproduction.

Experiment 106 (Paper 1, Legacy Reproduction):
  - Reproduce the classic setting: train on DREAMER+MAHNOB, test on DEAP
  - This mimics the original paper's experimental setup
  - Compare: WFSC-Mahalanobis vs single-source vs CORAL/TCA baselines
  - Uses the same 63-dim features and 3-class labels as all other experiments
  - 20 seeds for statistical comparison

Output: results/exp106_legacy/exp106_results.json
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
from shared.domain_adaptation import CORAL, TCA
from shared.metrics import compute_all_metrics, aggregate_seeds, wilcoxon_test
from shared.logger import setup_logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_prep_data(dataset_name, prep_dir):
    """Load features and labels from prep01/prep02 output."""
    feat_path = prep_dir / 'prep01_features' / f'{dataset_name}_features.npz'
    label_path = prep_dir / 'prep02_labels' / f'{dataset_name}_labels.npz'

    if not feat_path.exists() or not label_path.exists():
        return None, None, None

    feat_data = np.load(feat_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)

    features = feat_data['features']
    labels = label_data['labels']
    subj_ids = label_data['subject_ids']
    valid = labels >= 0
    return features[valid], labels[valid], subj_ids[valid]


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp106', str(PROJECT_ROOT / config['logs_dir']))

    seed_mgr = SeedManager(config['experiment']['seeds'])
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp106_legacy')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy setting: DREAMER + MAHNOB -> DEAP
    source_names = ['DREAMER', 'MAHNOB']
    target_name = 'DEAP'

    logger.info(f"Legacy Setting: Train on {source_names} -> Test on {target_name}")

    # Load data
    source_data = {}
    for src in source_names:
        X_s, y_s, _ = load_prep_data(src, prep_dir)
        if X_s is not None and len(X_s) > 0:
            source_data[src] = (X_s, y_s)
            logger.info(f"  Source {src}: {X_s.shape[0]} samples")

    X_target, y_target, target_subj_ids = load_prep_data(target_name, prep_dir)
    if X_target is None or len(X_target) < 10:
        logger.error(f"Target {target_name} has insufficient data. Aborting.")
        return

    logger.info(f"  Target {target_name}: {X_target.shape[0]} samples, "
                f"{len(set(str(s) for s in target_subj_ids))} subjects")

    all_results = {}
    rf_params = config['model']['rf']

    for seed in seed_mgr:
        seed_mgr.set_seed(seed)

        # Split target: 20% calibration, 80% test (by subject)
        np.random.seed(seed)
        unique_subjs = list(set(str(s) for s in target_subj_ids))
        np.random.shuffle(unique_subjs)
        n_calib = max(1, int(len(unique_subjs) * 0.2))
        calib_subjs = set(unique_subjs[:n_calib])

        calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
        X_calib = X_target[calib_mask]
        y_calib = y_target[calib_mask]
        X_test = X_target[~calib_mask]
        y_test = y_target[~calib_mask]

        if len(X_test) == 0:
            continue

        seed_results = {}

        # 1. WFSC-Mahalanobis
        try:
            wfsc_m = WFSC_Mahalanobis(random_state=seed, n_jobs=-1)
            wfsc_m.fit(source_data, X_calib, y_calib)
            y_pred = wfsc_m.predict(X_test)
            metrics = compute_all_metrics(y_test, y_pred, wfsc_m.predict_proba(X_test))
            metrics['weights'] = wfsc_m.get_weights()
            seed_results['wfsc_mahalanobis'] = metrics
        except Exception as e:
            logger.error(f"  [{seed}] WFSC-Mahalanobis failed: {e}")

        # 2. WFSC-Fixed
        try:
            wfsc_f = WFSC_Fixed(random_state=seed, n_jobs=-1)
            wfsc_f.fit(source_data, X_calib, y_calib)
            y_pred = wfsc_f.predict(X_test)
            metrics = compute_all_metrics(y_test, y_pred, wfsc_f.predict_proba(X_test))
            seed_results['wfsc_fixed'] = metrics
        except Exception as e:
            logger.error(f"  [{seed}] WFSC-Fixed failed: {e}")

        # 3. Single-source DREAMER
        try:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(source_data['DREAMER'][0])
            X_t = scaler.transform(X_test)
            rf = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                class_weight=rf_params['class_weight'],
                n_jobs=rf_params['n_jobs'], random_state=seed
            )
            rf.fit(X_s, source_data['DREAMER'][1])
            y_pred = rf.predict(X_t)
            seed_results['single_dreamer'] = compute_all_metrics(y_test, y_pred)
        except Exception as e:
            logger.error(f"  [{seed}] Single DREAMER failed: {e}")

        # 4. Single-source MAHNOB
        try:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(source_data['MAHNOB'][0])
            X_t = scaler.transform(X_test)
            rf = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                class_weight=rf_params['class_weight'],
                n_jobs=rf_params['n_jobs'], random_state=seed
            )
            rf.fit(X_s, source_data['MAHNOB'][1])
            y_pred = rf.predict(X_t)
            seed_results['single_mahnob'] = compute_all_metrics(y_test, y_pred)
        except Exception as e:
            logger.error(f"  [{seed}] Single MAHNOB failed: {e}")

        # 5. CORAL
        try:
            coral = CORAL()
            X_src_combined = np.vstack([v[0] for v in source_data.values()])
            y_src_combined = np.concatenate([v[1] for v in source_data.values()])
            X_src_aligned = coral.fit_transform(X_src_combined, X_calib)
            X_test_aligned = coral.transform(X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_src_aligned)
            X_eval = scaler.transform(X_test_aligned)
            rf = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                class_weight=rf_params['class_weight'],
                n_jobs=rf_params['n_jobs'], random_state=seed
            )
            rf.fit(X_train, y_src_combined)
            y_pred = rf.predict(X_eval)
            seed_results['coral'] = compute_all_metrics(y_test, y_pred)
        except Exception as e:
            logger.error(f"  [{seed}] CORAL failed: {e}")

        all_results[seed] = seed_results

        # Log summary
        best = max(
            [(m, r) for m, r in seed_results.items() if isinstance(r, dict) and 'error' not in r],
            key=lambda x: x[1].get('balanced_accuracy', 0),
            default=('none', {'balanced_accuracy': 0})
        )
        logger.info(f"  [{seed}] best={best[0]} BAcc={best[1].get('balanced_accuracy', 0):.4f}")

    # Aggregate
    summary = {}
    for method in ['wfsc_mahalanobis', 'wfsc_fixed', 'single_dreamer', 'single_mahnob', 'coral']:
        method_results = [r[method] for r in all_results.values()
                          if method in r and isinstance(r[method], dict)
                          and 'error' not in r[method]]
        if method_results:
            agg = aggregate_seeds(method_results)
            summary[method] = agg
            logger.info(f"\n{method}: "
                        f"BAcc={agg.get('balanced_accuracy_mean', 0):.4f} "
                        f"+/- {agg.get('balanced_accuracy_std', 0):.4f}")

    # Statistical comparison: WFSC-Mahalanobis vs best single-source
    mahal_results = [r['wfsc_mahalanobis'] for r in all_results.values()
                     if 'wfsc_mahalanobis' in r and 'error' not in r['wfsc_mahalanobis']]
    best_single_results = []
    for r in all_results.values():
        singles = {m: r[m] for m in ['single_dreamer', 'single_mahnob']
                   if m in r and isinstance(r[m], dict) and 'error' not in r[m]}
        if singles:
            best_single = max(singles.values(), key=lambda x: x.get('balanced_accuracy', 0))
            best_single_results.append(best_single)

    if mahal_results and best_single_results:
        wilcox = wilcoxon_test(mahal_results, best_single_results)
        summary['wilcoxon_wfsc_vs_best_single'] = wilcox
        logger.info(f"\nWilcoxon WFSC-Mahal vs Best-Single: "
                    f"W={wilcox['W_statistic']:.1f}, p={wilcox['p_value']:.4f}, "
                    f"significant={wilcox['significant']}")

    results_path = out_dir / 'exp106_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'setting': {'sources': source_names, 'target': target_name},
            'summary': summary,
            'per_seed': {str(k): v for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("exp106 complete.")


if __name__ == '__main__':
    main()
