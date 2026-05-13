#!/usr/bin/env python3
"""
exp105_real_da_baselines_coral_tca_adabn.py — Domain adaptation baselines: CORAL, TCA, AdaBN.

Experiment 105 (Paper 1, DA Baselines):
  - Compare traditional domain adaptation methods against WFSC
  - Methods: CORAL, TCA (Transfer Component Analysis), AdaBN
  - Protocol: LODO with inner LOSO calibration split
  - Each DA method aligns source features to target, then trains RF on aligned data
  - 20 seeds x 8 targets

Output: results/exp105_da_baselines/exp105_results.json
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
from shared.domain_adaptation import CORAL, TCA, AdaBN
from shared.metrics import compute_all_metrics, aggregate_seeds
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


def run_da_method(method_name, X_source, y_source, X_calib, X_test,
                  y_test, config, seed):
    """
    Run a domain adaptation method and evaluate.

    Args:
        method_name: str, one of 'CORAL', 'TCA', 'AdaBN'
        X_source: ndarray, combined source features
        y_source: ndarray, combined source labels
        X_calib: ndarray, target calibration features
        X_test: ndarray, target test features
        y_test: ndarray, target test labels
        config: dict
        seed: int

    Returns:
        dict: metrics
    """
    rf_params = config['model']['rf']

    if method_name == 'CORAL':
        da = CORAL()
        X_src_aligned = da.fit_transform(X_source, X_calib if len(X_calib) > 0 else X_test)
        X_test_aligned = da.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_src_aligned)
        X_eval = scaler.transform(X_test_aligned)

        # Include calibration if available
        if len(X_calib) > 0:
            X_calib_aligned = da.transform(X_calib)
            X_train = np.vstack([X_train, scaler.fit_transform(X_src_aligned)])
            X_calib_scaled = scaler.transform(X_calib_aligned)
            # Re-fit scaler on combined
            scaler2 = StandardScaler()
            X_train = scaler2.fit_transform(np.vstack([X_src_aligned, X_calib_aligned]))
            X_eval = scaler2.transform(X_test_aligned)
            y_train = np.concatenate([y_source, np.full(len(X_calib), -1)])
            # Only use source labels for training
            valid_mask = y_train >= 0
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
        else:
            y_train = y_source

    elif method_name == 'TCA':
        da = TCA(n_components=10, kernel_type='rbf', kernel_param=1.0, mu=0.1)
        da.fit(X_source, X_calib if len(X_calib) > 0 else X_test)
        X_src_proj = da.transform(X_source)
        X_test_proj = da.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_src_proj)
        X_eval = scaler.transform(X_test_proj)
        y_train = y_source

    elif method_name == 'AdaBN':
        da = AdaBN()
        da.fit(X_source, X_calib if len(X_calib) > 0 else X_test)

        X_src_norm = da.transform_source(X_source)
        X_test_norm = da.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_src_norm)
        X_eval = scaler.transform(X_test_norm)
        y_train = y_source

    else:
        raise ValueError(f"Unknown DA method: {method_name}")

    # Train RF on adapted features
    rf = RandomForestClassifier(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        class_weight=rf_params['class_weight'],
        n_jobs=rf_params['n_jobs'],
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_eval)

    try:
        y_proba = rf.predict_proba(X_eval)
    except Exception:
        y_proba = None

    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    return metrics


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp105', str(PROJECT_ROOT / config['logs_dir']))

    seed_mgr = SeedManager(config['experiment']['seeds'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp105_da_baselines')
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    da_methods = ['CORAL', 'TCA', 'AdaBN']
    all_results = {}

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n{'='*50}")
        logger.info(f"Target: {target_domain}")

        X_target, y_target, target_subj_ids = load_prep_data(target_domain, prep_dir)
        if X_target is None or len(X_target) < 10:
            logger.warning(f"  Skipping {target_domain}")
            continue

        # Combine all sources
        src_X_list, src_y_list = [], []
        for src in source_domains:
            X_s, y_s, _ = load_prep_data(src, prep_dir)
            if X_s is not None and len(X_s) > 0:
                src_X_list.append(X_s)
                src_y_list.append(y_s)

        if not src_X_list:
            continue

        X_source = np.vstack(src_X_list)
        y_source = np.concatenate(src_y_list)

        all_results[target_domain] = {}

        for seed in seed_mgr:
            seed_mgr.set_seed(seed)

            # Split target
            np.random.seed(seed)
            unique_subjs = list(set(str(s) for s in target_subj_ids))
            np.random.shuffle(unique_subjs)
            n_calib = max(1, int(len(unique_subjs) * 0.2))
            calib_subjs = set(unique_subjs[:n_calib])

            calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
            X_calib = X_target[calib_mask]
            X_test = X_target[~calib_mask]
            y_test = y_target[~calib_mask]

            if len(X_test) == 0:
                continue

            seed_results = {}
            for method in da_methods:
                try:
                    metrics = run_da_method(
                        method, X_source, y_source, X_calib, X_test, y_test,
                        config, seed
                    )
                    seed_results[method] = metrics
                except Exception as e:
                    logger.warning(f"  [{seed}] {target_domain} {method} failed: {e}")
                    seed_results[method] = {'error': str(e)}

            all_results[target_domain][seed] = seed_results

            # Log best method
            best_method = max(
                [(m, r) for m, r in seed_results.items() if 'error' not in r],
                key=lambda x: x[1].get('balanced_accuracy', 0),
                default=('none', {'balanced_accuracy': 0})
            )
            logger.info(f"  [{seed}] {target_domain}: best={best_method[0]} "
                        f"BAcc={best_method[1].get('balanced_accuracy', 0):.4f}")

    # Aggregate
    summary = {}
    for target, seed_results in all_results.items():
        for method in da_methods:
            method_results = [r[method] for r in seed_results.values()
                              if method in r and 'error' not in r[method]]
            if method_results:
                agg = aggregate_seeds(method_results)
                summary[f'{target}_{method}'] = agg
                logger.info(f"{target} [{method}]: "
                            f"BAcc={agg.get('balanced_accuracy_mean', 0):.4f} "
                            f"+/- {agg.get('balanced_accuracy_std', 0):.4f}")

    results_path = out_dir / 'exp105_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_seed': {
                k: {str(sk): sv for sk, sv in v.items()}
                for k, v in all_results.items()
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("exp105 complete.")


if __name__ == '__main__':
    main()
