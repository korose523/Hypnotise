#!/usr/bin/env python3
"""
exp103_wfsc_dynamic_mahalanobis_vs_fixedw.py — WFSC Dynamic Mahalanobis vs Fixed Weights.

Experiment 103 (Paper 1, WFSC Ablation):
  - Direct comparison of Mahalanobis dynamic weighting vs uniform fixed weighting
  - Analyzes how source domain weights change across LODO folds
  - Studies correlation between Mahalanobis distance and actual accuracy
  - Also tests: per-source single model performance (no ensemble)
  - 20 seeds x 8 targets

Output: results/exp103_mahal_vs_fixed/exp103_results.json
        results/exp103_mahal_vs_fixed/exp103_weight_analysis.json
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
from shared.metrics import compute_all_metrics, aggregate_seeds
from shared.logger import setup_logger


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
    logger = setup_logger('exp103', str(PROJECT_ROOT / config['logs_dir']))

    seed_mgr = SeedManager(config['experiment']['seeds'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp103_mahal_vs_fixed')
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    all_results = {}
    weight_analysis = {}  # Track weight evolution across seeds

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n{'='*50}")
        logger.info(f"Target: {target_domain}")

        # Load target data
        X_target, y_target, target_subj_ids = load_prep_data(target_domain, prep_dir)
        if X_target is None or len(X_target) < 10:
            logger.warning(f"  Skipping {target_domain}: insufficient data")
            continue

        # Load source data
        source_data = {}
        for src in source_domains:
            X_src, y_src, _ = load_prep_data(src, prep_dir)
            if X_src is not None and len(X_src) > 0:
                source_data[src] = (X_src, y_src)

        all_results[target_domain] = {}
        weight_analysis[target_domain] = {}

        for seed in seed_mgr:
            seed_mgr.set_seed(seed)

            # Calibration split (20% subjects)
            np.random.seed(seed)
            unique_subjs = list(set(str(s) for s in target_subj_ids))
            np.random.shuffle(unique_subjs)
            n_calib = max(1, int(len(unique_subjs) * 0.2))
            calib_subjs = unique_subjs[:n_calib]
            test_subjs = unique_subjs[n_calib:]

            calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
            test_mask = ~calib_mask
            X_calib = X_target[calib_mask]
            y_calib = y_target[calib_mask]
            X_test = X_target[test_mask]
            y_test = y_target[test_mask]

            if len(X_test) == 0:
                continue

            seed_results = {}

            # 1. WFSC-Mahalanobis
            wfsc_m = WFSC_Mahalanobis(random_state=seed, n_jobs=-1)
            wfsc_m.fit(source_data, X_calib, y_calib)
            y_pred_m = wfsc_m.predict(X_test)
            metrics_m = compute_all_metrics(y_test, y_pred_m, wfsc_m.predict_proba(X_test))
            seed_results['wfsc_mahalanobis'] = metrics_m

            # Record weight details
            wd = wfsc_m.get_weight_details()
            if 'weights' in wd:
                weight_analysis[target_domain][seed] = {
                    'weights': wd['weights'],
                    'distances': wd.get('distances', {}),
                    'method': wd.get('method', 'unknown'),
                }

            # 2. WFSC-Fixed
            wfsc_f = WFSC_Fixed(random_state=seed, n_jobs=-1)
            wfsc_f.fit(source_data, X_calib, y_calib)
            y_pred_f = wfsc_f.predict(X_test)
            metrics_f = compute_all_metrics(y_test, y_pred_f, wfsc_f.predict_proba(X_test))
            seed_results['wfsc_fixed'] = metrics_f

            # 3. Per-source single models (no ensemble)
            per_source = {}
            for src_name, (X_src, y_src) in source_data.items():
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X_src)
                    X_t = scaler.transform(X_test)
                    rf = RandomForestClassifier(
                        n_estimators=500, max_depth=20, min_samples_leaf=5,
                        class_weight='balanced', n_jobs=-1, random_state=seed
                    )
                    rf.fit(X_s, y_src)
                    y_pred_s = rf.predict(X_t)
                    ms = compute_all_metrics(y_test, y_pred_s)
                    per_source[src_name] = {
                        'balanced_accuracy': ms['balanced_accuracy'],
                        'accuracy': ms['accuracy'],
                        'macro_f1': ms['macro_f1'],
                    }
                except Exception:
                    per_source[src_name] = {'error': 'failed'}
            seed_results['per_source'] = per_source

            all_results[target_domain][seed] = seed_results

    # Aggregate and compare
    summary = {}
    for target, seed_results in all_results.items():
        mahal_list = [r['wfsc_mahalanobis'] for r in seed_results.values()
                      if 'wfsc_mahalanobis' in r]
        fixed_list = [r['wfsc_fixed'] for r in seed_results.values()
                      if 'wfsc_fixed' in r]

        if mahal_list and fixed_list:
            agg_m = aggregate_seeds(mahal_list)
            agg_f = aggregate_seeds(fixed_list)
            summary[target] = {
                'mahalanobis': agg_m,
                'fixed': agg_f,
                'diff_bacc': agg_m.get('balanced_accuracy_mean', 0) -
                            agg_f.get('balanced_accuracy_mean', 0),
                'n_seeds': len(seed_results),
            }

            logger.info(f"{target}: Mahal BA={agg_m.get('balanced_accuracy_mean', 0):.4f} "
                        f"vs Fixed BA={agg_f.get('balanced_accuracy_mean', 0):.4f} "
                        f"(diff={summary[target]['diff_bacc']:+.4f})")

    # Save results
    results_path = out_dir / 'exp103_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_seed': {
                k: {str(sk): sv for sk, sv in v.items()}
                for k, v in all_results.items()
            }
        }, f, indent=2, ensure_ascii=False, default=str)

    # Save weight analysis
    weight_path = out_dir / 'exp103_weight_analysis.json'
    with open(weight_path, 'w', encoding='utf-8') as f:
        json.dump(weight_analysis, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Weight analysis saved to: {weight_path}")
    logger.info("exp103 complete.")


if __name__ == '__main__':
    main()
