#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp11_lodo_multisource_3class.py — Experiment 11: LODO Multi-Source 3-Class Classification.

Leave-One-Domain-Out (LODO) cross-dataset evaluation:
  - Each dataset takes turns as target domain
  - All other 7 datasets serve as source
  - Methods: Source-Only, CORAL, TCA, AdaBN, WFSC-Mahalanobis
  - 20 seeds x 8 targets = 160 evaluations per method

Output:
  results/exp11_lodo/{target}_{method}_seed{N}.json
  results/exp11_lodo/summary.json
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
from shared.feature_extraction import FeatureExtractor
from shared.label_mapping import LabelMapper
from shared.domain_adaptation import CORAL, TCA, AdaBN
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
from shared.metrics import compute_all_metrics, print_metrics, aggregate_seeds, paired_ttest


ALL_DATASETS = [
    'DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV',
    'FACED', 'ds004572', 'ds006437'
]

METHODS = ['source_only', 'coral', 'tca', 'adabn', 'wfsc_mahalanobis']


def load_processed_dataset(processed_dir, dataset_name):
    """Load a preprocessed dataset from .npz file."""
    path = Path(processed_dir) / f"{dataset_name}_14ch_63feat.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    features = data['features']
    labels = data['labels'] if 'labels' in data else None
    return features, labels


def evaluate_lodo(cfg, logger):
    """Run LODO evaluation across all targets and methods."""
    sm = SeedManager(cfg['experiment']['seeds'])
    split_mgr = SplitManager(cfg['splits_dir'])
    lodo_splits = split_mgr.load_lodo_splits()
    processed_dir = Path(cfg['processed_dir'])
    results_dir = Path(cfg['output_dir']) / 'exp11_lodo'
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_params = cfg['model']['rf']

    all_results = {}  # {method: {target: {seed: metrics_dict}}}

    for target in ALL_DATASETS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Target domain: {target}")
        logger.info(f"{'='*50}")

        # Load target domain data
        X_target, y_target = load_processed_dataset(processed_dir, target)
        if X_target is None or y_target is None:
            logger.warning(f"Target {target} data not available — skipping")
            continue

        # Generate subject-level splits for target
        unique_subjects = np.arange(X_target.shape[0] // 10)  # Approximate subjects
        if len(unique_subjects) < 5:
            unique_subjects = np.arange(max(5, X_target.shape[0] // 5))
        labels_per_subject = {s: int(np.median(y_target[s * 10:(s + 1) * 10]))
                              for s in unique_subjects
                              if s * 10 < len(y_target)}

        # Load all source domains
        source_domains = lodo_splits[target]['source_domains']

        for seed in sm:
            sm.set_seed(seed)

            try:
                split = split_mgr.generate_subject_splits(
                    target, unique_subjects, labels_per_subject, seed,
                    cfg['experiment']['calib_ratio']
                )
            except Exception as e:
                logger.error(f"Split generation failed for {target} seed {seed}: {e}")
                continue

            calib_idx = split['calib_subjects']
            test_idx = split['test_subjects']

            # Create calib/test indices (map subject IDs to sample indices)
            samples_per_subject = max(1, X_target.shape[0] // max(len(unique_subjects), 1))
            calib_samples = []
            test_samples = []
            for s in calib_idx:
                start = int(s) * samples_per_subject
                end = min(start + samples_per_subject, X_target.shape[0])
                calib_samples.extend(range(start, end))
            for s in test_idx:
                start = int(s) * samples_per_subject
                end = min(start + samples_per_subject, X_target.shape[0])
                test_samples.extend(range(start, end))

            calib_samples = [i for i in calib_samples if i < X_target.shape[0]]
            test_samples = [i for i in test_samples if i < X_target.shape[0]]

            if not calib_samples or not test_samples:
                continue

            X_calib = X_target[calib_samples]
            y_calib = y_target[calib_samples]
            X_test = X_target[test_samples]
            y_test = y_target[test_samples]

            # Load and combine source data
            X_source_all = []
            y_source_all = []
            for src in source_domains:
                X_src, y_src = load_processed_dataset(processed_dir, src)
                if X_src is not None and y_src is not None:
                    X_source_all.append(X_src)
                    y_source_all.append(y_src)

            if not X_source_all:
                logger.warning(f"No source data available for target {target}")
                continue

            X_source = np.vstack(X_source_all)
            y_source = np.vstack(y_source_all).flatten() if len(y_source_all) > 1 else y_source_all[0]

            # ============================================================
            # Method 1: Source-Only (no adaptation)
            # ============================================================
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            source_scaler = StandardScaler()
            X_source_scaled = source_scaler.fit_transform(X_source)
            X_test_scaled = source_scaler.transform(X_test)

            clf = RandomForestClassifier(**rf_params, random_state=seed)
            clf.fit(X_source_scaled, y_source)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)

            metrics = compute_all_metrics(y_test, y_pred, y_proba)
            metrics['method'] = 'source_only'
            metrics['target'] = target
            metrics['seed'] = seed

            all_results.setdefault('source_only', {}).setdefault(target, {})[seed] = metrics
            logger.info(f"  [{seed}] source_only → Acc={metrics['accuracy']:.4f} "
                        f"BA={metrics['balanced_accuracy']:.4f}")

            # ============================================================
            # Method 2: CORAL
            # ============================================================
            try:
                coral = CORAL()
                X_source_coral = coral.fit_transform(X_source, X_calib)
                X_test_coral = coral.transform(X_test)

                scaler_coral = StandardScaler()
                X_src_sc = scaler_coral.fit_transform(X_source_coral)
                X_test_sc = scaler_coral.transform(X_test_coral)

                clf_coral = RandomForestClassifier(**rf_params, random_state=seed)
                clf_coral.fit(X_src_sc, y_source)
                y_pred_coral = clf_coral.predict(X_test_sc)
                y_proba_coral = clf_coral.predict_proba(X_test_sc)

                metrics_coral = compute_all_metrics(y_test, y_pred_coral, y_proba_coral)
                metrics_coral['method'] = 'coral'
                metrics_coral['target'] = target
                metrics_coral['seed'] = seed

                all_results.setdefault('coral', {}).setdefault(target, {})[seed] = metrics_coral
                logger.info(f"  [{seed}] coral → Acc={metrics_coral['accuracy']:.4f} "
                            f"BA={metrics_coral['balanced_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"  CORAL failed for {target} seed {seed}: {e}")

            # ============================================================
            # Method 3: TCA
            # ============================================================
            try:
                tca = TCA(n_components=10, kernel_param=1.0)
                X_source_tca = tca.fit_transform(X_source, X_calib)
                X_test_tca = tca.transform(X_test)

                clf_tca = RandomForestClassifier(**rf_params, random_state=seed)
                clf_tca.fit(X_source_tca, y_source)
                y_pred_tca = clf_tca.predict(X_test_tca)

                metrics_tca = compute_all_metrics(y_test, y_pred_tca)
                metrics_tca['method'] = 'tca'
                metrics_tca['target'] = target
                metrics_tca['seed'] = seed

                all_results.setdefault('tca', {}).setdefault(target, {})[seed] = metrics_tca
                logger.info(f"  [{seed}] tca → Acc={metrics_tca['accuracy']:.4f} "
                            f"BA={metrics_tca['balanced_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"  TCA failed for {target} seed {seed}: {e}")

            # ============================================================
            # Method 4: AdaBN
            # ============================================================
            try:
                adabn = AdaBN()
                X_source_adabn = adabn.fit_transform(X_source, X_calib)
                X_test_adabn = adabn.transform(X_test)

                clf_adabn = RandomForestClassifier(**rf_params, random_state=seed)
                clf_adabn.fit(X_source_adabn, y_source)
                y_pred_adabn = clf_adabn.predict(X_test_adabn)

                metrics_adabn = compute_all_metrics(y_test, y_pred_adabn)
                metrics_adabn['method'] = 'adabn'
                metrics_adabn['target'] = target
                metrics_adabn['seed'] = seed

                all_results.setdefault('adabn', {}).setdefault(target, {})[seed] = metrics_adabn
                logger.info(f"  [{seed}] adabn → Acc={metrics_adabn['accuracy']:.4f} "
                            f"BA={metrics_adabn['balanced_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"  AdaBN failed for {target} seed {seed}: {e}")

            # ============================================================
            # Method 5: WFSC-Mahalanobis
            # ============================================================
            try:
                source_data = {'all_sources': (X_source, y_source)}
                calib_data = {'all_sources': X_calib}

                wfsc = WFSC_Mahalanobis(n_jobs=rf_params['n_jobs'], random_state=seed)
                wfsc.fit(source_data, calib_data, y_calib)
                y_pred_wfsc = wfsc.predict(X_test)
                y_proba_wfsc = wfsc.predict_proba(X_test)

                metrics_wfsc = compute_all_metrics(y_test, y_pred_wfsc, y_proba_wfsc)
                metrics_wfsc['method'] = 'wfsc_mahalanobis'
                metrics_wfsc['target'] = target
                metrics_wfsc['seed'] = seed

                all_results.setdefault('wfsc_mahalanobis', {}).setdefault(target, {})[seed] = metrics_wfsc
                logger.info(f"  [{seed}] wfsc → Acc={metrics_wfsc['accuracy']:.4f} "
                            f"BA={metrics_wfsc['balanced_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"  WFSC failed for {target} seed {seed}: {e}")

        # Save per-target results
        for method in METHODS:
            if method in all_results and target in all_results[method]:
                seed_results = list(all_results[method][target].values())
                agg = aggregate_seeds(seed_results)
                agg_path = results_dir / f"{target}_{method}_aggregated.json"
                with open(agg_path, 'w') as f:
                    json.dump(agg, f, indent=2, default=str)

    # Save full summary
    summary = {}
    for method in METHODS:
        if method in all_results:
            summary[method] = {}
            for target in all_results[method]:
                seed_results = list(all_results[method][target].values())
                summary[method][target] = aggregate_seeds(seed_results)

    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return all_results


def main():
    """Main entry point for Experiment 11."""
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("exp11_lodo", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("Experiment 11: LODO Multi-Source 3-Class Classification")
    logger.info("=" * 60)

    all_results = evaluate_lodo(cfg, logger)

    # Statistical significance tests
    logger.info("\nStatistical Significance Tests:")
    for target in ALL_DATASETS:
        for method in METHODS[1:]:  # Compare against source_only
            if ('source_only' in all_results and target in all_results['source_only'] and
                    method in all_results and target in all_results[method]):
                baseline = list(all_results['source_only'][target].values())
                compared = list(all_results[method][target].values())
                test_result = paired_ttest(baseline, compared, 'balanced_accuracy')
                logger.info(f"  {target} | {method}: p={test_result['p_value']:.4f} "
                            f"{'*' if test_result['significant'] else 'ns'}")

    logger.info("=" * 60)
    logger.info("Experiment 11 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
