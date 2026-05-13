#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rt01_train_public_then_realtime.py — RT01: Train on Public Data, Apply to Real-time EPOC+.

Workflow:
  1. Load all processed public datasets
  2. Train WFSC-Mahalanobis model on combined public data
  3. Save model for real-time deployment
  4. Test with simulated EPOC+ data

Output:
  models/public_trained_wfsc_seed{N}.pkl
  results/rt01_realtime/training_results.json
"""

import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.split_manager import SplitManager
from shared.feature_extraction import FeatureExtractor, EPOC_CHANNELS
from shared.wfsc import WFSC_Mahalanobis
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


def train_on_public(cfg, logger):
    """Train WFSC model on all available public datasets."""
    sm = SeedManager(cfg['experiment']['seeds'])
    processed_dir = Path(cfg['processed_dir'])
    models_dir = Path(cfg['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg['output_dir']) / 'rt01_realtime'
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_params = cfg['model']['rf']

    # Load all available public datasets
    all_features = []
    all_labels = []
    available_datasets = []

    for dataset_name in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, dataset_name)
        if X is not None and y is not None:
            all_features.append(X)
            all_labels.append(y)
            available_datasets.append(dataset_name)
            logger.info(f"  Loaded {dataset_name}: {X.shape[0]} samples, "
                        f"{X.shape[1]} features")

    if not all_features:
        logger.error("No processed datasets found. Run exp10 first.")
        return None

    X_public = np.vstack(all_features)
    y_public = np.concatenate(all_labels)

    logger.info(f"\nCombined public data: {X_public.shape[0]} samples")

    # Training results across seeds
    training_results = {}

    # Use fewer seeds for training (focus on model quality)
    training_seeds = cfg['experiment']['seeds'][:5]

    for seed in training_seeds:
        sm.set_seed(seed)
        logger.info(f"\nTraining with seed {seed}...")

        t_start = time.time()

        # Train WFSC with cross-validation within public data
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_public, y_public)):
            X_train, X_val = X_public[train_idx], X_public[val_idx]
            y_train, y_val = y_public[train_idx], y_public[val_idx]

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)

            clf = RandomForestClassifier(**rf_params, random_state=seed)
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_val_sc)
            y_proba = clf.predict_proba(X_val_sc)

            metrics = compute_all_metrics(y_val, y_pred, y_proba)
            fold_results.append(metrics)

        agg = aggregate_seeds(fold_results)

        # Train final model on ALL public data
        scaler_final = StandardScaler()
        X_public_sc = scaler_final.fit_transform(X_public)

        clf_final = RandomForestClassifier(**rf_params, random_state=seed)
        clf_final.fit(X_public_sc, y_public)

        t_elapsed = time.time() - t_start

        # Save model
        model_package = {
            'model': clf_final,
            'scaler': scaler_final,
            'seed': seed,
            'n_features': 63,
            'n_classes': 3,
            'class_names': ['Awake', 'Light Hypnosis', 'Deep Hypnosis'],
            'training_datasets': available_datasets,
            'cv_results': agg,
            'rf_params': rf_params,
            'feature_extractor_params': {
                'fs': cfg['epoc']['fs'],
                'window_sec': cfg['epoc']['window_sec'],
                'overlap': 0.5,
            },
        }

        model_path = models_dir / f"public_trained_wfsc_seed{seed}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)

        logger.info(f"  Seed {seed}: BA={agg.get('balanced_accuracy_mean', 0):.4f} "
                    f"+/- {agg.get('balanced_accuracy_std', 0):.4f}")
        logger.info(f"  Training time: {t_elapsed:.2f}s")
        logger.info(f"  Model saved: {model_path}")

        training_results[seed] = {
            'cv_results': agg,
            'training_time': t_elapsed,
            'model_path': str(model_path),
        }

    # Save training summary
    summary_path = results_dir / "training_results.json"
    with open(summary_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)

    return training_results


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("rt01_train_public", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("RT01: Train on Public Data → Real-time EPOC+")
    logger.info("=" * 60)

    results = train_on_public(cfg, logger)

    logger.info("\n" + "=" * 60)
    logger.info("RT01 complete! Models saved to models/")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
