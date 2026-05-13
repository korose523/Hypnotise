#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rt03_realtime_ablation.py — RT03: Real-time Ablation Study.

Ablation experiments for the real-time pipeline:
  1. Window size comparison (2s, 4s, 8s)
  2. Step size comparison (1s, 2s, 4s)
  3. Feature subset ablation (DASM only, Regional only, Full 63)
  4. Model comparison (RF, SVM, LR) on EPOC+ features
  5. Source data ablation (leave-one-dataset-out from public training)

Output:
  results/rt03_realtime_ablation/window_size_results.json
  results/rt03_realtime_ablation/step_size_results.json
  results/rt03_realtime_ablation/feature_ablation_results.json
  results/rt03_realtime_ablation/model_comparison_results.json
  results/rt03_realtime_ablation/source_ablation_results.json
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
from shared.feature_extraction import FeatureExtractor, EPOC_CHANNELS
from shared.metrics import compute_all_metrics


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


def generate_simulated_protocol(cfg, fs, label, n_seconds):
    """Generate simulated EEG for a protocol phase."""
    n_samples = int(n_seconds * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((14, n_samples))

    if label == 0:  # Awake
        eeg += 0.6 * np.sin(2 * np.pi * 10 * t[np.newaxis, :])
        eeg += 0.5 * np.sin(2 * np.pi * 22 * t[np.newaxis, :])
        eeg += 0.2 * np.sin(2 * np.pi * 6 * t[np.newaxis, :])
    elif label == 1:  # Light
        eeg += 0.8 * np.sin(2 * np.pi * 10 * t[np.newaxis, :])
        eeg += 0.3 * np.sin(2 * np.pi * 18 * t[np.newaxis, :])
        eeg += 0.4 * np.sin(2 * np.pi * 7 * t[np.newaxis, :])
    else:  # Deep
        eeg += 0.5 * np.sin(2 * np.pi * 9 * t[np.newaxis, :])
        eeg += 0.1 * np.sin(2 * np.pi * 15 * t[np.newaxis, :])
        eeg += 0.7 * np.sin(2 * np.pi * 5 * t[np.newaxis, :])

    eeg += np.random.normal(0, 0.1, eeg.shape)
    return eeg


def evaluate_window_sizes(cfg, logger):
    """Ablation 1: Window size comparison."""
    logger.info("\nAblation 1: Window Size Comparison")
    logger.info("-" * 50)

    results_dir = Path(cfg['output_dir']) / 'rt03_realtime_ablation'
    fs = cfg['epoc']['fs']
    step_sec = cfg['epoc']['step_sec']
    protocol = cfg['epoc']['protocol']['phases']

    window_sizes = [2.0, 4.0, 6.0, 8.0]
    results = {}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Train a quick model for this test
    processed_dir = Path(cfg['processed_dir'])
    X_list, y_list = [], []
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        logger.warning("No processed data — skipping window size ablation")
        return {}

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)
    clf = RandomForestClassifier(**cfg['model']['rf'], random_state=42)
    clf.fit(X_sc, y_all)

    for ws in window_sizes:
        extractor = FeatureExtractor(fs=fs)
        all_preds, all_true = [], []

        for phase in protocol:
            eeg = generate_simulated_protocol(cfg, fs, phase['label'], 30)  # 30s per phase
            window_size = int(ws * fs)
            step_size = int(step_sec * fs)

            for start in range(0, eeg.shape[1] - window_size + 1, step_size):
                window = eeg[:, start:start + window_size]
                features = extractor.extract_features(window).reshape(1, -1)
                pred = clf.predict(scaler.transform(features))[0]
                all_preds.append(pred)
                all_true.append(phase['label'])

        metrics = compute_all_metrics(np.array(all_true), np.array(all_preds))
        results[f'{ws}s'] = {
            'window_sec': ws,
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'n_windows': len(all_preds),
        }
        logger.info(f"  Window={ws}s: BA={metrics['balanced_accuracy']:.4f}, "
                    f"Acc={metrics['accuracy']:.4f}")

    # Save
    path = results_dir / "window_size_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_step_sizes(cfg, logger):
    """Ablation 2: Step size comparison."""
    logger.info("\nAblation 2: Step Size Comparison")
    logger.info("-" * 50)

    results_dir = Path(cfg['output_dir']) / 'rt03_realtime_ablation'
    fs = cfg['epoc']['fs']
    window_sec = cfg['epoc']['window_sec']
    protocol = cfg['epoc']['protocol']['phases']

    step_sizes = [1.0, 2.0, 3.0, 4.0]
    results = {}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    processed_dir = Path(cfg['processed_dir'])
    X_list, y_list = [], []
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return {}

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)
    clf = RandomForestClassifier(**cfg['model']['rf'], random_state=42)
    clf.fit(X_sc, y_all)

    for ss in step_sizes:
        extractor = FeatureExtractor(fs=fs)
        all_preds, all_true, all_latencies = [], [], []

        for phase in protocol:
            eeg = generate_simulated_protocol(cfg, fs, phase['label'], 30)
            window_size = int(window_sec * fs)
            step_size = int(ss * fs)

            for start in range(0, eeg.shape[1] - window_size + 1, step_size):
                t0 = time.perf_counter()
                window = eeg[:, start:start + window_size]
                features = extractor.extract_features(window).reshape(1, -1)
                pred = clf.predict(scaler.transform(features))[0]
                latency = (time.perf_counter() - t0) * 1000
                all_preds.append(pred)
                all_true.append(phase['label'])
                all_latencies.append(latency)

        metrics = compute_all_metrics(np.array(all_true), np.array(all_preds))
        results[f'{ss}s'] = {
            'step_sec': ss,
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'mean_latency_ms': float(np.mean(all_latencies)),
            'windows_per_minute': float(60.0 / ss),
            'n_windows': len(all_preds),
        }
        logger.info(f"  Step={ss}s: BA={metrics['balanced_accuracy']:.4f}, "
                    f"Latency={np.mean(all_latencies):.1f}ms, "
                    f"Rate={60.0/ss:.1f} windows/min")

    path = results_dir / "step_size_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_feature_subsets(cfg, logger):
    """Ablation 3: Feature subset comparison."""
    logger.info("\nAblation 3: Feature Subset Ablation")
    logger.info("-" * 50)

    results_dir = Path(cfg['output_dir']) / 'rt03_realtime_ablation'
    fs = cfg['epoc']['fs']
    protocol = cfg['epoc']['protocol']['phases']

    feature_subsets = {
        'full_63': (0, 63),
        'dasm_only': (0, 35),
        'regional_only': (35, 55),
        'global_only': (55, 60),
        'dasm_regional': (0, 55),
    }
    results = {}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    processed_dir = Path(cfg['processed_dir'])
    X_list, y_list = [], []
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return {}

    extractor = FeatureExtractor(fs=fs)

    for subset_name, (start, end) in feature_subsets.items():
        # Train model with feature subset
        X_all_sub = []
        y_all_sub = []
        for X, y in zip(X_list, y_list):
            X_sub = X[:, start:end]
            X_all_sub.append(X_sub)
            y_all_sub.append(y)

        X_sub = np.vstack(X_all_sub)
        y_sub = np.concatenate(y_all_sub)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_sub)
        clf = RandomForestClassifier(**cfg['model']['rf'], random_state=42)
        clf.fit(X_sc, y_sub)

        all_preds, all_true = [], []
        for phase in protocol:
            eeg = generate_simulated_protocol(cfg, fs, phase['label'], 30)
            window_size = int(cfg['epoc']['window_sec'] * fs)
            step_size = int(cfg['epoc']['step_sec'] * fs)

            for start_idx in range(0, eeg.shape[1] - window_size + 1, step_size):
                window = eeg[:, start_idx:start_idx + window_size]
                full_feat = extractor.extract_features(window)
                sub_feat = full_feat[start:end].reshape(1, -1)
                pred = clf.predict(scaler.transform(sub_feat))[0]
                all_preds.append(pred)
                all_true.append(phase['label'])

        metrics = compute_all_metrics(np.array(all_true), np.array(all_preds))
        results[subset_name] = {
            'feature_range': [start, end],
            'n_features': end - start,
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
        }
        logger.info(f"  {subset_name} [{start}:{end}] ({end-start}d): "
                    f"BA={metrics['balanced_accuracy']:.4f}")

    path = results_dir / "feature_ablation_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_model_comparison(cfg, logger):
    """Ablation 4: Model comparison on EPOC+ features."""
    logger.info("\nAblation 4: Model Comparison")
    logger.info("-" * 50)

    results_dir = Path(cfg['output_dir']) / 'rt03_realtime_ablation'
    fs = cfg['epoc']['fs']
    protocol = cfg['epoc']['protocol']['phases']

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    models = {
        'RF': lambda: RandomForestClassifier(**cfg['model']['rf'], random_state=42),
        'SVM_RBF': lambda: SVC(kernel='rbf', C=10, gamma='scale',
                                class_weight='balanced', probability=True, random_state=42),
        'LR': lambda: LogisticRegression(max_iter=1000, class_weight='balanced',
                                          multi_class='multinomial', random_state=42),
        'GBM': lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                   random_state=42),
    }

    processed_dir = Path(cfg['processed_dir'])
    X_list, y_list = [], []
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return {}

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    results = {}

    for model_name, model_fn in models.items():
        t_start = time.time()
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_all)
        clf = model_fn()
        clf.fit(X_sc, y_all)
        train_time = time.time() - t_start

        extractor = FeatureExtractor(fs=fs)
        all_preds, all_true, all_latencies = [], [], []

        for phase in protocol:
            eeg = generate_simulated_protocol(cfg, fs, phase['label'], 30)
            window_size = int(cfg['epoc']['window_sec'] * fs)
            step_size = int(cfg['epoc']['step_sec'] * fs)

            for start in range(0, eeg.shape[1] - window_size + 1, step_size):
                t0 = time.perf_counter()
                window = eeg[:, start:start + window_size]
                features = extractor.extract_features(window).reshape(1, -1)
                pred = clf.predict(scaler.transform(features))[0]
                latency = (time.perf_counter() - t0) * 1000
                all_preds.append(pred)
                all_true.append(phase['label'])
                all_latencies.append(latency)

        metrics = compute_all_metrics(np.array(all_true), np.array(all_preds))
        results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'train_time_sec': float(train_time),
            'mean_latency_ms': float(np.mean(all_latencies)),
        }
        logger.info(f"  {model_name}: BA={metrics['balanced_accuracy']:.4f}, "
                    f"Train={train_time:.2f}s, Latency={np.mean(all_latencies):.1f}ms")

    path = results_dir / "model_comparison_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_source_ablation(cfg, logger):
    """Ablation 5: Leave-one-dataset-out from public training."""
    logger.info("\nAblation 5: Source Data Ablation (LODO)")
    logger.info("-" * 50)

    results_dir = Path(cfg['output_dir']) / 'rt03_realtime_ablation'
    fs = cfg['epoc']['fs']
    protocol = cfg['epoc']['protocol']['phases']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    processed_dir = Path(cfg['processed_dir'])

    # Load all datasets
    dataset_data = {}
    for d in ALL_DATASETS:
        X, y = load_processed_dataset(processed_dir, d)
        if X is not None and y is not None:
            dataset_data[d] = (X, y)

    available = list(dataset_data.keys())
    if len(available) < 2:
        logger.warning("Not enough datasets for source ablation")
        return {}

    results = {}

    # All sources (baseline)
    X_all = np.vstack([dataset_data[d][0] for d in available])
    y_all = np.concatenate([dataset_data[d][1] for d in available])

    scaler_all = StandardScaler()
    X_sc_all = scaler_all.fit_transform(X_all)
    clf_all = RandomForestClassifier(**cfg['model']['rf'], random_state=42)
    clf_all.fit(X_sc_all, y_all)

    extractor = FeatureExtractor(fs=fs)

    def _evaluate_model(scaler, clf, label=""):
        all_preds, all_true = [], []
        for phase in protocol:
            eeg = generate_simulated_protocol(cfg, fs, phase['label'], 30)
            window_size = int(cfg['epoc']['window_sec'] * fs)
            step_size = int(cfg['epoc']['step_sec'] * fs)
            for start in range(0, eeg.shape[1] - window_size + 1, step_size):
                window = eeg[:, start:start + window_size]
                features = extractor.extract_features(window).reshape(1, -1)
                pred = clf.predict(scaler.transform(features))[0]
                all_preds.append(pred)
                all_true.append(phase['label'])
        m = compute_all_metrics(np.array(all_true), np.array(all_preds))
        logger.info(f"  {label}: BA={m['balanced_accuracy']:.4f}, "
                    f"Acc={m['accuracy']:.4f}")
        return m

    # All sources
    m_all = _evaluate_model(scaler_all, clf_all, "All sources")
    results['all_sources'] = {
        'balanced_accuracy': float(m_all['balanced_accuracy']),
        'n_training_samples': X_all.shape[0],
    }

    # Leave-one-out
    for excluded in available:
        remaining = [d for d in available if d != excluded]
        X_rem = np.vstack([dataset_data[d][0] for d in remaining])
        y_rem = np.concatenate([dataset_data[d][1] for d in remaining])

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_rem)
        clf = RandomForestClassifier(**cfg['model']['rf'], random_state=42)
        clf.fit(X_sc, y_rem)

        m = _evaluate_model(scaler, clf, f"Exclude {excluded}")
        results[f'exclude_{excluded}'] = {
            'balanced_accuracy': float(m['balanced_accuracy']),
            'n_training_samples': X_rem.shape[0],
            'performance_drop': float(m_all['balanced_accuracy'] - m['balanced_accuracy']),
        }

    path = results_dir / "source_ablation_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("rt03_realtime_ablation", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("RT03: Real-time Ablation Study")
    logger.info("=" * 60)

    evaluate_window_sizes(cfg, logger)
    evaluate_step_sizes(cfg, logger)
    evaluate_feature_subsets(cfg, logger)
    evaluate_model_comparison(cfg, logger)
    evaluate_source_ablation(cfg, logger)

    logger.info("\n" + "=" * 60)
    logger.info("RT03 complete! All ablation results saved.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
