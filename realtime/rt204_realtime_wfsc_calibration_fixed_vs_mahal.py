#!/usr/bin/env python3
"""
rt204_realtime_wfsc_calibration_fixed_vs_mahal.py — Real-time WFSC calibration comparison.

Real-time Experiment 204 (Paper 2, Calibration Study):
  - Compares WFSC-Mahalanobis vs WFSC-Fixed during real-time operation
  - Studies how calibration data quality affects real-time performance
  - Implements online calibration update (incremental weight adjustment)
  - Measures adaptation speed: how quickly weights converge after calibration

Input:  Pretrained models + simulated calibration stream
Output: results/rt204_wfsc_calib/calibration_comparison.json
        results/rt204_wfsc_calib/weight_convergence.npz
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
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


def simulate_realtime_calibration(source_data, target_calib, target_test,
                                   n_incremental_steps=10):
    """
    Simulate incremental calibration during real-time operation.

    Args:
        source_data: dict {source_name: (X, y)}
        target_calib: ndarray (n_calib, 63)
        target_test: ndarray (n_test, 63)
        y_test: ndarray (n_test,)
        n_incremental_steps: int, number of incremental calibration updates

    Returns:
        step_results: list of dict, results at each calibration step
    """
    step_results = []
    calib_step_size = max(1, len(target_calib) // n_incremental_steps)

    for step in range(n_incremental_steps + 1):
        # Use increasing amounts of calibration data
        n_calib = min(step * calib_step_size, len(target_calib))
        if step == n_incremental_steps:
            n_calib = len(target_calib)

        X_calib = target_calib[:n_calib] if n_calib > 0 else None

        # WFSC-Mahalanobis
        wfsc_m = WFSC_Mahalanobis(random_state=42)
        wfsc_m.fit(source_data, X_calib)
        y_pred_m = wfsc_m.predict(target_test)
        metrics_m = compute_all_metrics(y_test, y_pred_m)

        # WFSC-Fixed
        wfsc_f = WFSC_Fixed(random_state=42)
        wfsc_f.fit(source_data, X_calib)
        y_pred_f = wfsc_f.predict(target_test)
        metrics_f = compute_all_metrics(y_test, y_pred_f)

        step_results.append({
            'step': step,
            'n_calib_samples': n_calib,
            'wfsc_mahalanobis': {
                'balanced_accuracy': metrics_m['balanced_accuracy'],
                'accuracy': metrics_m['accuracy'],
                'macro_f1': metrics_m['macro_f1'],
                'weights': wfsc_m.get_weights(),
            },
            'wfsc_fixed': {
                'balanced_accuracy': metrics_f['balanced_accuracy'],
                'accuracy': metrics_f['accuracy'],
                'macro_f1': metrics_f['macro_f1'],
            },
        })

    return step_results


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('rt204', str(PROJECT_ROOT / config['logs_dir']))

    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'rt204_wfsc_calib')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate using DEAP as target, all others as sources
    target_name = 'DEAP'
    source_names = ['DREAMER', 'MAHNOB', 'SEED', 'SEED_IV', 'FACED']

    logger.info(f"Real-time Calibration Study:")
    logger.info(f"  Target: {target_name}")
    logger.info(f"  Sources: {source_names}")

    # Load target data
    X_target, y_target, target_subj_ids = load_prep_data(target_name, prep_dir)
    if X_target is None or len(X_target) < 20:
        logger.error(f"Target data not available. Aborting.")
        return

    # Split target into calibration and test
    np.random.seed(42)
    unique_subjs = list(set(str(s) for s in target_subj_ids))
    np.random.shuffle(unique_subjs)
    n_calib_subjs = max(1, int(len(unique_subjs) * 0.2))
    calib_subjs = set(unique_subjs[:n_calib_subjs])

    calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
    X_calib = X_target[calib_mask]
    y_calib = y_target[calib_mask]
    X_test = X_target[~calib_mask]
    y_test = y_target[~calib_mask]

    logger.info(f"  Calibration: {len(X_calib)} samples ({len(calib_subjs)} subjects)")
    logger.info(f"  Test: {len(X_test)} samples ({len(unique_subjs) - len(calib_subjs)} subjects)")

    # Load source data
    source_data = {}
    for src in source_names:
        X_s, y_s, _ = load_prep_data(src, prep_dir)
        if X_s is not None and len(X_s) > 0:
            source_data[src] = (X_s, y_s)
            logger.info(f"  Source {src}: {X_s.shape[0]} samples")

    if len(source_data) == 0:
        logger.error("No source data available. Aborting.")
        return

    # Simulate incremental calibration
    logger.info("\nSimulating incremental calibration (10 steps)...")
    step_results = simulate_realtime_calibration(
        source_data, X_calib, X_test, n_incremental_steps=10
    )

    # Print convergence trace
    logger.info(f"\n{'Step':>5} {'Calib N':>8} {'Mahal BA':>10} {'Fixed BA':>10} {'Diff':>8}")
    logger.info("-" * 50)
    for sr in step_results:
        m_ba = sr['wfsc_mahalanobis']['balanced_accuracy']
        f_ba = sr['wfsc_fixed']['balanced_accuracy']
        diff = m_ba - f_ba
        logger.info(f"{sr['step']:>5d} {sr['n_calib_samples']:>8d} "
                     f"{m_ba:>10.4f} {f_ba:>10.4f} {diff:>+8.4f}")

    # Analyze weight convergence
    weight_history = {}
    for sr in step_results:
        if sr['n_calib_samples'] > 0:
            for src, w in sr['wfsc_mahalanobis']['weights'].items():
                if src not in weight_history:
                    weight_history[src] = []
                weight_history[src].append(w)

    logger.info("\nWeight Convergence:")
    for src, weights in weight_history.items():
        logger.info(f"  {src}: {weights[0]:.4f} -> {weights[-1]:.4f} "
                     f"(delta={weights[-1] - weights[0]:+.4f})")

    # Save results
    with open(out_dir / 'calibration_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({
            'target': target_name,
            'sources': source_names,
            'n_calib': len(X_calib),
            'n_test': len(X_test),
            'steps': step_results,
            'weight_history': weight_history,
        }, f, indent=2, ensure_ascii=False, default=str)

    # Save weight convergence as numpy
    if weight_history:
        sources = sorted(weight_history.keys())
        weights_matrix = np.array([weight_history[s] for s in sources])
        np.savez_compressed(
            out_dir / 'weight_convergence.npz',
            weights=weights_matrix,
            source_names=np.array(sources),
        )

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info("rt204 complete.")


if __name__ == '__main__':
    main()
