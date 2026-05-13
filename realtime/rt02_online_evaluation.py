#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rt02_online_evaluation.py — RT02: Online Evaluation with Simulated EPOC+ Data.

Simulates real-time classification pipeline:
  1. Load pre-trained WFSC model from rt01
  2. Stream simulated EPOC+ data (14ch, 128Hz)
  3. Apply sliding window feature extraction
  4. Real-time classification with latency tracking
  5. Track classification accuracy over time

Output:
  results/rt02_online_eval/online_eval_results.json
"""

import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.feature_extraction import FeatureExtractor, EPOC_CHANNELS


def generate_realistic_eeg(n_seconds, fs, label, ch_noise_level=0.1):
    """
    Generate realistic EEG data with label-dependent characteristics.

    Args:
        n_seconds: float
        fs: int, sampling frequency
        label: int, 0=Awake, 1=Light, 2=Deep
        ch_noise_level: float, noise amplitude

    Returns:
        eeg_data: ndarray (14, n_samples)
    """
    n_samples = int(n_seconds * fs)
    t = np.arange(n_samples) / fs
    eeg_data = np.zeros((14, n_samples))

    # Label-dependent EEG characteristics
    if label == 0:  # Awake: high beta, low theta
        alpha_freq, alpha_amp = 10, 0.6
        beta_freq, beta_amp = 22, 0.5
        theta_freq, theta_amp = 6, 0.2
    elif label == 1:  # Light Hypnosis: increased alpha, decreased beta
        alpha_freq, alpha_amp = 10, 0.8
        beta_freq, beta_amp = 18, 0.3
        theta_freq, theta_amp = 7, 0.4
    else:  # Deep Hypnosis: high theta, very low beta
        alpha_freq, alpha_amp = 9, 0.5
        beta_freq, beta_amp = 15, 0.1
        theta_freq, theta_amp = 5, 0.7

    for ch in range(14):
        # Add frequency components with channel-specific variations
        freq_var = np.random.uniform(-0.5, 0.5)
        amp_var = np.random.uniform(0.8, 1.2)

        eeg_data[ch] += (alpha_amp * amp_var *
                         np.sin(2 * np.pi * (alpha_freq + freq_var) * t))
        eeg_data[ch] += (beta_amp * amp_var *
                         np.sin(2 * np.pi * (beta_freq + freq_var * 1.5) * t))
        eeg_data[ch] += (theta_amp * amp_var *
                         np.sin(2 * np.pi * (theta_freq + freq_var * 0.5) * t))

        # Add 1/f pink noise component
        pink_noise = np.random.normal(0, ch_noise_level, n_samples)
        # Simple pink noise approximation via filtering
        for _ in range(3):
            pink_noise = np.convolve(pink_noise, [0.25, 0.5, 0.25], mode='same')
        eeg_data[ch] += pink_noise

        # Add 50Hz line noise (common in real EEG)
        eeg_data[ch] += 0.05 * np.sin(2 * np.pi * 50 * t)

    return eeg_data


def run_online_evaluation(cfg, logger):
    """Run simulated online evaluation."""
    results_dir = Path(cfg['output_dir']) / 'rt02_online_eval'
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg['models_dir'])

    # Find latest trained model
    model_files = sorted(models_dir.glob("public_trained_wfsc_seed*.pkl"))
    if not model_files:
        logger.error("No pre-trained model found. Run rt01 first.")
        return None

    model_path = model_files[-1]
    logger.info(f"Loading model: {model_path}")

    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)

    clf = model_pkg['model']
    scaler = model_pkg['scaler']
    extractor = FeatureExtractor(fs=cfg['epoc']['fs'])

    # EPOC+ parameters
    fs = cfg['epoc']['fs']
    window_sec = cfg['epoc']['window_sec']
    step_sec = cfg['epoc']['step_sec']
    n_channels = cfg['epoc']['channels']

    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)

    # Protocol phases
    protocol = cfg['epoc']['protocol']['phases']

    logger.info(f"\nOnline Evaluation Parameters:")
    logger.info(f"  Sampling rate: {fs} Hz")
    logger.info(f"  Window: {window_sec}s ({window_size} samples)")
    logger.info(f"  Step: {step_sec}s ({step_size} samples)")
    logger.info(f"  Channels: {n_channels}")

    # Simulate protocol execution
    all_predictions = []
    all_true_labels = []
    all_latencies = []
    all_confidences = []

    phase_results = {}

    for phase in protocol:
        phase_name = phase['name']
        true_label = phase['label']
        duration = phase['duration_sec']

        logger.info(f"\n  Phase: {phase_name} (label={true_label}, duration={duration}s)")
        logger.info(f"  Instruction: {phase['instruction']}")

        # Generate simulated EEG for this phase
        eeg_data = generate_realistic_eeg(duration, fs, true_label)

        # Process with sliding window
        phase_preds = []
        phase_true = []
        phase_latencies = []
        phase_confidences = []

        n_windows = 0
        for start in range(0, eeg_data.shape[1] - window_size + 1, step_size):
            window = eeg_data[:, start:start + window_size]

            # Feature extraction
            t_feat_start = time.perf_counter()
            features = extractor.extract_features(window)
            features_2d = features.reshape(1, -1)
            t_feat_end = time.perf_counter()

            # Inference
            X_scaled = scaler.transform(features_2d)
            t_inf_start = time.perf_counter()
            prediction = clf.predict(X_scaled)[0]
            probability = clf.predict_proba(X_scaled)[0]
            t_inf_end = time.perf_counter()

            confidence = np.max(probability)
            latency = (t_inf_end - t_feat_start) * 1000  # Total latency in ms

            phase_preds.append(int(prediction))
            phase_true.append(true_label)
            phase_latencies.append(latency)
            phase_confidences.append(confidence)
            n_windows += 1

        phase_acc = np.mean([p == t for p, t in zip(phase_preds, phase_true)])
        phase_mean_conf = np.mean(phase_confidences)
        phase_mean_latency = np.mean(phase_latencies)

        phase_results[phase_name] = {
            'true_label': int(true_label),
            'duration_sec': duration,
            'n_windows': n_windows,
            'accuracy': float(phase_acc),
            'mean_confidence': float(phase_mean_conf),
            'mean_latency_ms': float(phase_mean_latency),
            'predictions': phase_preds,
            'true_labels': phase_true,
        }

        all_predictions.extend(phase_preds)
        all_true_labels.extend(phase_true)
        all_latencies.extend(phase_latencies)
        all_confidences.extend(phase_confidences)

        logger.info(f"    Windows: {n_windows}")
        logger.info(f"    Accuracy: {phase_acc:.4f}")
        logger.info(f"    Mean confidence: {phase_mean_conf:.4f}")
        logger.info(f"    Mean latency: {phase_mean_latency:.2f}ms")

    # Overall results
    overall_acc = np.mean([p == t for p, t in zip(all_predictions, all_true_labels)])
    overall_conf = np.mean(all_confidences)
    overall_latency = np.mean(all_latencies)

    from shared.metrics import compute_all_metrics
    metrics = compute_all_metrics(
        np.array(all_true_labels), np.array(all_predictions)
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Overall Results:")
    logger.info(f"  Total windows: {len(all_predictions)}")
    logger.info(f"  Accuracy: {overall_acc:.4f}")
    logger.info(f"  Mean confidence: {overall_conf:.4f}")
    logger.info(f"  Mean latency: {overall_latency:.2f}ms")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro-F1: {metrics['macro_f1']:.4f}")

    # Save results
    results = {
        'model_path': str(model_path),
        'evaluation_time': datetime.now().isoformat(),
        'overall': {
            'accuracy': float(overall_acc),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'cohens_kappa': float(metrics['cohens_kappa']),
            'mean_confidence': float(overall_conf),
            'mean_latency_ms': float(overall_latency),
            'max_latency_ms': float(np.max(all_latencies)),
            'min_latency_ms': float(np.min(all_latencies)),
            'n_windows': len(all_predictions),
        },
        'per_phase': phase_results,
        'confusion_matrix': metrics['confusion_matrix'],
    }

    results_path = results_dir / "online_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    return results


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("rt02_online_eval", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("RT02: Online Evaluation (Simulated EPOC+)")
    logger.info("=" * 60)

    results = run_online_evaluation(cfg, logger)

    logger.info("\n" + "=" * 60)
    logger.info("RT02 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
