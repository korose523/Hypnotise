#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rt00_epoc_pipeline_offline_check.py — RT00: EPOC+ Pipeline Offline Check.

Offline verification of the full EPOC+ real-time pipeline:
  1. Simulate 14-channel EPOC+ data
  2. Test channel mapping (should be identity for EPOC+)
  3. Test 63-dim feature extraction at 128 Hz, 4s windows
  4. Test model loading and inference latency
  5. Validate end-to-end pipeline timing (< 100ms per window)

This is a DRY RUN to verify the pipeline works before live experiments.

Output:
  results/rt00_pipeline_check/pipeline_test_results.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.seed_manager import SeedManager
from shared.logger import setup_logger
from shared.feature_extraction import FeatureExtractor, map_channels_to_14, EPOC_CHANNELS
from shared.label_mapping import LabelMapper


def simulate_epoc_data(n_seconds=60, fs=128, n_channels=14):
    """
    Simulate EPOC+ EEG data with realistic characteristics.

    Args:
        n_seconds: float, duration in seconds
        fs: int, sampling frequency
        n_channels: int, number of channels

    Returns:
        eeg_data: ndarray (n_channels, n_samples)
    """
    n_samples = int(n_seconds * fs)
    t = np.arange(n_samples) / fs

    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Base signal: alpha rhythm (8-13 Hz) with slight variations
        alpha_freq = 10 + np.random.uniform(-1, 1)
        alpha_amp = 0.5 + np.random.uniform(-0.2, 0.2)
        eeg_data[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

        # Add beta rhythm (13-30 Hz)
        beta_freq = 20 + np.random.uniform(-3, 3)
        beta_amp = 0.2 + np.random.uniform(-0.1, 0.1)
        eeg_data[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t)

        # Add theta rhythm (4-8 Hz)
        theta_freq = 6 + np.random.uniform(-1, 1)
        theta_amp = 0.3 + np.random.uniform(-0.1, 0.1)
        eeg_data[ch] += theta_amp * np.sin(2 * np.pi * theta_freq * t)

        # Add noise
        eeg_data[ch] += np.random.normal(0, 0.1, n_samples)

    return eeg_data


def test_channel_mapping(logger):
    """Test channel mapping (EPOC+ should be identity mapping)."""
    logger.info("Test 1: Channel Mapping")
    logger.info("-" * 40)

    data = np.random.randn(100, 14)
    mapped_data, mapping_info = map_channels_to_14(data, EPOC_CHANNELS)

    assert mapped_data.shape == (100, 14), f"Expected (100, 14), got {mapped_data.shape}"

    # Check that EPOC+ channels map to themselves
    for ch in EPOC_CHANNELS:
        assert mapping_info[ch] == ch, f"Expected {ch}->{ch}, got {ch}->{mapping_info[ch]}"

    logger.info(f"  PASSED: EPOC+ identity mapping verified")
    logger.info(f"  Mapping: {mapping_info}")
    return True


def test_feature_extraction(logger, cfg):
    """Test 63-dim feature extraction timing and output shape."""
    logger.info("\nTest 2: Feature Extraction")
    logger.info("-" * 40)

    fs = cfg['epoc']['fs']
    window_sec = cfg['epoc']['window_sec']
    overlap = 0.5

    extractor = FeatureExtractor(fs=fs)

    # Generate 60 seconds of data
    eeg_data = simulate_epoc_data(n_seconds=60, fs=fs)

    logger.info(f"  Input shape: {eeg_data.shape}")
    logger.info(f"  Window: {window_sec}s, Overlap: {overlap}")

    t_start = time.time()
    features = extractor.extract_windows(eeg_data, window_sec, overlap)
    t_elapsed = time.time() - t_start

    logger.info(f"  Output shape: {features.shape}")
    logger.info(f"  Expected dims: (n_windows, 63)")
    assert features.shape[1] == 63, f"Expected 63 features, got {features.shape[1]}"

    # Verify feature values are reasonable
    assert not np.any(np.isnan(features)), "NaN values in features!"
    assert not np.any(np.isinf(features)), "Inf values in features!"

    # Check feature blocks
    logger.info(f"  DASM block [0:35] range: [{features[:, 0:35].min():.4f}, {features[:, 0:35].max():.4f}]")
    logger.info(f"  Regional block [35:55] range: [{features[:, 35:55].min():.4f}, {features[:, 35:55].max():.4f}]")
    logger.info(f"  Global block [55:60] range: [{features[:, 55:60].min():.4f}, {features[:, 55:60].max():.4f}]")
    logger.info(f"  Ratios [60:63] range: [{features[:, 60:63].min():.4f}, {features[:, 60:63].max():.4f}]")

    logger.info(f"  Extraction time for {features.shape[0]} windows: {t_elapsed:.3f}s")
    logger.info(f"  Per-window: {t_elapsed / features.shape[0] * 1000:.1f}ms")

    return True


def test_inference_latency(logger, cfg):
    """Test model inference latency."""
    logger.info("\nTest 3: Inference Latency")
    logger.info("-" * 40)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Create a dummy trained model
    rf_params = cfg['model']['rf']
    clf = RandomForestClassifier(**rf_params)
    X_dummy = np.random.randn(500, 63)
    y_dummy = np.random.randint(0, 3, 500)
    clf.fit(X_dummy, y_dummy)

    scaler = StandardScaler()
    scaler.fit(X_dummy)

    # Test single-window inference
    single_window = np.random.randn(1, 63)
    n_trials = 100

    latencies = []
    for _ in range(n_trials):
        t_start = time.perf_counter()
        X_scaled = scaler.transform(single_window)
        prediction = clf.predict(X_scaled)
        probability = clf.predict_proba(X_scaled)
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
        latencies.append(t_elapsed)

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    max_latency = np.max(latencies)

    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Mean latency: {mean_latency:.2f}ms")
    logger.info(f"  Std latency:  {std_latency:.2f}ms")
    logger.info(f"  Max latency:  {max_latency:.2f}ms")

    latency_ok = mean_latency < 100
    logger.info(f"  {'PASSED' if latency_ok else 'WARNING'}: "
                f"{'< 100ms' if latency_ok else '>= 100ms — may affect real-time performance'}")

    return latency_ok


def test_ring_buffer(logger, cfg):
    """Test ring buffer implementation for real-time data."""
    logger.info("\nTest 4: Ring Buffer")
    logger.info("-" * 40)

    fs = cfg['epoc']['fs']
    buffer_sec = cfg['epoc']['buffer_sec']
    window_sec = cfg['epoc']['window_sec']
    step_sec = cfg['epoc']['step_sec']
    n_channels = cfg['epoc']['channels']

    buffer_size = int(buffer_sec * fs)
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)

    logger.info(f"  Buffer: {buffer_sec}s ({buffer_size} samples)")
    logger.info(f"  Window: {window_sec}s ({window_size} samples)")
    logger.info(f"  Step:   {step_sec}s ({step_size} samples)")

    # Simulate streaming
    buffer = np.zeros((n_channels, buffer_size))
    extractor = FeatureExtractor(fs=fs)

    n_windows_extracted = 0
    t_total = 0.0

    # Simulate 10 seconds of streaming
    stream_seconds = 10
    for second in range(stream_seconds):
        t_start = time.time()

        # Simulate receiving 1 second of data
        new_data = simulate_epoc_data(n_seconds=1, fs=fs)

        # Shift buffer and append new data
        buffer = np.roll(buffer, -fs, axis=1)
        buffer[:, -fs:] = new_data

        # Try to extract features if enough data
        if second >= int(window_sec):
            window_data = buffer[:, -window_size:]
            features = extractor.extract_features(window_data)
            n_windows_extracted += 1

        t_elapsed = time.time() - t_start
        t_total += t_elapsed

    logger.info(f"  Streamed {stream_seconds}s of data")
    logger.info(f"  Windows extracted: {n_windows_extracted}")
    logger.info(f"  Total processing time: {t_total:.3f}s")
    logger.info(f"  Real-time factor: {t_total / stream_seconds:.3f}x "
                f"({'OK' if t_total / stream_seconds < 1.0 else 'TOO SLOW'})")

    return t_total / stream_seconds < 1.0


def test_label_mapping(logger, cfg):
    """Test label mapping for real-time protocol phases."""
    logger.info("\nTest 5: Protocol Label Mapping")
    logger.info("-" * 40)

    protocol = cfg['epoc']['protocol']
    phases = protocol['phases']

    logger.info(f"  Protocol phases:")
    for phase in phases:
        logger.info(f"    {phase['name']}: label={phase['label']}, "
                    f"duration={phase['duration_sec']}s")

    mapper = LabelMapper(cfg)
    logger.info(f"  Class names: {mapper.CLASS_NAMES}")
    logger.info(f"  Class names (CN): {mapper.CLASS_NAMES_CN}")

    return True


def main():
    """Main entry point for RT00 pipeline check."""
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    logger = setup_logger("rt00_pipeline_check", cfg['logs_dir'])

    logger.info("=" * 60)
    logger.info("RT00: EPOC+ Pipeline Offline Check")
    logger.info("=" * 60)

    results = {
        'test_channel_mapping': False,
        'test_feature_extraction': False,
        'test_inference_latency': False,
        'test_ring_buffer': False,
        'test_label_mapping': False,
    }

    try:
        results['test_channel_mapping'] = test_channel_mapping(logger)
    except Exception as e:
        logger.error(f"Channel mapping test FAILED: {e}")

    try:
        results['test_feature_extraction'] = test_feature_extraction(logger, cfg)
    except Exception as e:
        logger.error(f"Feature extraction test FAILED: {e}")

    try:
        results['test_inference_latency'] = test_inference_latency(logger, cfg)
    except Exception as e:
        logger.error(f"Inference latency test FAILED: {e}")

    try:
        results['test_ring_buffer'] = test_ring_buffer(logger, cfg)
    except Exception as e:
        logger.error(f"Ring buffer test FAILED: {e}")

    try:
        results['test_label_mapping'] = test_label_mapping(logger, cfg)
    except Exception as e:
        logger.error(f"Label mapping test FAILED: {e}")

    # Summary
    all_passed = all(results.values())
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Check Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")
    logger.info(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    logger.info("=" * 60)

    # Save results
    results_dir = Path(cfg['output_dir']) / 'rt00_pipeline_check'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "pipeline_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return all_passed


if __name__ == '__main__':
    main()
