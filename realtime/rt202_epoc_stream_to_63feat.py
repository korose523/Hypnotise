#!/usr/bin/env python3
"""
rt202_epoc_stream_to_63feat.py — EPOC+ real-time stream to 63-dim feature conversion.

Real-time Experiment 202 (Paper 2, Feature Pipeline):
  - Converts incoming EPOC+ EEG stream chunks into 63-dimensional features
  - Implements a circular buffer for real-time window accumulation
  - Produces feature vectors at the same rate as offline pipeline (every 2s)
  - Compatible with both simulation mode and live EPOC+ stream

Input:  EPOC+ EEG stream (14 channels x 128 Hz)
Output: 63-dim feature vector every 2 seconds
        results/rt202_stream/stream_features.npz (simulation mode)
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import deque
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.feature_extraction import FeatureExtractor, EPOC_CHANNELS, FEATURE_ORDER
from shared.logger import setup_logger


class CircularEEGBuffer:
    """
    Circular buffer for accumulating EEG samples in real-time.

    Maintains a rolling window of the most recent `buffer_sec` seconds of data
    and outputs 2-second windows for feature extraction.
    """

    def __init__(self, n_channels=14, fs=128, window_sec=2.0, buffer_sec=60.0):
        self.n_channels = n_channels
        self.fs = fs
        self.window_sec = window_sec
        self.buffer_sec = buffer_sec

        self.window_samples = int(window_sec * fs)
        self.buffer_samples = int(buffer_sec * fs)

        # Ring buffer
        self.buffer = np.zeros((n_channels, self.buffer_samples))
        self.write_idx = 0
        self.total_written = 0

    def add_samples(self, chunk):
        """
        Add new EEG samples to the buffer.

        Args:
            chunk: ndarray (n_channels, n_new_samples) or (n_new_samples, n_channels)
        """
        if chunk.ndim == 2 and chunk.shape[0] == self.n_channels:
            pass  # Already (n_channels, n_samples)
        elif chunk.ndim == 2:
            chunk = chunk.T
        elif chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        n_new = chunk.shape[1]
        for i in range(n_new):
            self.buffer[:, self.write_idx % self.buffer_samples] = chunk[:, i]
            self.write_idx += 1
        self.total_written += n_new

    def get_latest_window(self):
        """
        Get the most recent complete window.

        Returns:
            window: ndarray (n_channels, window_samples) or None if insufficient data
        """
        if self.total_written < self.window_samples:
            return None

        # Extract the last window_samples from the ring buffer
        indices = [(self.write_idx - self.window_samples + i) % self.buffer_samples
                   for i in range(self.window_samples)]
        window = self.buffer[:, indices]
        return window

    def get_n_complete_windows(self):
        """How many complete windows can be extracted."""
        if self.total_written < self.window_samples:
            return 0
        return (self.total_written - self.window_samples) // self.window_samples + 1

    def is_window_ready(self):
        """Check if at least one complete window is available."""
        return self.total_written >= self.window_samples

    def reset(self):
        """Reset the buffer."""
        self.buffer.fill(0)
        self.write_idx = 0
        self.total_written = 0


class RealtimeFeatureExtractor:
    """
    Real-time 63-dimensional feature extraction from EPOC+ stream.

    Processes incoming EEG chunks and outputs feature vectors every 2 seconds.
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config(str(PROJECT_ROOT / 'config.yaml'))

        self.config = config
        self.fs = config['features']['fs_target']
        self.window_sec = config['features']['window_sec']
        self.stride_sec = config['features']['stride_sec']
        self.nperseg = config['features']['nperseg']
        self.n_channels = 14
        self.n_features = 63

        self.extractor = FeatureExtractor(fs=self.fs, nperseg=self.nperseg)
        self.buffer = CircularEEGBuffer(
            n_channels=self.n_channels,
            fs=self.fs,
            window_sec=self.window_sec,
            buffer_sec=config['epoc']['buffer_sec'],
        )

        self.last_window_idx = 0
        self.feature_count = 0
        self.latency_log = []

    def process_chunk(self, eeg_chunk, timestamp=None):
        """
        Process an incoming EEG chunk and extract features if a new window is ready.

        Args:
            eeg_chunk: ndarray (n_channels, n_samples) or (n_samples, n_channels)
            timestamp: float, optional timestamp

        Returns:
            feature_vector: ndarray (63,) or None if no new window
            info: dict with metadata
        """
        t0 = time.time()
        self.buffer.add_samples(eeg_chunk)

        # Check if a new window is available
        current_windows = self.buffer.get_n_complete_windows()
        if current_windows <= self.last_window_idx:
            return None, {}

        self.last_window_idx = current_windows
        window = self.buffer.get_latest_window()

        if window is None:
            return None, {}

        # Extract features
        features = self.extractor.extract_features(window)
        self.feature_count += 1

        latency = time.time() - t0
        self.latency_log.append(latency)

        info = {
            'feature_count': self.feature_count,
            'window_index': current_windows,
            'latency_ms': latency * 1000,
            'timestamp': timestamp,
        }

        return features, info

    def get_latency_stats(self):
        """Get feature extraction latency statistics."""
        if not self.latency_log:
            return {'mean_ms': 0, 'std_ms': 0, 'max_ms': 0, 'n': 0}

        lats = np.array(self.latency_log) * 1000  # Convert to ms
        return {
            'mean_ms': float(np.mean(lats)),
            'std_ms': float(np.std(lats)),
            'max_ms': float(np.max(lats)),
            'min_ms': float(np.min(lats)),
            'p95_ms': float(np.percentile(lats, 95)),
            'p99_ms': float(np.percentile(lats, 99)),
            'n': len(lats),
        }

    def reset(self):
        """Reset the extractor state."""
        self.buffer.reset()
        self.last_window_idx = 0
        self.feature_count = 0
        self.latency_log = []


def simulate_stream(rfe, duration_sec=60, chunk_sec=0.1):
    """
    Simulate an EPOC+ stream for testing.

    Args:
        rfe: RealtimeFeatureExtractor instance
        duration_sec: float, simulation duration
        chunk_sec: float, chunk size in seconds

    Returns:
        all_features: list of ndarray (63,)
        all_info: list of dict
    """
    fs = rfe.fs
    chunk_samples = int(chunk_sec * fs)

    all_features = []
    all_info = []

    np.random.seed(42)
    n_total = int(duration_sec * fs)

    for start in range(0, n_total, chunk_samples):
        # Generate synthetic EEG chunk
        chunk = np.random.randn(14, chunk_samples) * 10
        t = np.arange(start, start + chunk_samples) / fs
        for ch in range(14):
            chunk[ch] += 5 * np.sin(2 * np.pi * 10 * t + ch * 0.3)

        features, info = rfe.process_chunk(chunk, timestamp=start / fs)

        if features is not None:
            all_features.append(features)
            all_info.append(info)

    return all_features, all_info


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('rt202', str(PROJECT_ROOT / config['logs_dir']))

    rfe = RealtimeFeatureExtractor(config)

    logger.info("Real-time Feature Extractor Configuration:")
    logger.info(f"  Channels: {rfe.n_channels}")
    logger.info(f"  Fs: {rfe.fs} Hz")
    logger.info(f"  Window: {rfe.window_sec}s ({int(rfe.window_sec * rfe.fs)} samples)")
    logger.info(f"  Features: {rfe.n_features} dims")
    logger.info(f"  Buffer: {rfe.buffer.buffer_sec}s")

    # Simulate a 60-second stream
    logger.info("\nSimulating 60s EPOC+ stream...")
    features, infos = simulate_stream(rfe, duration_sec=60, chunk_sec=0.1)

    logger.info(f"  Extracted {len(features)} feature vectors")
    logger.info(f"  Expected: {int(60 / rfe.window_sec)} (60s / {rfe.window_sec}s)")

    # Latency stats
    latency = rfe.get_latency_stats()
    logger.info(f"\n  Latency statistics:")
    logger.info(f"    Mean: {latency['mean_ms']:.2f} ms")
    logger.info(f"    Std:  {latency['std_ms']:.2f} ms")
    logger.info(f"    P95:  {latency['p95_ms']:.2f} ms")
    logger.info(f"    P99:  {latency['p99_ms']:.2f} ms")
    logger.info(f"    Max:  {latency['max_ms']:.2f} ms")

    # Save results
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'rt202_stream')
    out_dir.mkdir(parents=True, exist_ok=True)

    if features:
        np.savez_compressed(
            out_dir / 'stream_features.npz',
            features=np.array(features),
            feature_order=np.array(FEATURE_ORDER),
        )

    with open(out_dir / 'latency_stats.json', 'w') as f:
        json.dump(latency, f, indent=2)

    with open(out_dir / 'stream_info.json', 'w') as f:
        json.dump(infos, f, indent=2)

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info("rt202 complete.")


if __name__ == '__main__':
    main()
