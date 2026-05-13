#!/usr/bin/env python3
"""
rt201_epoc_protocol_segment_and_rating.py — EPOC+ protocol segmentation and subjective rating.

Real-time Experiment 201 (Paper 2, Protocol Design):
  - Defines the 4-phase hypnosis protocol for EPOC+ data collection
  - Segments continuous EEG into phase-specific windows
  - Integrates subjective rating system (0-10 depth scale)
  - Maps ratings to 3-class labels: Awake(0-3), Light(4-6), Deep(7-10)
  - Generates timeline markers for synchronization with analysis pipeline

Protocol phases:
  1. Awake baseline (3 min)     → label 0
  2. Light induction (5 min)    → label 1
  3. Deep induction (5 min)     → label 2
  4. Awakening (2 min)          → label 0

Input:  Raw EPOC+ EEG stream (.csv or real-time buffer)
Output: results/rt201_protocol/segmented_epochs.npz
        results/rt201_protocol/subjective_ratings.json
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.logger import setup_logger


class ProtocolPhase:
    """Represents a single phase of the hypnosis protocol."""

    def __init__(self, name, duration_sec, label, instruction, onset_sec=0):
        self.name = name
        self.duration_sec = duration_sec
        self.label = label
        self.instruction = instruction
        self.onset_sec = onset_sec
        self.offset_sec = onset_sec + duration_sec

    def contains_time(self, t_sec):
        """Check if a timestamp falls within this phase."""
        return self.onset_sec <= t_sec < self.offset_sec

    def get_progress(self, t_sec):
        """Get progress through this phase (0.0 to 1.0)."""
        if not self.contains_time(t_sec):
            return 0.0
        return (t_sec - self.onset_sec) / self.duration_sec

    def to_dict(self):
        return {
            'name': self.name,
            'duration_sec': self.duration_sec,
            'label': int(self.label),
            'instruction': self.instruction,
            'onset_sec': self.onset_sec,
            'offset_sec': self.offset_sec,
        }


class HypnosisProtocol:
    """
    Full hypnosis protocol for EPOC+ data collection.

    Manages phase transitions, subjective ratings, and timeline synchronization.
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config(str(PROJECT_ROOT / 'config.yaml'))

        self.config = config
        self.epoc_config = config['epoc']
        self.phases = []
        self.ratings = []
        self.total_duration = 0

        # Build phases from config
        onset = 0
        for phase_cfg in self.epoc_config['protocol']['phases']:
            phase = ProtocolPhase(
                name=phase_cfg['name'],
                duration_sec=phase_cfg['duration_sec'],
                label=phase_cfg['label'],
                instruction=phase_cfg['instruction'],
                onset_sec=onset,
            )
            self.phases.append(phase)
            onset += phase_cfg['duration_sec']

        self.total_duration = onset

        # Rating config
        self.rating_scale = self.epoc_config['protocol']['subjective_rating']['scale']
        self.rating_prompt = self.epoc_config['protocol']['subjective_rating']['prompt']
        self.rating_to_class = self.epoc_config['protocol']['rating_to_class']

    def get_phase_at(self, t_sec):
        """Get the protocol phase at a given time."""
        for phase in self.phases:
            if phase.contains_time(t_sec):
                return phase
        return None

    def get_label_at(self, t_sec):
        """Get the label (0/1/2) at a given time."""
        phase = self.get_phase_at(t_sec)
        return phase.label if phase else -1

    def segment_eeg(self, eeg_data, fs, window_sec=2.0, stride_sec=2.0):
        """
        Segment continuous EEG into phase-labeled windows.

        Args:
            eeg_data: ndarray (14, n_total_samples) — continuous EPOC+ EEG
            fs: float, sampling rate
            window_sec: float, window duration
            stride_sec: float, stride between windows

        Returns:
            windows: ndarray (n_windows, 14, window_samples)
            labels: ndarray (n_windows,) — phase labels
            timestamps: ndarray (n_windows,) — onset times in seconds
            phase_names: list of str
        """
        n_samples = eeg_data.shape[1]
        window_size = int(window_sec * fs)
        stride = int(stride_sec * fs)

        windows = []
        labels = []
        timestamps = []
        phase_names = []

        for start in range(0, n_samples - window_size + 1, stride):
            t_sec = start / fs
            window = eeg_data[:, start:start + window_size]
            phase = self.get_phase_at(t_sec)

            if phase is not None:
                windows.append(window)
                labels.append(phase.label)
                timestamps.append(t_sec)
                phase_names.append(phase.name)

        if not windows:
            return np.zeros((0, 14, window_size)), np.array([], dtype=int), \
                   np.array([]), []

        return np.array(windows), np.array(labels), np.array(timestamps), phase_names

    def record_rating(self, phase_name, rating_value, timestamp=None):
        """
        Record a subjective rating after a protocol phase.

        Args:
            phase_name: str, which phase was just completed
            rating_value: int, subjective depth rating (0-10)
            timestamp: float, optional time of rating
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        mapped_class = self._rating_to_class(rating_value)

        self.ratings.append({
            'phase': phase_name,
            'rating': int(rating_value),
            'mapped_class': int(mapped_class),
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
        })

        return mapped_class

    def _rating_to_class(self, rating):
        """Map subjective rating to 3-class label."""
        r2c = self.rating_to_class
        if r2c['awake'][0] <= rating <= r2c['awake'][1]:
            return 0
        elif r2c['light'][0] <= rating <= r2c['light'][1]:
            return 1
        elif r2c['deep'][0] <= rating <= r2c['deep'][1]:
            return 2
        return 0

    def get_phase_distribution(self, labels):
        """Count samples per phase."""
        dist = {}
        for phase in self.phases:
            dist[phase.name] = int(np.sum(labels == phase.label))
        return dist

    def to_dict(self):
        return {
            'total_duration_sec': self.total_duration,
            'n_phases': len(self.phases),
            'phases': [p.to_dict() for p in self.phases],
            'ratings': self.ratings,
            'rating_config': {
                'scale': self.rating_scale,
                'prompt': self.rating_prompt,
                'rating_to_class': self.rating_to_class,
            },
        }


def simulate_protocol_run(protocol, duration_factor=1.0, out_dir=None):
    """
    Simulate a protocol run for testing (generates synthetic EEG-like data).

    Args:
        protocol: HypnosisProtocol instance
        duration_factor: float, scale factor for duration
        out_dir: Path, output directory

    Returns:
        windows, labels, timestamps, phase_names
    """
    from shared.feature_extraction import EPOC_CHANNELS

    fs = protocol.epoc_config['fs']
    total_samples = int(protocol.total_duration * fs * duration_factor)

    # Generate synthetic EEG data (noise + slow drift)
    np.random.seed(42)
    eeg_data = np.random.randn(14, total_samples) * 10  # 10 uV noise

    # Add slow alpha oscillation (8-12 Hz) modulated by phase
    t = np.arange(total_samples) / fs
    for ch in range(14):
        alpha = 5 * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        eeg_data[ch] += alpha

    # Segment
    windows, labels, timestamps, phase_names = protocol.segment_eeg(
        eeg_data, fs, window_sec=2.0, stride_sec=2.0
    )

    # Simulate ratings
    rating_examples = [
        ('awake_baseline', 1),
        ('light_induction', 4),
        ('deep_induction', 7),
        ('awakening', 2),
    ]
    for phase_name, rating in rating_examples:
        protocol.record_rating(phase_name, rating)

    return windows, labels, timestamps, phase_names


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('rt201', str(PROJECT_ROOT / config['logs_dir']))

    protocol = HypnosisProtocol(config)

    logger.info("EPOC+ Hypnosis Protocol Configuration:")
    logger.info(f"  Total duration: {protocol.total_duration} sec "
                f"({protocol.total_duration / 60:.1f} min)")
    for phase in protocol.phases:
        logger.info(f"  Phase: {phase.name:20s} | "
                     f"{phase.duration_sec:3d}s | "
                     f"Label={phase.label} | "
                     f"[{phase.onset_sec}s - {phase.offset_sec}s]")
        logger.info(f"    Instruction: {phase.instruction}")

    logger.info(f"\n  Rating scale: {protocol.rating_scale}")
    logger.info(f"  Rating prompt: {protocol.rating_prompt}")

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'rt201_protocol')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a protocol run for testing
    logger.info("\nSimulating protocol run...")
    windows, labels, timestamps, phase_names = simulate_protocol_run(protocol, out_dir=out_dir)

    logger.info(f"  Generated {len(windows)} windows")
    dist = protocol.get_phase_distribution(labels)
    logger.info(f"  Phase distribution: {dist}")

    # Save segmented data
    np.savez_compressed(
        out_dir / 'segmented_epochs.npz',
        windows=windows,
        labels=labels,
        timestamps=timestamps,
        phase_names=np.array(phase_names),
    )

    # Save protocol config and ratings
    with open(out_dir / 'protocol_config.json', 'w', encoding='utf-8') as f:
        json.dump(protocol.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info("rt201 complete.")


if __name__ == '__main__':
    main()
