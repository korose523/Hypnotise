"""
feature_extraction.py — 14-channel mapping + 63-dimensional feature extraction.

Feature vector (63 dims):
  [0:35]  = 5 bands x 7 DASM (differential asymmetry) pairs
  [35:55] = 5 bands x 4 regional mean powers
  [55:60] = 5 bands x 1 global mean power
  [60]    = theta/beta ratio (global)
  [61]    = alpha/theta ratio (global)
  [62]    = alpha peak frequency (global)
"""

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# 10-20 standard electrode coordinates (x, y) for nearest-neighbor mapping
# ---------------------------------------------------------------------------
STANDARD_1020_COORDS = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'AF3': (-0.41, 0.81), 'AF4': (0.41, 0.81),
    'F7': (-0.81, 0.59),  'F8': (0.81, 0.59),
    'F3': (-0.55, 0.59),  'F4': (0.55, 0.59),
    'Fz': (0.0, 0.59),
    'FC5': (-0.71, 0.41), 'FC6': (0.71, 0.41),
    'FC1': (-0.24, 0.41), 'FC2': (0.24, 0.41),
    'T7': (-1.0, 0.0),    'T8': (1.0, 0.0),
    'C3': (-0.55, 0.0),   'C4': (0.55, 0.0),
    'Cz': (0.0, 0.0),
    'CP5': (-0.71, -0.41),'CP6': (0.71, -0.41),
    'CP1': (-0.24, -0.41),'CP2': (0.24, -0.41),
    'P7': (-0.81, -0.59), 'P8': (0.81, -0.59),
    'P3': (-0.55, -0.59), 'P4': (0.55, -0.59),
    'Pz': (0.0, -0.59),
    'PO3': (-0.41, -0.81),'PO4': (0.41, -0.81),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
    'Oz': (0.0, -0.95),
}

EPOC_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]


def map_channels_to_14(data, source_channels, target_channels=None):
    """
    Map any EEG montage to the 14 EPOC+ channels via nearest-neighbor
    on 10-20 standard coordinates.

    Args:
        data: ndarray (n_samples, n_channels) or (n_channels, n_samples)
        source_channels: list of str, channel names in source data
        target_channels: list of str, default EPOC+ 14 channels

    Returns:
        mapped_data: ndarray (n_samples, 14)
        mapping_info: dict {target_ch: source_ch}
    """
    if target_channels is None:
        target_channels = EPOC_CHANNELS

    # Ensure data is (n_samples, n_channels)
    if data.ndim == 2 and data.shape[0] == len(source_channels):
        data = data.T

    source_ch_lower = {ch.lower(): i for i, ch in enumerate(source_channels)}
    mapping_info = {}
    mapped_indices = []

    # Build case-insensitive lookup for 10-20 coords
    coord_lookup = {k.lower(): k for k in STANDARD_1020_COORDS}

    for tch in target_channels:
        # Exact match first
        if tch.lower() in source_ch_lower:
            mapped_indices.append(source_ch_lower[tch.lower()])
            mapping_info[tch] = tch
        else:
            # Nearest neighbor in 10-20 coordinates
            if tch not in STANDARD_1020_COORDS:
                raise ValueError(f"Target channel {tch} not in 10-20 coords table")
            tx, ty = STANDARD_1020_COORDS[tch]
            best_dist = float('inf')
            best_idx = 0
            best_name = source_channels[0]

            for sch in source_channels:
                sch_lower = sch.lower()
                if sch_lower not in coord_lookup:
                    continue
                sch_key = coord_lookup[sch_lower]
                sx, sy = STANDARD_1020_COORDS[sch_key]
                dist = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = source_ch_lower[sch_lower]
                    best_name = sch

            mapped_indices.append(best_idx)
            mapping_info[tch] = best_name

    mapped_data = data[:, mapped_indices]
    return mapped_data, mapping_info


class FeatureExtractor:
    """
    Extract 63-dimensional features from 14-channel EEG.

    Feature vector layout (63 dims):
      [0:35]  = 5 bands x 7 DASM pairs
      [35:55] = 5 bands x 4 regional mean powers
      [55:60] = 5 bands x 1 global mean power
      [60]    = theta/beta ratio (global)
      [61]    = alpha/theta ratio (global)
      [62]    = alpha peak frequency (global)
    """

    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45),
    }

    BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    ASYM_PAIRS = [
        ('AF3', 'AF4'), ('F3', 'F4'), ('F7', 'F8'),
        ('FC5', 'FC6'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')
    ]

    REGIONS = {
        'frontal':   ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8'],
        'temporal':  ['T7', 'T8', 'FC5', 'FC6'],
        'parietal':  ['P7', 'P8'],
        'occipital': ['O1', 'O2'],
    }

    def __init__(self, fs=128, nperseg=None):
        """
        Args:
            fs: int, sampling frequency in Hz (default 128)
            nperseg: int, Welch segment length (default 2*fs)
        """
        self.fs = fs
        self.nperseg = nperseg if nperseg else min(256, 2 * fs)

    def _compute_band_power(self, freqs, psd, band):
        """Compute mean power in a frequency band."""
        fmin, fmax = band
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if len(idx) == 0:
            return 0.0
        return np.mean(psd[idx])

    def extract_features(self, eeg_window):
        """
        Extract 63-dimensional feature vector from a single EEG window.

        Args:
            eeg_window: ndarray (14, n_samples) — 14 channels x samples

        Returns:
            features: ndarray (63,) — 63-dimensional feature vector
        """
        features = np.zeros(63)
        ch_names = EPOC_CHANNELS
        n_channels = eeg_window.shape[0]

        # Compute PSD for each channel
        psd_dict = {}
        for i in range(n_channels):
            freqs, psd = welch(eeg_window[i], fs=self.fs, nperseg=self.nperseg)
            psd_dict[ch_names[i]] = {'freqs': freqs, 'psd': psd}

        # Feature block 1: DASM (5 bands x 7 pairs = 35)
        idx = 0
        for band_name in self.BAND_NAMES:
            band = self.BANDS[band_name]
            for ch_l, ch_r in self.ASYM_PAIRS:
                psd_l = self._compute_band_power(
                    psd_dict[ch_l]['freqs'], psd_dict[ch_l]['psd'], band)
                psd_r = self._compute_band_power(
                    psd_dict[ch_r]['freqs'], psd_dict[ch_r]['psd'], band)
                features[idx] = np.log1p(psd_l) - np.log1p(psd_r)
                idx += 1

        # Feature block 2: Regional mean powers (5 bands x 4 regions = 20)
        for band_name in self.BAND_NAMES:
            band = self.BANDS[band_name]
            for region_name, region_chs in self.REGIONS.items():
                powers = [
                    self._compute_band_power(
                        psd_dict[ch]['freqs'], psd_dict[ch]['psd'], band)
                    for ch in region_chs
                ]
                features[idx] = np.mean(powers)
                idx += 1

        # Feature block 3: Global mean powers (5 bands x 1 = 5)
        for band_name in self.BAND_NAMES:
            band = self.BANDS[band_name]
            powers = [
                self._compute_band_power(
                    psd_dict[ch]['freqs'], psd_dict[ch]['psd'], band)
                for ch in ch_names
            ]
            features[idx] = np.mean(powers)
            idx += 1

        # Feature 60: theta/beta ratio
        theta_power = np.mean([
            self._compute_band_power(
                psd_dict[ch]['freqs'], psd_dict[ch]['psd'], self.BANDS['theta'])
            for ch in ch_names
        ])
        beta_power = np.mean([
            self._compute_band_power(
                psd_dict[ch]['freqs'], psd_dict[ch]['psd'], self.BANDS['beta'])
            for ch in ch_names
        ])
        features[60] = theta_power / (beta_power + 1e-10)

        # Feature 61: alpha/theta ratio
        alpha_power = np.mean([
            self._compute_band_power(
                psd_dict[ch]['freqs'], psd_dict[ch]['psd'], self.BANDS['alpha'])
            for ch in ch_names
        ])
        features[61] = alpha_power / (theta_power + 1e-10)

        # Feature 62: Alpha peak frequency
        alpha_powers = []
        alpha_freqs = []
        for ch in ch_names:
            freqs = psd_dict[ch]['freqs']
            psd = psd_dict[ch]['psd']
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            if np.any(alpha_mask):
                alpha_powers.append(psd[alpha_mask])
                alpha_freqs.append(freqs[alpha_mask])
        if alpha_powers:
            all_powers = np.concatenate(alpha_powers)
            all_freqs = np.concatenate(alpha_freqs)
            peak_idx = np.argmax(all_powers)
            features[62] = all_freqs[peak_idx]

        return features

    def extract_windows(self, eeg_data, window_sec=4.0, overlap=0.5):
        """
        Extract feature vectors from a continuous EEG signal using sliding windows.

        Args:
            eeg_data: ndarray (14, n_total_samples) — 14 channels
            window_sec: float, window duration in seconds
            overlap: float, overlap ratio (0-1)

        Returns:
            features: ndarray (n_windows, 63) — feature matrix
        """
        fs = self.fs
        n_samples = eeg_data.shape[1]
        window_size = int(window_sec * fs)
        step = int(window_size * (1 - overlap))

        features_list = []
        for start in range(0, n_samples - window_size + 1, step):
            window = eeg_data[:, start:start + window_size]
            feat = self.extract_features(window)
            features_list.append(feat)

        if len(features_list) == 0:
            return np.zeros((0, 63))

        return np.array(features_list)
