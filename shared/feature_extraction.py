"""
feature_extraction.py — 14-channel mapping + 63-dimensional feature extraction.

Feature vector layout (63 dims, ORDER LOCKED via FEATURE_ORDER):
  [0:42]   = 7 DASM pairs x 6 band definitions
             (delta, theta, alpha, beta, gamma, broadband 1-45Hz)
             DASM[i] = log(1+P_left) - log(1+P_right)
  [42:63]  = 21 log-bandpower (theta, alpha, beta x 7 channels)
             BP_channels: AF3, F3, F7, FC5, T7, P7, O1
             log(1 + mean_PSD(band))

Channels are always in EPOC+ order: AF3, F7, F3, FC5, T7, P7, O1,
                                        O2, P8, T8, FC6, F4, F8, AF4
"""

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 10-20 standard electrode coordinates (normalized 2D, nose=up)
# ---------------------------------------------------------------------------
STANDARD_1020_COORDS = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'AF3': (-0.41, 0.81), 'AF4': (0.41, 0.81),
    'F7': (-0.81, 0.59),  'F8': (0.81, 0.59),
    'F3': (-0.55, 0.59),  'F4': (0.55, 0.59),
    'Fz': (0.0, 0.59),
    'FC5': (-0.71, 0.41), 'FC6': (0.71, 0.41),
    'FC1': (-0.24, 0.41), 'FC2': (0.24, 0.41),
    'FT7': (-0.85, 0.30), 'FT8': (0.85, 0.30),
    'T7': (-1.0, 0.0),    'T8': (1.0, 0.0),
    'TP7': (-0.85, -0.30),'TP8': (0.85, -0.30),
    'C3': (-0.55, 0.0),   'C4': (0.55, 0.0),
    'Cz': (0.0, 0.0),
    'C5': (-0.71, 0.0),   'C6': (0.71, 0.0),
    'C1': (-0.24, 0.0),   'C2': (0.24, 0.0),
    'CP5': (-0.71, -0.41),'CP6': (0.71, -0.41),
    'CP1': (-0.24, -0.41),'CP2': (0.24, -0.41),
    'P7': (-0.81, -0.59), 'P8': (0.81, -0.59),
    'P3': (-0.55, -0.59), 'P4': (0.55, -0.59),
    'Pz': (0.0, -0.59),
    'P9': (-0.90, -0.59), 'P10': (0.90, -0.59),
    'PO3': (-0.41, -0.81),'PO4': (0.41, -0.81),
    'PO7': (-0.60, -0.81),'PO8': (0.60, -0.81),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
    'Oz': (0.0, -0.95),
    'Iz': (0.0, -1.0),
    # Fpz / FPz variants
    'Fpz': (0.0, 0.95), 'FPz': (0.0, 0.95),
    # FCz variant
    'FCz': (0.0, 0.41),
    # Extra electrodes that may appear in SEED/SEED_IV/FACED
    'F1': (-0.41, 0.59), 'F2': (0.41, 0.59),
    'F5': (-0.71, 0.59), 'F6': (0.71, 0.59),
    'F9': (-0.90, 0.50), 'F10': (0.90, 0.50),
    'FT9': (-0.95, 0.15), 'FT10': (0.95, 0.15),
    'TP9': (-0.95, -0.15), 'TP10': (0.95, -0.15),
    'P1': (-0.41, -0.59), 'P2': (0.41, -0.59),
    'P5': (-0.71, -0.59), 'P6': (0.71, -0.59),
    'PO5': (-0.55, -0.85), 'PO6': (0.55, -0.85),
    'PO9': (-0.70, -0.90), 'PO10': (0.70, -0.90),
    'I1': (-0.20, -1.0), 'I2': (0.20, -1.0),
    'CB1': (-0.5, -1.0), 'CB2': (0.5, -1.0),
}

EPOC_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# ---------------------------------------------------------------------------
# Feature order (LOCKED) — must be consistent across ALL experiments
# ---------------------------------------------------------------------------
ASYM_PAIRS = [
    ('AF3', 'AF4'), ('F3', 'F4'), ('F7', 'F8'),
    ('FC5', 'FC6'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')
]

BAND_DEFS = {
    'delta':    (1, 4),
    'theta':    (4, 8),
    'alpha':    (8, 13),
    'beta':     (13, 30),
    'gamma':    (30, 45),
    'broadband': (1, 45),
}

# Bandpower channels (left-hemisphere + midline for asymmetric capture)
BP_CHANNELS = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1']
BP_BANDS = ['theta', 'alpha', 'beta']

def _build_feature_order():
    """Build the locked feature order list. Do not modify."""
    order = []
    # Block 1: DASM — 7 pairs x 6 bands = 42
    for pair in ASYM_PAIRS:
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']:
            order.append(f"DASM_{pair[0]}-{pair[1]}_{band_name}")
    # Block 2: log-bandpower — 3 bands x 7 channels = 21
    for band_name in BP_BANDS:
        for ch in BP_CHANNELS:
            order.append(f"logBP_{band_name}_{ch}")
    assert len(order) == 63, f"Feature order must be 63, got {len(order)}"
    return order

FEATURE_ORDER = _build_feature_order()


def resample_to_target(eeg_data, orig_fs, target_fs=128):
    """
    Resample EEG data to target sampling rate via linear interpolation.

    Args:
        eeg_data: ndarray (n_channels, n_samples)
        orig_fs: float, original sampling rate
        target_fs: float, target sampling rate (default 128 Hz)

    Returns:
        resampled: ndarray (n_channels, n_target_samples)
    """
    if orig_fs == target_fs:
        return eeg_data

    n_channels, n_samples = eeg_data.shape
    orig_time = np.linspace(0, 1, n_samples, endpoint=False)
    target_time = np.linspace(0, 1, int(n_samples * target_fs / orig_fs), endpoint=False)

    resampled = np.zeros((n_channels, len(target_time)))
    for ch in range(n_channels):
        interp_func = interp1d(orig_time, eeg_data[ch], kind='linear', fill_value='extrapolate')
        resampled[ch] = interp_func(target_time)

    return resampled


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

    Feature vector layout (63 dims, ORDER LOCKED):
      [0:42]   = 7 DASM pairs x 6 band definitions
      [42:63]  = 3 bands x 7 channels log-bandpower
    """

    def __init__(self, fs=128, nperseg=None):
        self.fs = fs
        self.nperseg = nperseg if nperseg else min(256, 2 * fs)

    def _compute_band_power(self, freqs, psd, fmin, fmax):
        """Compute mean power in a frequency band."""
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
        n_channels = min(eeg_window.shape[0], 14)

        # Compute PSD for all 14 channels
        psd_dict = {}
        for i in range(n_channels):
            freqs, psd = welch(eeg_window[i], fs=self.fs, nperseg=self.nperseg)
            psd_dict[EPOC_CHANNELS[i]] = {'freqs': freqs, 'psd': psd}

        # Feature block 1: DASM (7 pairs x 6 bands = 42)
        idx = 0
        for ch_l, ch_r in ASYM_PAIRS:
            for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']:
                fmin, fmax = BAND_DEFS[band_name]
                p_l = self._compute_band_power(
                    psd_dict[ch_l]['freqs'], psd_dict[ch_l]['psd'], fmin, fmax)
                p_r = self._compute_band_power(
                    psd_dict[ch_r]['freqs'], psd_dict[ch_r]['psd'], fmin, fmax)
                features[idx] = np.log1p(p_l) - np.log1p(p_r)
                idx += 1

        # Feature block 2: log-bandpower (3 bands x 7 channels = 21)
        for band_name in BP_BANDS:
            fmin, fmax = BAND_DEFS[band_name]
            for ch in BP_CHANNELS:
                bp = self._compute_band_power(
                    psd_dict[ch]['freqs'], psd_dict[ch]['psd'], fmin, fmax)
                features[idx] = np.log1p(bp)
                idx += 1

        return features

    def extract_windows(self, eeg_data, window_sec=2.0, stride_sec=2.0):
        """
        Extract feature vectors from a continuous EEG signal using sliding windows.

        Args:
            eeg_data: ndarray (14, n_total_samples) — 14 channels
            window_sec: float, window duration in seconds
            stride_sec: float, stride between consecutive windows in seconds

        Returns:
            features: ndarray (n_windows, 63) — feature matrix
        """
        fs = self.fs
        n_samples = eeg_data.shape[1]
        window_size = int(window_sec * fs)
        stride = int(stride_sec * fs)

        features_list = []
        for start in range(0, n_samples - window_size + 1, stride):
            window = eeg_data[:, start:start + window_size]
            feat = self.extract_features(window)
            features_list.append(feat)

        if len(features_list) == 0:
            return np.zeros((0, 63))

        return np.array(features_list)
