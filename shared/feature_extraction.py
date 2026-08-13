"""
feature_extraction.py — 14-channel mapping + 63-dimensional feature extraction.

All experiments' unified feature entry point. Feature order is LOCKED via FEAT_NAMES.

Feature vector layout (63 dims):
  [0:42]  = 14 channels x 3 bands Log-Bandpower  (ch_band)
  [42:63] = 7 asymmetry pairs x 3 bands DASM      (DASM(L-R)_band)

Bands: Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz)

Channels always in EPOC+ order:
  AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

Asymmetry pairs (left -> right):
  (AF3,AF4), (F7,F8), (F3,F4), (FC5,FC6), (T7,T8), (P7,P8), (O1,O2)
"""

import numpy as np
from scipy.signal import welch, resample_poly
from math import gcd


# ===========================================================================
# Constants: 14-channel EPOC+ layout
# ===========================================================================
EPOC_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# 7 symmetric electrode pairs (left -> right) for DASM
ASYM_PAIRS = [
    ('AF3', 'AF4'),
    ('F7',  'F8'),
    ('F3',  'F4'),
    ('FC5', 'FC6'),
    ('T7',  'T8'),
    ('P7',  'P8'),
    ('O1',  'O2'),
]

# 3 frequency bands (Paper Section 3.3)
BANDS = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
}
BAND_NAMES = ['Theta', 'Alpha', 'Beta']

# Aliases for backward compatibility
BAND_DEFS = BANDS
BP_CHANNELS = EPOC_CHANNELS
BP_BANDS = BAND_NAMES

# 10-20 standard electrode coordinates (normalized 2D, nose=up)
STANDARD_1020_COORDS = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'AF3': (-0.41, 0.81), 'AF4': (0.41, 0.81),
    'F7': (-0.81, 0.59),  'F8': (0.81, 0.59),
    'F3': (-0.55, 0.59),  'F4': (0.55, 0.59),
    'Fz': (0.00, 0.59),
    'FC5': (-0.71, 0.41), 'FC6': (0.71, 0.41),
    'FC1': (-0.24, 0.41), 'FC2': (0.24, 0.41),
    'T7': (-1.00, 0.00),  'T8': (1.00, 0.00),
    'C3': (-0.55, 0.00),  'C4': (0.55, 0.00),
    'Cz': (0.00, 0.00),
    'CP5': (-0.71, -0.41), 'CP6': (0.71, -0.41),
    'CP1': (-0.24, -0.41), 'CP2': (0.24, -0.41),
    'P7': (-0.81, -0.59), 'P8': (0.81, -0.59),
    'P3': (-0.55, -0.59), 'P4': (0.55, -0.59),
    'Pz': (0.00, -0.59),
    'PO3': (-0.41, -0.81), 'PO4': (0.41, -0.81),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
    'Oz': (0.00, -0.95),
    # Extra electrodes for SEED/SEED_IV/FACED
    'F1': (-0.41, 0.59), 'F2': (0.41, 0.59),
    'F5': (-0.71, 0.59), 'F6': (0.71, 0.59),
    'F9': (-0.90, 0.50), 'F10': (0.90, 0.50),
    'FT7': (-0.85, 0.30), 'FT8': (0.85, 0.30),
    'FT9': (-0.95, 0.15), 'FT10': (0.95, 0.15),
    'TP7': (-0.85, -0.30), 'TP8': (0.85, -0.30),
    'TP9': (-0.95, -0.15), 'TP10': (0.95, -0.15),
    'P1': (-0.41, -0.59), 'P2': (0.41, -0.59),
    'P5': (-0.71, -0.59), 'P6': (0.71, -0.59),
    'PO5': (-0.55, -0.85), 'PO6': (0.55, -0.85),
    'PO7': (-0.60, -0.81), 'PO8': (0.60, -0.81),
    'PO9': (-0.70, -0.90), 'PO10': (0.70, -0.90),
    'C5': (-0.71, 0.00), 'C6': (0.71, 0.00),
    'C1': (-0.24, 0.00), 'C2': (0.24, 0.00),
    'Iz': (0.00, -1.0),
    'Fpz': (0.00, 0.95), 'FPz': (0.00, 0.95),
    'FCz': (0.00, 0.41),
    'I1': (-0.20, -1.0), 'I2': (0.20, -1.0),
    'P9': (-0.90, -0.59), 'P10': (0.90, -0.59),
    'CB1': (-0.5, -1.0), 'CB2': (0.5, -1.0),
}


# ===========================================================================
# Feature order (LOCKED) — must be consistent across ALL experiments
# ===========================================================================
def _make_feat_names():
    """
    Generate 63-dim feature name list (order fixed).

    [0:42]  = 14 channels x 3 bands Log-Bandpower
    [42:63] = 7 pairs x 3 bands DASM
    """
    names = []
    # Log-Bandpower: ch_band
    for ch in EPOC_CHANNELS:
        for band in BAND_NAMES:
            names.append(f'{ch}_{band}')
    # DASM: DASM(L-R)_band
    for (l, r) in ASYM_PAIRS:
        for band in BAND_NAMES:
            names.append(f'DASM({l}-{r})_{band}')
    assert len(names) == 63, f"Feature dimension error: {len(names)}"
    return names


FEAT_NAMES = _make_feat_names()
FEATURE_ORDER = FEAT_NAMES  # Alias for backward compatibility


# ===========================================================================
# Channel mapping
# ===========================================================================
def map_channels_to_14(data, source_channels, target_channels=None):
    """
    Map any EEG montage to 14 EPOC+ channels via nearest-neighbor on 10-20 coords.

    Args:
        data: ndarray (n_samples, n_channels) or (n_channels, n_samples)
        source_channels: list of str, channel names in source data
        target_channels: list of str, default EPOC_CHANNELS

    Returns:
        mapped_data: ndarray (n_samples, 14)
        mapping_info: dict {target_ch: source_ch}
    """
    if target_channels is None:
        target_channels = EPOC_CHANNELS

    # Ensure data is (n_samples, n_channels)
    if data.ndim == 2 and data.shape[0] == len(source_channels):
        data = data.T

    src_lower = {ch.lower(): i for i, ch in enumerate(source_channels)}
    mapping_info = {}
    mapped_indices = []

    for tch in target_channels:
        tch_lower = tch.lower()
        if tch_lower in src_lower:
            mapped_indices.append(src_lower[tch_lower])
            mapping_info[tch] = tch
        else:
            # Nearest neighbor in 10-20 coordinates
            if tch not in STANDARD_1020_COORDS:
                raise ValueError(f"Target channel {tch} not in 10-20 coords table")
            tx, ty = STANDARD_1020_COORDS[tch]
            best_dist = float('inf')
            best_src_idx = 0
            best_src_name = source_channels[0]

            for sch in source_channels:
                sch_key = None
                for k in STANDARD_1020_COORDS:
                    if k.lower() == sch.lower():
                        sch_key = k
                        break
                if sch_key is None:
                    continue
                sx, sy = STANDARD_1020_COORDS[sch_key]
                dist = (tx - sx) ** 2 + (ty - sy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_src_idx = src_lower.get(sch.lower(), 0)
                    best_src_name = sch

            mapped_indices.append(best_src_idx)
            mapping_info[tch] = best_src_name

    mapped = data[:, mapped_indices]
    return mapped, mapping_info


# ===========================================================================
# Resampling (integer-ratio via resample_poly, avoids float precision issues)
# ===========================================================================
def resample_eeg(data, fs_orig, fs_target=128):
    """
    Resample EEG data using integer-ratio polyphase resampling.

    Args:
        data: ndarray (n_samples, n_channels)
        fs_orig: int, original sampling rate
        fs_target: int, target sampling rate (default 128)

    Returns:
        resampled: ndarray (n_samples_new, n_channels)
    """
    if fs_orig == fs_target:
        return data
    g = gcd(int(fs_orig), int(fs_target))
    up = int(fs_target) // g
    down = int(fs_orig) // g
    return resample_poly(data, up, down, axis=0)


# Backward-compatible alias
def resample_to_target(eeg_data, orig_fs, target_fs=128):
    """
    Resample EEG data. Accepts (n_channels, n_samples) or (n_samples, n_channels).
    Backward-compatible wrapper around resample_eeg.
    """
    if eeg_data.ndim == 2 and eeg_data.shape[0] <= eeg_data.shape[1] and eeg_data.shape[0] < 20:
        # Likely (n_channels, n_samples) — transpose for resample_eeg
        eeg_data = eeg_data.T
    result = resample_eeg(eeg_data, int(orig_fs), int(target_fs))
    return result.T  # Back to (n_channels, n_samples)


# ===========================================================================
# Single window feature extraction
# ===========================================================================
def extract_features_window(window, fs=128):
    """
    Extract 63-dimensional feature vector from a single EEG window.

    Args:
        window: ndarray (n_samples, 14) — already mapped to EPOC_CHANNELS order
        fs: int, sampling rate (default 128 Hz)

    Returns:
        feat: ndarray (63,) — 63-dimensional feature vector
    """
    n_ch = window.shape[1]
    assert n_ch == 14, f"Channel count error: {n_ch}, expected 14"

    ch_idx = {ch: i for i, ch in enumerate(EPOC_CHANNELS)}

    # --- Log-Bandpower (42 dims): 14 channels x 3 bands ---
    # Per Paper §3.2: Welch PSD with nperseg=128 (1s @ 128Hz)
    log_bp = np.zeros((14, 3))
    for ci, ch in enumerate(EPOC_CHANNELS):
        f, psd = welch(window[:, ci], fs=fs, nperseg=min(128, len(window)))
        for bi, band in enumerate(BAND_NAMES):
            lo, hi = BANDS[band]
            mask = (f >= lo) & (f < hi)
            # Manual trapezoidal integration (avoids np.trapz removal in NumPy 2.0+)
            if mask.sum() > 1:
                f_masked = f[mask]
                p_masked = psd[mask]
                # Uniform spacing: sum * df approximates integral
                df = f_masked[1] - f_masked[0] if len(f_masked) > 1 else 1.0
                power = float(np.sum(p_masked) * df)
            else:
                power = 1e-10
            log_bp[ci, bi] = np.log10(max(power, 1e-10))

    feat_bp = log_bp.flatten()  # (42,)

    # --- DASM (21 dims): 7 pairs x 3 bands ---
    feat_dasm = np.zeros(len(ASYM_PAIRS) * 3)
    for pi, (l, r) in enumerate(ASYM_PAIRS):
        li = ch_idx[l]
        ri = ch_idx[r]
        for bi in range(3):
            feat_dasm[pi * 3 + bi] = log_bp[li, bi] - log_bp[ri, bi]

    feat = np.concatenate([feat_bp, feat_dasm])
    assert feat.shape[0] == 63
    return feat


# ===========================================================================
# Sliding window feature extraction
# ===========================================================================
def extract_features_sliding(eeg, fs=128, window_sec=2.0, step_sec=1.0):
    """
    Extract features from continuous EEG using sliding windows.

    Per Paper §3.2: Window W=256 samples (2s), Step S=128 (1s) at 128Hz.

    Args:
        eeg: ndarray (n_samples, 14) — already channel-mapped and resampled
        fs: int, sampling rate
        window_sec: float, window duration in seconds
        step_sec: float, step between consecutive windows in seconds

    Returns:
        features: ndarray (n_windows, 63)
    """
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    n_samples = eeg.shape[0]

    features = []
    start = 0
    while start + win_len <= n_samples:
        window = eeg[start: start + win_len]
        feat = extract_features_window(window, fs=fs)
        features.append(feat)
        start += step_len

    if len(features) == 0:
        return np.zeros((0, 63))
    return np.vstack(features)


# ===========================================================================
# Subject-level Z-score normalization
# ===========================================================================
def subject_zscore(X, eps=1e-8):
    """
    Subject-level Z-score normalization (compute mean/std over sample dimension).

    Args:
        X: ndarray (n_windows, 63)

    Returns:
        X_norm: ndarray (n_windows, 63)
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sigma


# ===========================================================================
# Save/Load feature names (ensure order locking)
# ===========================================================================
def save_feat_names(path='feat_names.json'):
    """Save FEAT_NAMES to JSON for verification."""
    import json
    with open(path, 'w') as f:
        json.dump(FEAT_NAMES, f, indent=2)
    print(f"[feature_extraction] Feature names saved: {path}")


def load_and_verify_feat_names(path='feat_names.json'):
    """Load and verify feature names match current FEAT_NAMES."""
    import json
    with open(path) as f:
        saved = json.load(f)
    assert saved == FEAT_NAMES, (
        "Feature order mismatch! Please regenerate feature cache.\n"
        f"Expected: {FEAT_NAMES[:5]}...\n"
        f"Actual:   {saved[:5]}..."
    )
    return saved


# ===========================================================================
# Backward-compatible FeatureExtractor class
# ===========================================================================
class FeatureExtractor:
    """
    Extract 63-dimensional features from 14-channel EEG.

    Feature vector layout (63 dims, ORDER LOCKED):
      [0:42]  = 14 channels x 3 bands Log-Bandpower
      [42:63] = 7 pairs x 3 bands DASM

    Backward-compatible wrapper around module-level functions.
    """

    def __init__(self, fs=128, nperseg=None):
        self.fs = fs
        self.nperseg = nperseg if nperseg else min(128, fs)  # 1s segment per Paper §3.2

    def extract_features(self, eeg_window):
        """
        Extract 63-dimensional feature vector from a single EEG window.

        Args:
            eeg_window: ndarray (14, n_samples) or (n_samples, 14)

        Returns:
            features: ndarray (63,)
        """
        # Handle (14, n_samples) input format
        if eeg_window.ndim == 2 and eeg_window.shape[0] <= 14:
            eeg_window = eeg_window.T
        return extract_features_window(eeg_window, fs=self.fs)

    def extract_windows(self, eeg_data, window_sec=2.0, stride_sec=1.0):
        """
        Extract feature vectors from continuous EEG using sliding windows.

        Args:
            eeg_data: ndarray (14, n_total_samples)
            window_sec: float, window duration in seconds
            stride_sec: float, stride between consecutive windows in seconds

        Returns:
            features: ndarray (n_windows, 63)
        """
        # Handle (14, n_samples) input format
        if eeg_data.ndim == 2 and eeg_data.shape[0] <= 14:
            eeg_data = eeg_data.T
        return extract_features_sliding(eeg_data, fs=self.fs,
                                         window_sec=window_sec,
                                         step_sec=stride_sec)
