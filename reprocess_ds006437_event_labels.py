"""reprocess_ds006437_event_labels.py — Re-process ds006437 with event-aware labels.

The ds006437 LIGHT dataset does not provide per-session objective hypnosis depth
scores. It does provide hypnotherapy phase-transition event markers in the EEGLAB
.set files (I_pressed, D_pressed, A_pressed, F_pressed, etc.). This script uses
those markers to assign each 2-second window a phase-based Awake/Light/Deep label
that is more fine-grained than the simple session-number proxy used previously.

Event-to-label mapping (from code/g_bids_script.m):
    I_pressed = Induction - Relaxation                -> Light (1)
    P_pressed = Imagining a perfect place - Safe haven  -> Light (1)
    S_pressed = Sit in Chair - Settling In              -> Deep  (2)
    D_pressed = Descend Stairs - Going deeper           -> Deep  (2)
    C_pressed = Crown Appears                           -> Deep  (2)
    L_pressed = Light Appears                           -> Deep  (2)
    R_pressed = Arc of Lights Appears                   -> Deep  (2)
    N_pressed = Neural Pathways                         -> Deep  (2)
    B_pressed = Bookmark Feeling                        -> Deep  (2)
    A_pressed = Ascend Stairs / Returning to wakefulness -> Awake (0)
    F_pressed = Finish Hypnotherapy / Closure           -> Awake (0)

Baseline recordings (task-baseline* in ses-0/1/4/8) are always Awake (0).

Output: overwrites processed/prep01_windows/ds006437_windows.npz
                              processed/prep02_features/ds006437_features.npz
                              processed/prep03_labels/ds006437_labels.npz
"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from shared.config_loader import load_config
from shared.feature_extraction import (
    map_channels_to_14, resample_eeg, EPOC_CHANNELS, FEAT_NAMES, ASYM_PAIRS, BANDS, BAND_NAMES
)
from shared.logger import setup_logger


EVENT_LABEL_MAP = {
    'I_pressed': 1,  # Induction - Relaxation
    'P_pressed': 1,  # Imagining a perfect place - Safe haven
    'S_pressed': 2,  # Sit in Chair - Settling In
    'D_pressed': 2,  # Descend Stairs - Going deeper
    'C_pressed': 2,  # Crown Appears
    'L_pressed': 2,  # Light Appears
    'R_pressed': 2,  # Arc of Lights Appears
    'N_pressed': 2,  # Neural Pathways
    'B_pressed': 2,  # Bookmark Feeling
    'A_pressed': 0,  # Ascend Stairs / Returning to wakefulness
    'F_pressed': 0,  # Finish Hypnotherapy / Closure
}


def load_set_with_events(set_path, logger=None):
    """Load EEGLAB .set file and return raw data, channel names, fs, and events."""
    import mne
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose='ERROR')
    data = raw.get_data().T  # (n_samples, n_channels)
    fs = int(raw.info['sfreq'])
    ch_names = raw.ch_names

    # Get events from annotations
    try:
        events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    except Exception as e:
        if logger:
            logger.warning(f'  No events in {set_path}: {e}')
        events = np.zeros((0, 3), dtype=int)
        event_id = {}

    # event_id maps event_name -> event_code; events[:,2] stores event_code
    # Build list of (sample, event_name)
    event_list = []
    for ev in events:
        code = ev[2]
        name = [k for k, v in event_id.items() if v == code]
        if name:
            event_list.append((int(ev[0]), name[0]))
    event_list.sort()
    return data, ch_names, fs, event_list


def assign_window_labels(n_samples, fs, event_list, is_baseline):
    """Return a label for each 2s/1s-step window based on event timeline."""
    window_len = int(2.0 * fs)   # 2 s
    step_len = int(1.0 * fs)     # 1 s
    n_windows = max(0, (n_samples - window_len) // step_len + 1)
    labels = np.full(n_windows, -1, dtype=int)

    if is_baseline or len(event_list) == 0:
        labels[:] = 0  # Awake for baseline or no-event files
        return labels

    # Sort events by sample and build phase intervals
    event_list = sorted(event_list, key=lambda x: x[0])
    # For each window, use the most recent event before the window center
    centers = np.arange(n_windows) * step_len + window_len // 2
    for i, c in enumerate(centers):
        # find last event <= c
        latest_event = None
        for sample, name in event_list:
            if sample <= c:
                latest_event = name
            else:
                break
        if latest_event is None:
            latest_event = event_list[0][1]  # use first event if before it
        labels[i] = EVENT_LABEL_MAP.get(latest_event, 1)
    return labels


def process_ds006437_events(config, logger):
    """Re-process ds006437 with event-based labels and return windows/labels."""
    data_dir = Path(config['data_paths']['ds006437'])
    if not data_dir.exists():
        raise FileNotFoundError(f'ds006437 data not found: {data_dir}')

    source_channels = config['channel_mapping']['ds006437']['original_channels']
    fs_target = config['features']['fs_target']
    window_len = int(config['features']['window_sec'] * fs_target)
    step_len = int(config['features']['step_sec'] * fs_target)

    all_windows = []
    all_subjects = []
    all_trials = []
    all_labels = []

    set_files = sorted(data_dir.rglob('*.set'))
    logger.info(f'Found {len(set_files)} .set files in ds006437')

    t0 = time.time()
    for idx, set_path in enumerate(set_files):
        # parse BIDS entities from filename
        parts = set_path.stem.split('_')
        subj = [p for p in parts if p.startswith('sub-')]
        sess = [p for p in parts if p.startswith('ses-')]
        task = [p for p in parts if p.startswith('task-')]
        subj = subj[0] if subj else 'unknown'
        sess = sess[0] if sess else 'unknown'
        task = task[0] if task else 'unknown'
        is_baseline = 'baseline' in task

        try:
            data, ch_names, fs, event_list = load_set_with_events(set_path, logger)
        except Exception as e:
            logger.warning(f'  Skip {set_path}: {e}')
            continue

        # Map to 14 channels
        mapped, mapping_info = map_channels_to_14(data, ch_names, EPOC_CHANNELS)
        # Resample to 128 Hz
        mapped = resample_eeg(mapped, fs, fs_target)
        n_samples = mapped.shape[0]

        # Generate windows
        n_windows = max(0, (n_samples - window_len) // step_len + 1)
        labels = assign_window_labels(n_samples, fs_target, event_list, is_baseline)

        if n_windows == 0:
            logger.warning(f'  {set_path.stem}: too short ({n_samples} samples)')
            continue
        if len(labels) != n_windows:
            logger.warning(f'  {set_path.stem}: window/label mismatch {n_windows} vs {len(labels)}')
            labels = labels[:n_windows]
            if len(labels) < n_windows:
                labels = np.pad(labels, (0, n_windows - len(labels)), constant_values=-1)

        for i in range(n_windows):
            s = i * step_len
            w = mapped[s:s + window_len]
            if w.shape[0] < window_len:
                continue
            all_windows.append(w.astype(np.float32))
            all_subjects.append(f'{subj}_{sess}')
            all_trials.append(f'{task}_win{i:04d}')
            all_labels.append(int(labels[i]))

        logger.info(f'  [{idx+1}/{len(set_files)}] {set_path.stem}: {n_windows} windows, '
                    f'events={len(event_list)}, baseline={is_baseline} ({time.time()-t0:.1f}s)')

    windows = np.array(all_windows, dtype=np.float32)
    labels = np.array(all_labels, dtype=int)
    subjects = np.array(all_subjects)
    trials = np.array(all_trials)

    logger.info(f'Total windows: {len(windows)}')
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    logger.info(f'Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}')

    return windows, labels, subjects, trials


def save_prep01(windows, subjects, trials, config, logger):
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / 'ds006437_windows.npz'
    np.savez_compressed(save_path,
                        windows=windows,
                        subject_ids=subjects,
                        trial_ids=trials)
    logger.info(f'prep01 saved: {save_path} shape={windows.shape}')
    return save_path


def extract_features_batch(windows, fs=128):
    """
    Fast batch feature extraction for windows (n, n_samples, 14).

    Feature order matches FEAT_NAMES:
      [0:42]  14 channels x 3 bands log-bandpower
      [42:63] 7 asymmetry pairs x 3 bands DASM
    """
    n, n_samples, n_ch = windows.shape
    assert n_ch == 14

    hamming = np.hamming(n_samples).reshape(1, -1, 1)
    windowed = windows * hamming
    fft = np.fft.rfft(windowed, axis=1)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)

    log_bp = np.zeros((n, n_ch, len(BAND_NAMES)))
    for bi, band in enumerate(BAND_NAMES):
        lo, hi = BANDS[band]
        mask = (freqs >= lo) & (freqs < hi)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        power = np.sum(psd[:, mask, :], axis=1) * df
        log_bp[:, :, bi] = np.log10(np.maximum(power, 1e-10))

    ch_idx = {ch: i for i, ch in enumerate(EPOC_CHANNELS)}
    dasm = np.zeros((n, len(ASYM_PAIRS), len(BAND_NAMES)))
    for pi, (l, r) in enumerate(ASYM_PAIRS):
        li = ch_idx[l]
        ri = ch_idx[r]
        dasm[:, pi, :] = log_bp[:, li, :] - log_bp[:, ri, :]

    features = np.concatenate([log_bp.reshape(n, -1), dasm.reshape(n, -1)], axis=1)
    assert features.shape[1] == 63
    return features


def run_prep02(config, logger):
    prep01_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(prep01_dir / 'ds006437_windows.npz', allow_pickle=True)
    windows = data['windows']
    subj_ids = data['subject_ids']
    trial_ids = data['trial_ids']
    data.close()

    logger.info(f'prep02: fast batch extracting features from {len(windows)} windows...')
    t0 = time.time()
    features = extract_features_batch(windows, fs=config['features']['fs_target'])
    out_path = out_dir / 'ds006437_features.npz'
    np.savez_compressed(out_path,
                        features=features,
                        subject_ids=subj_ids,
                        trial_ids=trial_ids,
                        feature_names=FEAT_NAMES)
    logger.info(f'prep02 saved: {out_path} shape={features.shape} ({time.time()-t0:.1f}s)')
    return out_path


def run_prep03(labels, subjects, trials, config, logger):
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep03_labels')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ds006437_labels.npz'
    np.savez_compressed(out_path,
                        labels=labels,
                        subject_ids=subjects,
                        trial_ids=trials,
                        label_type='event_phase_hypnosis_proxy')

    meta_df = pd.DataFrame({
        'subject': subjects,
        'trial': trials,
        'label': labels,
    })
    meta_df.to_csv(out_dir / 'ds006437_meta.csv', index=False)
    logger.info(f'prep03 saved: {out_path}')
    return out_path


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('reprocess_ds006437_events', str(PROJECT_ROOT / config['logs_dir']))

    logger.info('=' * 60)
    logger.info('Reprocessing ds006437 with event-phase-aware labels')
    logger.info('=' * 60)

    windows, labels, subjects, trials = process_ds006437_events(config, logger)
    save_prep01(windows, subjects, trials, config, logger)
    run_prep02(config, logger)
    run_prep03(labels, subjects, trials, config, logger)

    logger.info('Done.')


if __name__ == '__main__':
    main()
