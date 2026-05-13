#!/usr/bin/env python3
"""
prep01_map14_windowing_all.py — 14-channel mapping + sliding window segmentation.

This script:
  1. Reads raw EEG data from each of the 8 datasets
  2. Maps to 14-channel EPOC+ montage via nearest-neighbor on 10-20 coordinates
  3. Resamples to 128 Hz using integer-ratio polyphase resampling
  4. Applies sliding window segmentation (4s window, 2s step = 50% overlap)
  5. Saves segmented windows as .npz files to processed/prep01_windows/

Output: processed/prep01_windows/{dataset}_windows.npz
        Each .npz contains: windows (n_windows, n_samples, 14), subject_ids, trial_ids
"""

import sys
import numpy as np
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.feature_extraction import (
    map_channels_to_14, resample_eeg, EPOC_CHANNELS, FEAT_NAMES
)
from shared.logger import setup_logger


# ============================================================================
# Dataset-specific loaders
# ============================================================================

def load_dreamer(config):
    """Load DREAMER dataset."""
    import scipy.io as sio

    data_path = Path(config['data_paths']['DREAMER'])
    if not data_path.exists():
        raise FileNotFoundError(f"DREAMER data not found: {data_path}")

    mat = sio.loadmat(str(data_path))
    data = mat['DREAMER'][0, 0]

    eeg_data = data['EEG'][0, 0]

    n_subjects = eeg_data.shape[0]
    n_trials = eeg_data.shape[1]

    segments = []
    for subj in range(n_subjects):
        for trial in range(n_trials):
            eeg = eeg_data[subj, trial]  # (14, n_samples)
            segments.append({
                'eeg': eeg,
                'fs': 128,
                'dataset': 'DREAMER',
                'subject_id': f'DREAMER_S{subj + 1:02d}',
                'trial_id': trial,
                'channels': EPOC_CHANNELS,
            })

    return segments


def load_deap(config):
    """Load DEAP dataset from .dat files."""
    import pickle

    data_dir = Path(config['data_paths']['DEAP'])
    if not data_dir.exists():
        raise FileNotFoundError(f"DEAP data not found: {data_dir}")

    deap_channels = config['channel_mapping']['DEAP']['original_channels']

    segments = []
    dat_files = sorted(data_dir.glob('s*.dat'))
    for dat_file in dat_files:
        subj_idx = int(dat_file.stem[1:]) - 1
        with open(dat_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg = data['data']  # (40 trials, 40 channels, n_samples)
        n_trials = eeg.shape[0]

        for trial in range(n_trials):
            eeg_trial = eeg[trial]  # (40 channels, n_samples)
            segments.append({
                'eeg': eeg_trial,
                'fs': 128,
                'dataset': 'DEAP',
                'subject_id': f'DEAP_S{subj_idx + 1:02d}',
                'trial_id': trial,
                'channels': deap_channels,
                'raw_arousal': float(data['labels'][trial, 1]),
            })

    return segments


def load_mahnob(config):
    """Load MAHNOB-HCI dataset."""
    import h5py

    data_dir = Path(config['data_paths']['MAHNOB'])
    if not data_dir.exists():
        raise FileNotFoundError(f"MAHNOB data not found: {data_dir}")

    mahnob_channels = config['channel_mapping']['MAHNOB']['original_channels']

    segments = []
    session_dirs = sorted(data_dir.glob('Sessions/*'))
    for session_dir in session_dirs:
        subj_match = session_dir.name
        h5_files = list(session_dir.glob('*.hdf5')) + list(session_dir.glob('*.h5'))

        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'EEG' in f:
                        eeg = f['EEG'][:]
                        fs_raw = int(f['EEG'].attrs.get('samplerate', 256))
                    else:
                        continue
                    arousal = float(f['arousal'][()]) if 'arousal' in f else 5.0
            except Exception as e:
                print(f"  Warning: skip {h5_file}: {e}")
                continue

            if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T

            segments.append({
                'eeg': eeg,
                'fs': fs_raw,
                'dataset': 'MAHNOB',
                'subject_id': f'MAHNOB_{subj_match}',
                'trial_id': len(segments),
                'channels': mahnob_channels,
                'raw_arousal': arousal,
            })

    return segments


def load_seed(config):
    """Load SEED dataset."""
    data_dir = Path(config['data_paths']['SEED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED data not found: {data_dir}")

    seed_channels = config['channel_mapping']['SEED']['original_channels']

    segments = []
    import scipy.io as sio
    for mat_file in sorted(list(data_dir.glob('**/*.mat'))):
        try:
            data = sio.loadmat(str(mat_file))
            eeg = data.get('eeg', None)
            if eeg is None:
                continue

            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    segments.append({
                        'eeg': eeg[trial_idx],
                        'fs': 200,
                        'dataset': 'SEED',
                        'subject_id': f'SEED_{mat_file.stem}',
                        'trial_id': trial_idx,
                        'channels': seed_channels,
                    })
            elif eeg.ndim == 2:
                segments.append({
                    'eeg': eeg,
                    'fs': 200,
                    'dataset': 'SEED',
                    'subject_id': f'SEED_{mat_file.stem}',
                    'trial_id': 0,
                    'channels': seed_channels,
                })
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments


def load_seed_iv(config):
    """Load SEED-IV dataset."""
    data_dir = Path(config['data_paths']['SEED_IV'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED_IV data not found: {data_dir}")

    seed_iv_channels = config['channel_mapping']['SEED_IV']['original_channels']

    segments = []
    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            eeg = mat.get('eeg', None)
            if eeg is None:
                continue

            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    seg = {
                        'eeg': eeg[trial_idx],
                        'fs': 200,
                        'dataset': 'SEED_IV',
                        'subject_id': f'SEED_IV_{mat_file.parent.name}_{mat_file.stem}',
                        'trial_id': trial_idx,
                        'channels': seed_iv_channels,
                    }
                    segments.append(seg)
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments


def load_faced(config):
    """Load FACED dataset."""
    data_dir = Path(config['data_paths']['FACED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"FACED data not found: {data_dir}")

    faced_channels = config['channel_mapping']['FACED']['original_channels']

    segments = []
    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            eeg = mat.get('eeg', None)
            if eeg is None:
                continue

            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    segments.append({
                        'eeg': eeg[trial_idx],
                        'fs': 128,
                        'dataset': 'FACED',
                        'subject_id': f'FACED_{mat_file.parent.name}',
                        'trial_id': trial_idx,
                        'channels': faced_channels,
                    })
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments


def load_ds004572(config):
    """Load ds004572 (real hypnosis BIDS dataset)."""
    data_dir = Path(config['data_paths']['ds004572'])
    if not data_dir.exists():
        raise FileNotFoundError(f"ds004572 data not found: {data_dir}")

    ds_channels = config['channel_mapping']['ds004572']['original_channels']
    segments = []

    try:
        import mne
    except ImportError:
        print("  Warning: MNE not installed, skipping ds004572.")
        return segments

    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        func_dir = sub_dir / 'func'
        if not func_dir.exists():
            func_dir = sub_dir / 'eeg'

        eeg_files = sorted(func_dir.glob('*.fif')) + sorted(func_dir.glob('*.bdf'))
        for eeg_file in eeg_files:
            try:
                raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
            except Exception:
                try:
                    raw = mne.io.read_raw_bdf(str(eeg_file), preload=True, verbose=False)
                except Exception as e:
                    print(f"  Warning: skip {eeg_file}: {e}")
                    continue

            eeg_data = raw.get_data()
            orig_fs = int(raw.info['sfreq'])

            segments.append({
                'eeg': eeg_data,
                'fs': orig_fs,
                'dataset': 'ds004572',
                'subject_id': f'ds004572_{subj_name}',
                'trial_id': 0,
                'channels': ds_channels,
            })

    return segments


def load_ds006437(config):
    """Load ds006437 (real hypnosis BIDS dataset)."""
    data_dir = Path(config['data_paths']['ds006437'])
    if not data_dir.exists():
        raise FileNotFoundError(f"ds006437 data not found: {data_dir}")

    ds_channels = config['channel_mapping']['ds006437']['original_channels']
    segments = []

    try:
        import mne
    except ImportError:
        print("  Warning: MNE not installed, skipping ds006437.")
        return segments

    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        eeg_files = sorted(sub_dir.glob('**/*.fif')) + sorted(sub_dir.glob('**/*.bdf'))
        for eeg_file in eeg_files:
            try:
                raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
            except Exception:
                try:
                    raw = mne.io.read_raw_bdf(str(eeg_file), preload=True, verbose=False)
                except Exception as e:
                    print(f"  Warning: skip {eeg_file}: {e}")
                    continue

            eeg_data = raw.get_data()
            orig_fs = int(raw.info['sfreq'])

            segments.append({
                'eeg': eeg_data,
                'fs': orig_fs,
                'dataset': 'ds006437',
                'subject_id': f'ds006437_{subj_name}',
                'trial_id': 0,
                'channels': ds_channels,
            })

    return segments


DATASET_LOADERS = {
    'DREAMER': load_dreamer,
    'DEAP': load_deap,
    'MAHNOB': load_mahnob,
    'SEED': load_seed,
    'SEED_IV': load_seed_iv,
    'FACED': load_faced,
    'ds004572': load_ds004572,
    'ds006437': load_ds006437,
}


def process_dataset(dataset_name, config, logger):
    """
    Process a single dataset: load → map channels → resample → sliding window.

    Returns:
        all_windows: list of ndarray (n_samples_in_window, 14)
        all_subject_ids: list of str
        all_trial_ids: list of int
    """
    logger.info(f"Loading {dataset_name}...")
    t0 = time.time()

    loader = DATASET_LOADERS[dataset_name]
    segments = loader(config)

    if len(segments) == 0:
        logger.warning(f"  {dataset_name}: No segments loaded. Skipping.")
        return [], [], []

    logger.info(f"  {dataset_name}: {len(segments)} segments loaded ({time.time() - t0:.1f}s)")

    fs_target = config['features']['fs_target']
    window_sec = config['features']['window_sec']
    step_sec = config['features']['step_sec']

    all_windows = []
    all_subject_ids = []
    all_trial_ids = []

    for i, seg in enumerate(segments):
        eeg = seg['eeg']
        orig_fs = seg['fs']
        source_channels = seg['channels']

        try:
            # Step 1: Map to 14 EPOC+ channels (input as n_samples x n_ch or n_ch x n_samples)
            eeg_14ch, mapping_info = map_channels_to_14(eeg, source_channels, EPOC_CHANNELS)
            # eeg_14ch is now (n_samples, 14)

            # Step 2: Resample to target fs
            if orig_fs != fs_target:
                eeg_14ch = resample_eeg(eeg_14ch, orig_fs, fs_target)

            # Step 3: Sliding window segmentation
            win_len = int(window_sec * fs_target)
            step_len = int(step_sec * fs_target)
            n_samples = eeg_14ch.shape[0]

            n_windows = 0
            start = 0
            while start + win_len <= n_samples:
                window = eeg_14ch[start: start + win_len]
                all_windows.append(window)
                all_subject_ids.append(seg['subject_id'])
                all_trial_ids.append(seg['trial_id'])
                start += step_len
                n_windows += 1

            if (i + 1) % 10 == 0 or i == len(segments) - 1:
                logger.info(f"  Processed {i + 1}/{len(segments)} segments, "
                            f"{len(all_windows)} windows so far")

        except Exception as e:
            logger.warning(f"  Segment {i} failed: {e}. Skipping.")
            continue

    elapsed = time.time() - t0
    logger.info(f"  {dataset_name} done: {len(all_windows)} windows ({elapsed:.1f}s)")
    return all_windows, all_subject_ids, all_trial_ids


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep01', str(PROJECT_ROOT / config['logs_dir']))

    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_name in DATASET_LOADERS:
        try:
            windows, subj_ids, trial_ids = process_dataset(dataset_name, config, logger)

            if len(windows) == 0:
                logger.warning(f"{dataset_name}: No windows. Check data path.")
                summary[dataset_name] = {'status': 'empty', 'n_windows': 0}
                continue

            # Pad/truncate windows to uniform length if needed
            window_lengths = [w.shape[0] for w in windows]
            target_len = int(config['features']['window_sec'] * config['features']['fs_target'])
            padded_windows = []
            for w in windows:
                if w.shape[0] >= target_len:
                    padded_windows.append(w[:target_len])
                else:
                    pad = np.zeros((target_len - w.shape[0], w.shape[1]))
                    padded_windows.append(np.vstack([w, pad]))

            windows_array = np.array(padded_windows)

            save_path = out_dir / f'{dataset_name}_windows.npz'
            np.savez_compressed(
                save_path,
                windows=windows_array,
                subject_ids=np.array(subj_ids),
                trial_ids=np.array(trial_ids),
            )

            unique_subjs = list(set(subj_ids))
            summary[dataset_name] = {
                'status': 'ok',
                'n_windows': len(windows),
                'window_shape': list(windows_array.shape[1:]),
                'n_subjects': len(unique_subjs),
                'save_path': str(save_path),
            }

            logger.info(f"{dataset_name}: Saved {windows_array.shape} to {save_path}")

        except Exception as e:
            logger.error(f"{dataset_name}: Failed — {e}")
            import traceback
            traceback.print_exc()
            summary[dataset_name] = {'status': 'error', 'error': str(e)}

    summary_path = out_dir / 'prep01_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("prep01 complete.")
    for ds, info in summary.items():
        logger.info(f"  {ds}: {info.get('status', '?')} — {info.get('n_windows', 0)} windows")


if __name__ == '__main__':
    main()
