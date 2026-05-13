#!/usr/bin/env python3
"""
prep01_build_63feat_all_datasets.py — Build 63-dimensional feature matrices for all 8 datasets.

This script:
  1. Reads raw EEG data from each of the 8 datasets
  2. Maps to 14-channel EPOC+ montage via nearest-neighbor on 10-20 coordinates
  3. Resamples to 128 Hz
  4. Extracts 63-dimensional features per 2s window (non-overlapping)
     - 42 DASM (7 asymmetry pairs x 6 freq bands)
     - 21 log-bandpower (theta/alpha/beta x 7 left channels)
  5. Saves features as .npz files to processed/prep01_features/

Input:  Raw data from data_paths defined in config.yaml
Output: processed/prep01_features/{dataset}_features.npz
        Each .npz contains: features (n_windows, 63), subject_ids (n_windows,),
                            session_ids (n_windows,), timestamps (n_windows,)
"""

import sys
import numpy as np
from pathlib import Path
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.feature_extraction import (
    FeatureExtractor, map_channels_to_14, resample_to_target,
    EPOC_CHANNELS, FEATURE_ORDER
)
from shared.logger import setup_logger


# ============================================================================
# Dataset-specific loaders
# ============================================================================

def load_dreamer(config):
    """Load DREAMER dataset. Returns dict with EEG segments, subject info."""
    import scipy.io as sio

    data_path = Path(config['data_paths']['DREAMER'])
    if not data_path.exists():
        raise FileNotFoundError(f"DREAMER data not found: {data_path}")

    mat = sio.loadmat(str(data_path))
    data = mat['DREAMER'][0, 0]

    eeg_data = data['EEG'][0, 0]  # (n_subjects, n_videos, n_channels, n_samples)
    stim_labels = data['Valence'][0, 0]  # (n_subjects, n_videos)
    arous_labels = data['Arousal'][0, 0]

    n_subjects = eeg_data.shape[0]
    n_trials = eeg_data.shape[1]

    segments = []
    subject_ids = []
    trial_ids = []

    for subj in range(n_subjects):
        for trial in range(n_trials):
            # DREAMER EEG is already in EPOC+ compatible montage (14 channels)
            eeg = eeg_data[subj, trial]  # (14, n_samples)
            orig_fs = 128

            segments.append({
                'eeg': eeg,
                'fs': orig_fs,
                'dataset': 'DREAMER',
                'subject_id': f'DREAMER_S{subj + 1:02d}',
                'trial_id': trial,
                'channels': EPOC_CHANNELS,
            })
            subject_ids.append(f'DREAMER_S{subj + 1:02d}')
            trial_ids.append(trial)

    return segments, subject_ids, trial_ids


def load_deap(config):
    """Load DEAP dataset from .dat files."""
    import pickle

    data_dir = Path(config['data_paths']['DEAP'])
    if not data_dir.exists():
        raise FileNotFoundError(f"DEAP data not found: {data_dir}")

    deap_channels = config['channel_mapping']['DEAP']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    dat_files = sorted(data_dir.glob('s*.dat'))
    for dat_file in dat_files:
        subj_idx = int(dat_file.stem[1:]) - 1
        with open(dat_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg = data['data']  # (40 trials, 40 channels, n_samples)
        labels = data['labels']  # (40 trials, 4) — valence, arousal, dominance, liking

        n_trials = eeg.shape[0]
        orig_fs = 128

        for trial in range(n_trials):
            eeg_trial = eeg[trial]  # (40 channels, n_samples)
            raw_arousal = labels[trial, 1]

            segments.append({
                'eeg': eeg_trial,
                'fs': orig_fs,
                'dataset': 'DEAP',
                'subject_id': f'DEAP_S{subj_idx + 1:02d}',
                'trial_id': trial,
                'channels': deap_channels,
                'raw_arousal': float(raw_arousal),
            })
            subject_ids.append(f'DEAP_S{subj_idx + 1:02d}')
            trial_ids.append(trial)

    return segments, subject_ids, trial_ids


def load_mahnob(config):
    """Load MAHNOB-HCI dataset."""
    import h5py

    data_dir = Path(config['data_paths']['MAHNOB'])
    if not data_dir.exists():
        raise FileNotFoundError(f"MAHNOB data not found: {data_dir}")

    mahnob_channels = config['channel_mapping']['MAHNOB']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    session_dirs = sorted(data_dir.glob('Sessions/*'))
    for session_dir in session_dirs:
        subj_match = session_dir.name
        h5_files = list(session_dir.glob('*.hdf5'))
        if not h5_files:
            h5_files = list(session_dir.glob('*.h5'))

        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'EEG' in f:
                        eeg = f['EEG'][:]
                        fs_raw = f['EEG'].attrs.get('samplerate', 256)
                    else:
                        continue
                    if 'arousal' in f:
                        arousal = float(f['arousal'][()])
                    else:
                        arousal = 5.0  # default
            except Exception as e:
                print(f"  Warning: skip {h5_file}: {e}")
                continue

            if eeg.ndim == 2 and eeg.shape[0] < eeg.shape[1]:
                pass  # (n_channels, n_samples) — correct
            elif eeg.ndim == 2:
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
            subject_ids.append(f'MAHNOB_{subj_match}')
            trial_ids.append(len(segments) - 1)

    return segments, subject_ids, trial_ids


def load_seed(config):
    """Load SEED dataset (pre-extracted features or raw)."""
    data_dir = Path(config['data_paths']['SEED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED data not found: {data_dir}")

    seed_channels = config['channel_mapping']['SEED']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    # Check for pre-extracted features
    feat_dir = data_dir / 'ExtractedFeatures'
    if feat_dir.exists():
        import scipy.io as sio
        for session_file in sorted(feat_dir.glob('*.mat')):
            mat = sio.loadmat(str(session_file))
            # SEED extracted features: (n_trials, n_channels, n_features_per_band)
            # This is a simplified loader — adapt to actual file structure
            print(f"  Note: SEED pre-extracted features detected. Adapt loader for: {session_file}")
            continue

    # Fallback: look for raw EEG files
    eeg_dir = data_dir / 'raw_eeg'
    if not eeg_dir.exists():
        eeg_dir = data_dir

    raw_files = sorted(list(eeg_dir.glob('*.mat')) + list(eeg_dir.glob('*.npy')))
    for raw_file in raw_files:
        try:
            if raw_file.suffix == '.mat':
                import scipy.io as sio
                data = sio.loadmat(str(raw_file))
                eeg = data['eeg'] if 'eeg' in data else None
            elif raw_file.suffix == '.npy':
                eeg = np.load(str(raw_file))
            else:
                continue

            if eeg is None:
                continue

            orig_fs = 200  # SEED default
            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    segments.append({
                        'eeg': eeg[trial_idx],
                        'fs': orig_fs,
                        'dataset': 'SEED',
                        'subject_id': f'SEED_{raw_file.stem}',
                        'trial_id': trial_idx,
                        'channels': seed_channels,
                    })
                    subject_ids.append(f'SEED_{raw_file.stem}')
                    trial_ids.append(trial_idx)
            elif eeg.ndim == 2:
                segments.append({
                    'eeg': eeg,
                    'fs': orig_fs,
                    'dataset': 'SEED',
                    'subject_id': f'SEED_{raw_file.stem}',
                    'trial_id': 0,
                    'channels': seed_channels,
                })
                subject_ids.append(f'SEED_{raw_file.stem}')
                trial_ids.append(0)
        except Exception as e:
            print(f"  Warning: skip {raw_file}: {e}")

    return segments, subject_ids, trial_ids


def load_seed_iv(config):
    """Load SEED-IV dataset."""
    data_dir = Path(config['data_paths']['SEED_IV'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED_IV data not found: {data_dir}")

    seed_iv_channels = config['channel_mapping']['SEED_IV']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            eeg = mat.get('eeg', None)
            labels = mat.get('label', None)

            if eeg is None:
                continue

            orig_fs = 200
            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    seg = {
                        'eeg': eeg[trial_idx],
                        'fs': orig_fs,
                        'dataset': 'SEED_IV',
                        'subject_id': f'SEED_IV_{mat_file.parent.name}_{mat_file.stem}',
                        'trial_id': trial_idx,
                        'channels': seed_iv_channels,
                    }
                    if labels is not None and trial_idx < len(labels):
                        seg['raw_emotion'] = int(labels[trial_idx])
                    segments.append(seg)
                    subject_ids.append(seg['subject_id'])
                    trial_ids.append(trial_idx)
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments, subject_ids, trial_ids


def load_faced(config):
    """Load FACED dataset."""
    data_dir = Path(config['data_paths']['FACED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"FACED data not found: {data_dir}")

    faced_channels = config['channel_mapping']['FACED']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            eeg = mat.get('eeg', None)
            if eeg is None:
                continue

            orig_fs = 128
            if eeg.ndim == 3:
                for trial_idx in range(eeg.shape[0]):
                    segments.append({
                        'eeg': eeg[trial_idx],
                        'fs': orig_fs,
                        'dataset': 'FACED',
                        'subject_id': f'FACED_{mat_file.parent.name}',
                        'trial_id': trial_idx,
                        'channels': faced_channels,
                    })
                    subject_ids.append(f'FACED_{mat_file.parent.name}')
                    trial_ids.append(trial_idx)
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments, subject_ids, trial_ids


def load_ds004572(config):
    """Load ds004572 (real hypnosis BIDS dataset)."""
    data_dir = Path(config['data_paths']['ds004572'])
    if not data_dir.exists():
        raise FileNotFoundError(f"ds004572 data not found: {data_dir}")

    ds_channels = config['channel_mapping']['ds004572']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    # BIDS format: sub-XX/func/sub-XX_task-hypnosis_eeg.fif or .bdf
    try:
        import mne
    except ImportError:
        print("  Warning: MNE not installed, cannot load BIDS data. Skipping ds004572.")
        return segments, subject_ids, trial_ids

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
            orig_fs = raw.info['sfreq']

            segments.append({
                'eeg': eeg_data,
                'fs': orig_fs,
                'dataset': 'ds004572',
                'subject_id': f'ds004572_{subj_name}',
                'trial_id': 0,
                'channels': ds_channels,
            })
            subject_ids.append(f'ds004572_{subj_name}')
            trial_ids.append(0)

    return segments, subject_ids, trial_ids


def load_ds006437(config):
    """Load ds006437 (real hypnosis BIDS dataset)."""
    data_dir = Path(config['data_paths']['ds006437'])
    if not data_dir.exists():
        raise FileNotFoundError(f"ds006437 data not found: {data_dir}")

    ds_channels = config['channel_mapping']['ds006437']['original_channels']

    segments = []
    subject_ids = []
    trial_ids = []

    try:
        import mne
    except ImportError:
        print("  Warning: MNE not installed, cannot load BIDS data. Skipping ds006437.")
        return segments, subject_ids, trial_ids

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
            orig_fs = raw.info['sfreq']

            segments.append({
                'eeg': eeg_data,
                'fs': orig_fs,
                'dataset': 'ds006437',
                'subject_id': f'ds006437_{subj_name}',
                'trial_id': 0,
                'channels': ds_channels,
            })
            subject_ids.append(f'ds006437_{subj_name}')
            trial_ids.append(0)

    return segments, subject_ids, trial_ids


# ============================================================================
# Main pipeline
# ============================================================================

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


def process_dataset(dataset_name, config, extractor, logger):
    """
    Process a single dataset through the full feature extraction pipeline.

    Args:
        dataset_name: str, one of the 8 dataset names
        config: dict, loaded configuration
        extractor: FeatureExtractor instance
        logger: logging.Logger

    Returns:
        features: ndarray (n_total_windows, 63)
        subject_ids: list of str
        trial_ids: list of int
    """
    logger.info(f"Loading {dataset_name}...")
    t0 = time.time()

    loader = DATASET_LOADERS[dataset_name]
    segments, subj_ids, trial_ids = loader(config)

    if len(segments) == 0:
        logger.warning(f"  {dataset_name}: No segments loaded. Skipping.")
        return np.zeros((0, 63)), [], []

    logger.info(f"  {dataset_name}: {len(segments)} segments loaded ({time.time() - t0:.1f}s)")

    all_features = []
    all_subject_ids = []
    all_trial_ids = []

    for i, seg in enumerate(segments):
        eeg = seg['eeg']
        orig_fs = seg['fs']
        source_channels = seg['channels']

        # Step 1: Map to 14 EPOC+ channels
        try:
            eeg_14ch, mapping_info = map_channels_to_14(
                eeg.T, source_channels, EPOC_CHANNELS
            )
        except Exception as e:
            logger.warning(f"  Segment {i} channel mapping failed: {e}. Skipping.")
            continue

        # Step 2: Resample to 128 Hz
        eeg_14ch = eeg_14ch.T  # (14, n_samples)
        if orig_fs != 128:
            eeg_14ch = resample_to_target(eeg_14ch, orig_fs, 128)

        # Step 3: Extract features
        window_sec = config['features']['window_sec']
        stride_sec = config['features']['stride_sec']

        try:
            feats = extractor.extract_windows(eeg_14ch, window_sec=window_sec,
                                              stride_sec=stride_sec)
        except Exception as e:
            logger.warning(f"  Segment {i} feature extraction failed: {e}. Skipping.")
            continue

        if feats.shape[0] == 0:
            continue

        all_features.append(feats)
        all_subject_ids.extend([seg['subject_id']] * feats.shape[0])
        all_trial_ids.extend([seg['trial_id']] * feats.shape[0])

        if (i + 1) % 10 == 0 or i == len(segments) - 1:
            logger.info(f"  Processed {i + 1}/{len(segments)} segments, "
                        f"{sum(f.shape[0] for f in all_features)} windows so far")

    if not all_features:
        return np.zeros((0, 63)), [], []

    features = np.vstack(all_features)
    elapsed = time.time() - t0
    logger.info(f"  {dataset_name} done: {features.shape[0]} windows x {features.shape[1]} "
                f"features ({elapsed:.1f}s)")

    return features, all_subject_ids, all_trial_ids


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep01', str(PROJECT_ROOT / config['logs_dir']))

    extractor = FeatureExtractor(
        fs=config['features']['fs_target'],
        nperseg=config['features']['nperseg']
    )

    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_name in DATASET_LOADERS:
        try:
            features, subj_ids, trial_ids = process_dataset(
                dataset_name, config, extractor, logger
            )

            if features.shape[0] == 0:
                logger.warning(f"{dataset_name}: No features extracted. Check data path.")
                summary[dataset_name] = {'status': 'empty', 'n_windows': 0}
                continue

            # Save
            save_path = out_dir / f'{dataset_name}_features.npz'
            np.savez_compressed(
                save_path,
                features=features,
                subject_ids=np.array(subj_ids),
                trial_ids=np.array(trial_ids),
                feature_order=np.array(FEATURE_ORDER),
            )

            # Compute basic stats
            unique_subjs = list(set(subj_ids))
            summary[dataset_name] = {
                'status': 'ok',
                'n_windows': int(features.shape[0]),
                'n_features': int(features.shape[1]),
                'n_subjects': len(unique_subjs),
                'save_path': str(save_path),
            }

            logger.info(f"{dataset_name}: Saved {features.shape} to {save_path}")

        except Exception as e:
            logger.error(f"{dataset_name}: Failed — {e}")
            summary[dataset_name] = {'status': 'error', 'error': str(e)}

    # Save summary
    summary_path = out_dir / 'prep01_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("prep01 complete. Summary:")
    for ds, info in summary.items():
        logger.info(f"  {ds}: {info.get('status', '?')} — "
                     f"{info.get('n_windows', 0)} windows, "
                     f"{info.get('n_subjects', 0)} subjects")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
