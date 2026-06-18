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
    """Load DREAMER dataset.

    DREAMER.mat structure (with struct_as_record=False, squeeze_me=True):
      DREAMER.Data[i]: subject i
        .EEG.stimuli[j]: (n_samples, 14) ndarray — trial j
        .EEG.baseline[j]: (n_samples, 14) ndarray — baseline for trial j
        .ScoreArousal[j]: uint8 — arousal score (1-5)
      DREAMER.EEG_Electrodes: list of 14 channel names
      DREAMER.EEG_SamplingRate: 128 Hz
    """
    import scipy.io as sio

    data_path = Path(config['data_paths']['DREAMER'])
    if not data_path.exists():
        raise FileNotFoundError(f"DREAMER data not found: {data_path}")

    mat = sio.loadmat(str(data_path), struct_as_record=False, squeeze_me=True)
    dreamer = mat['DREAMER']

    subjects = dreamer.Data
    n_subjects = len(subjects)
    n_trials = 18  # fixed in DREAMER

    segments = []
    for subj_idx in range(n_subjects):
        subj = subjects[subj_idx]
        eeg_struct = subj.EEG
        arousal_scores = subj.ScoreArousal  # (18,) array

        # Get channel names from first trial if available
        first_stim = eeg_struct.stimuli[0]
        if hasattr(first_stim, 'dtype') and first_stim.dtype.names:
            # Trial has named fields, extract EEG array
            eeg = first_stim[0]  # depends on field name
        else:
            eeg = first_stim  # already (n_samples, 14)

        for trial_idx in range(n_trials):
            stim_eeg = eeg_struct.stimuli[trial_idx]
            # stim_eeg should be (n_samples, 14) ndarray
            if hasattr(stim_eeg, 'dtype') and stim_eeg.dtype.names:
                stim_eeg = stim_eeg[0]

            arousal_val = int(arousal_scores[trial_idx])

            segments.append({
                'eeg': stim_eeg,
                'fs': 128,
                'dataset': 'DREAMER',
                'subject_id': f'DREAMER_S{subj_idx + 1:02d}',
                'trial_id': trial_idx,
                'channels': EPOC_CHANNELS,  # DREAMER already uses EPOC+ 14ch layout
                'raw_arousal': float(arousal_val),
            })

    return segments


def load_deap(config):
    """Load DEAP dataset from preprocessed .dat files.

    DEAP preprocessed format:
      data['data']: (40 trials, 40 channels, n_samples) — first 32 are EEG
      data['labels']: (40 trials, 4) — valence, arousal, dominance, liking
    Channels in config: 32 EEG channels matching source_channels list.
    """
    import pickle

    data_dir = Path(config['data_paths']['DEAP'])
    if not data_dir.exists():
        raise FileNotFoundError(f"DEAP data not found: {data_dir}")

    deap_channels = config['channel_mapping']['DEAP']['original_channels']
    n_eeg_ch = len(deap_channels)  # 32

    segments = []
    dat_files = sorted(data_dir.glob('s*.dat'))
    for dat_file in dat_files:
        subj_idx = int(dat_file.stem[1:]) - 1
        with open(dat_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg = data['data']  # (40 trials, 40 channels, n_samples)
        labels = data['labels']  # (40 trials, 4)
        n_trials = eeg.shape[0]

        for trial in range(n_trials):
            # Take only the first 32 EEG channels (last 8 are peripheral)
            eeg_trial = eeg[trial, :n_eeg_ch, :]  # (32, n_samples)
            segments.append({
                'eeg': eeg_trial,
                'fs': 128,
                'dataset': 'DEAP',
                'subject_id': f'DEAP_S{subj_idx + 1:02d}',
                'trial_id': trial,
                'channels': deap_channels,
                'raw_arousal': float(labels[trial, 1]),  # column 1 = arousal
            })

    return segments


def load_mahnob(config):
    """Load MAHNOB-HCI dataset (BDF files + session.xml for arousal labels)."""
    data_dir = Path(config['data_paths']['MAHNOB'])
    if not data_dir.exists():
        raise FileNotFoundError(f"MAHNOB data not found: {data_dir}")

    mahnob_channels = config['channel_mapping']['MAHNOB']['original_channels']
    fs_raw = 256  # MAHNOB-HCI recorded at 256 Hz

    segments = []
    import xml.etree.ElementTree as ET

    # Support two possible directory layouts:
    #   (a) data_paths.MAHNOB points directly to the Sessions/ folder
    #   (b) data_paths.MAHNOB/Sessions/ exists as a sub-folder
    if (data_dir / 'Sessions').is_dir():
        session_dirs = sorted((data_dir / 'Sessions').glob('*'))
    else:
        session_dirs = sorted(data_dir.glob('*'))

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue
        subj_match = session_dir.name

        # Parse session.xml for arousal/valence self-report and real subject ID
        xml_file = session_dir / 'session.xml'
        arousal = 5.0  # default middle value
        real_subject = session_dir.name  # fallback
        if xml_file.exists():
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()
                arousal = float(root.get('feltArsl', 5.0))
                real_subject = root.get('subjectID') or root.get('subjectid') or root.get('subject_id') or session_dir.name
            except Exception:
                pass

        # Load BDF files (one per trial)
        bdf_files = sorted(session_dir.glob('*.bdf'))
        for trial_idx, bdf_file in enumerate(bdf_files):
            try:
                import mne
                raw = mne.io.read_raw_bdf(str(bdf_file), preload=True, verbose=False)
                
                # Only keep EEG channels (first 32 channels, exclude EXG/GSR/Resp/Temp/Status)
                # Channel indices 0-31 are EEG, 32-46 are non-EEG (EXG1-8, GSR1-2, etc.)
                eeg_data = raw.get_data()  # (47, n_samples)
                eeg_only = eeg_data[:32, :]  # Take only first 32 EEG channels
                
                actual_fs = int(raw.info['sfreq'])
            except ImportError:
                print("  Warning: MNE not installed, skipping MAHNOB BDF files.")
                break
            except Exception as e:
                print(f"  Warning: skip {bdf_file}: {e}")
                continue

            # Ensure (n_channels, n_samples)
            if eeg_only.ndim != 2:
                continue

            segments.append({
                'eeg': eeg_only,
                'fs': actual_fs,
                'dataset': 'MAHNOB',
                'subject_id': f'MAHNOB_{real_subject}',
                'trial_id': trial_idx,
                'channels': mahnob_channels,
                'raw_arousal': float(arousal),
            })

    return segments


# ---- 10-20 Channel Coordinates for nearest-neighbor mapping ----
_1020_COORDS = {
    'Fp1': (-85, 60), 'Fpz': (0, 90), 'Fp2': (85, 60),
    'AF3': (-55, 80), 'AF4': (55, 80),
    'F7': (-85, 30), 'F5': (-53, 42), 'F3': (-53, 52), 'F1': (-21, 60), 'Fz': (0, 55),
    'F2': (21, 60), 'F4': (53, 52), 'F6': (53, 42), 'F8': (85, 30),
    'FC5': (-75, 5), 'FC3': (-53, 18), 'FC1': (-21, 28), 'FCz': (0, 27),
    'FC2': (21, 28), 'FC4': (53, 18), 'FC6': (75, 5),
    'T7': (-95, -8), 'C5': (-53, -5), 'C3': (-53, 10), 'C1': (-21, 15), 'Cz': (0, 13),
    'C2': (21, 15), 'C4': (53, 10), 'C6': (53, -5), 'T8': (95, -8),
    'TP7': (-85, -30), 'CP5': (-53, -25), 'CP3': (-53, -12), 'CP1': (-21, -3),
    'CPz': (0, -3), 'CP2': (21, -3), 'CP4': (53, -12), 'CP6': (53, -25), 'TP8': (85, -30),
    'P7': (-80, -55), 'P5': (-53, -42), 'P3': (-53, -30), 'P1': (-21, -20),
    'Pz': (0, -22), 'P2': (21, -20), 'P4': (53, -30), 'P6': (53, -42), 'P8': (80, -55),
    'PO7': (-65, -70), 'PO3': (-45, -58), 'POz': (0, -58), 'PO4': (45, -58), 'PO8': (65, -70),
    'O1': (-45, -85), 'Oz': (0, -90), 'O2': (45, -85), 'Iz': (0, -100),
    'FT7': (-75, 15), 'FT8': (75, 15), 'FT9': (-92, 15), 'FT10': (92, 15),
    'F9': (-100, 20), 'F10': (100, 20), 'TP9': (-92, -20), 'TP10': (92, -20),
    'P9': (-80, -70), 'P10': (80, -70), 'PO5': (-55, -58), 'PO6': (55, -58),
}
# Normalize all keys for case-insensitive lookup
_1020_COORDS_CI = {k.lower(): v for k, v in _1020_COORDS.items()}


def _nearest_mapping(src_channels, tgt_channels):
    """Build source→target channel indices via nearest-neighbor on 10-20 coordinates."""
    import numpy as np
    src_coords = []
    for ch in src_channels:
        pos = _1020_COORDS_CI.get(ch.lower())
        if pos is None:
            pos = (0, 0)  # fallback for unknown channels
        src_coords.append(pos)
    tgt_coords = []
    for ch in tgt_channels:
        pos = _1020_COORDS_CI.get(ch.lower(), (0, 0))
        tgt_coords.append(pos)
    src_coords = np.array(src_coords)
    tgt_coords = np.array(tgt_coords)
    # For each target channel, find nearest source channel
    indices = []
    for tc in tgt_coords:
        dists = np.sqrt(np.sum((src_coords - tc) ** 2, axis=1))
        indices.append(int(np.argmin(dists)))
    return np.array(indices, dtype=int)

# ---- End channel mapping ----


def load_seed(config):
    """Load SEED dataset (pre-extracted DE features from ExtractedFeatures_1s).

    SEED provides pre-computed Differential Entropy (DE) features in .mat files.
    Each file contains 15 trials: de_movingAve1...de_movingAve15, shape (62, time, 5).
    Bands: delta=0, theta=1, alpha=2, beta=3, gamma=4
    We extract theta/alpha/beta bands, map 62→14 EPOC channels via nearest-1020.
    """
    data_dir = Path(config['data_paths']['SEED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED data not found: {data_dir}")

    seed_62ch = config['channel_mapping']['SEED']['original_channels']
    from shared.feature_extraction import EPOC_CHANNELS

    # Build 62→14 mapping via nearest-neighbor on 10-20 coords
    src_indices = _nearest_mapping(seed_62ch, EPOC_CHANNELS)

    # SEED band indices: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
    THETA_IDX, ALPHA_IDX, BETA_IDX = 1, 2, 3

    segments = []
    import scipy.io as sio
    for mat_file in sorted(list(data_dir.rglob('*.mat'))):
        try:
            data = sio.loadmat(str(mat_file))
            # Collect all de_movingAve trials
            trial_keys = sorted([k for k in data.keys() if k.startswith('de_movingAve')])
            if not trial_keys:
                continue

            for trial_key in trial_keys:
                de = data[trial_key]  # (62, time_segments, 5)
                if de.ndim != 3 or de.shape[0] < 10:
                    continue
                # Map 62 → 14 channels
                de_14 = de[src_indices, :, :]  # (14, time, 5)
                # Extract theta/alpha/beta → (14, time, 3)
                de_tab = np.stack([de_14[:, :, THETA_IDX], de_14[:, :, ALPHA_IDX], de_14[:, :, BETA_IDX]], axis=-1)
                # Rearrange to (time, 14, 3)
                win_feat = de_tab.transpose(1, 0, 2)

                # SEED filename format: <subject>_<date>, e.g. "10_20131130"
                subject_num = mat_file.stem.split('_')[0]
                trial_num = int(trial_key.replace('de_movingAve', '').replace('de_LDS', ''))

                segments.append({
                    '_precomputed': True,
                    '_win_feat': win_feat,
                    'subject_id': f'SEED_{subject_num}',
                    'trial_id': trial_num,
                    'dataset': 'SEED',
                })
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments


def load_seed_iv(config):
    """Load SEED-IV dataset (pre-extracted DE features).

    SEED-IV provides pre-computed DE features in .mat files per subject per session.
    Each file contains 24 trials: de_movingAve1...de_movingAve24, shape (62, time, 5).
    Bands: delta=0, theta=1, alpha=2, beta=3, gamma=4
    """
    data_dir = Path(config['data_paths']['SEED_IV'])
    if not data_dir.exists():
        raise FileNotFoundError(f"SEED_IV data not found: {data_dir}")

    seed_iv_62ch = config['channel_mapping']['SEED_IV']['original_channels']
    from shared.feature_extraction import EPOC_CHANNELS

    src_indices = _nearest_mapping(seed_iv_62ch, EPOC_CHANNELS)
    THETA_IDX, ALPHA_IDX, BETA_IDX = 1, 2, 3

    segments = []
    import scipy.io as sio
    for mat_file in sorted(data_dir.rglob('*.mat')):
        try:
            data = sio.loadmat(str(mat_file))
            trial_keys = sorted([k for k in data.keys() if k.startswith('de_movingAve')])
            if not trial_keys:
                # SEED_IV may also have psd_movingAve — use de_LDS as fallback
                trial_keys = sorted([k for k in data.keys() if k.startswith('de_LDS')])
            if not trial_keys:
                continue

            for trial_key in trial_keys:
                de = data[trial_key]
                if de.ndim != 3 or de.shape[0] < 10:
                    continue
                de_14 = de[src_indices, :, :]
                de_tab = np.stack([de_14[:, :, THETA_IDX], de_14[:, :, ALPHA_IDX], de_14[:, :, BETA_IDX]], axis=-1)
                win_feat = de_tab.transpose(1, 0, 2)

                # SEED-IV filename format: <subject>_<date>, e.g. "10_20151014"
                subject_num = mat_file.stem.split('_')[0]
                trial_num = int(trial_key.replace('de_movingAve', '').replace('de_LDS', ''))

                segments.append({
                    '_precomputed': True,
                    '_win_feat': win_feat,
                    'subject_id': f'SEED_IV_{subject_num}',
                    'trial_id': trial_num,
                    'dataset': 'SEED_IV',
                })
        except Exception as e:
            print(f"  Warning: skip {mat_file}: {e}")

    return segments


def load_faced(config):
    """Load FACED dataset.

    FACED provides pre-extracted DE (Differential Entropy) features in .pkl.pkl files.
    Format: ndarray (28 videos, 32 channels, 30 windows, 5 bands)
    Bands: 0=delta(1-4Hz), 1=theta(4-8Hz), 2=alpha(8-14Hz), 3=beta(14-30Hz), 4=gamma(30-47Hz)
    Channels: 32-channel cap (mapped to 14 EPOC channels via nearest-neighbor)
    Sampling: 250 Hz, 1-second windows (pre-computed)

    Since FACED provides pre-extracted features (not raw EEG), we reconstruct a pseudo-raw
    signal from the DE features that preserves the theta/alpha/beta band structure needed
    for the 63-dim feature pipeline.  Concretely, for each 1-second window we place the
    DE value of each channel into the corresponding frequency band using a single-bin
    sinusoidal signal scaled to reproduce that DE value.  This allows the downstream
    channel-mapping and feature-extraction stages to operate unchanged.

    Alternatively (and more robustly), we flag each segment with _precomputed=True and
    return the DE values directly as (window, 14_channels_after_mapping, 3_bands).
    The process_dataset function detects this flag and writes the windows straight to disk,
    bypassing the raw-EEG windowing step.

    Here we choose the robust path: return _precomputed windows.
    """
    import pickle

    data_dir = Path(config['data_paths']['FACED'])
    if not data_dir.exists():
        raise FileNotFoundError(f"FACED data not found: {data_dir}")

    # DE features live in FACED/EEG_Features/DE/  (inside the extracted zip)
    de_dir = data_dir / 'EEG_Features' / 'DE'
    if not de_dir.exists():
        # fallback: search for DE subdirectory anywhere under data_dir
        candidates = list(data_dir.glob('**/DE'))
        if candidates:
            de_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"FACED DE features not found under {data_dir}. "
                "Expected: <data_dir>/EEG_Features/DE/subXXX.pkl.pkl"
            )

    # FACED 32-channel names (standard 32-ch cap used in FACED, from paper appendix)
    faced_32ch = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
    ]

    # Band indices inside FACED DE (0-indexed)
    # FACED bands: delta=0, theta=1, alpha=2, beta=3, gamma=4
    THETA_IDX = 1
    ALPHA_IDX = 2
    BETA_IDX = 3

    # Compute 14-channel mapping indices once (from 32 FACED channels to 14 EPOC channels)
    from shared.feature_extraction import EPOC_CHANNELS
    # mapping_info values are matched source channel names; build integer index array
    # FACED 32ch contains all 14 EPOC channels directly (exact name matches)
    ch_lower = [c.lower() for c in faced_32ch]
    src_indices = np.array([
        ch_lower.index(epoc_ch.lower()) for epoc_ch in EPOC_CHANNELS
    ])  # shape (14,) — index into faced_32ch for each EPOC channel

    segments = []
    pkl_files = sorted(de_dir.glob('*.pkl.pkl'))
    if not pkl_files:
        pkl_files = sorted(de_dir.glob('*.pkl'))  # fallback

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                de = pickle.load(f)  # (28, 32, 30, 5)

            if de.ndim != 4:
                print(f"  Warning: unexpected DE shape {de.shape} in {pkl_file}, skipping")
                continue

            n_videos, n_ch, n_windows, n_bands = de.shape
            subj_name = pkl_file.stem.replace('.pkl', '')  # sub000

            # For each video clip and each 1-second window, build a pseudo-feature entry.
            # We store DE values for the 14 mapped channels x 3 bands as a (n_windows, 14, 3)
            # precomputed array flagged via _precomputed=True.
            de_14ch = de[:, src_indices, :, :]  # (28, 14, 30, 5)

            # Extract theta/alpha/beta → shape (28, 14, 30, 3)
            de_tab = np.stack(
                [de_14ch[:, :, :, THETA_IDX],
                 de_14ch[:, :, :, ALPHA_IDX],
                 de_14ch[:, :, :, BETA_IDX]],
                axis=-1
            )  # (28, 14, 30, 3)

            for vid_idx in range(n_videos):
                # de_tab[vid_idx]: (14, 30, 3)
                # Rearrange to (30 windows, 14 channels, 3 bands)
                win_feat = de_tab[vid_idx].transpose(1, 0, 2)  # (30, 14, 3)

                segments.append({
                    '_precomputed': True,         # signal to process_dataset to skip raw processing
                    '_win_feat': win_feat,         # (n_windows, 14, 3) DE values
                    'subject_id': f'FACED_{subj_name}',
                    'trial_id': vid_idx,
                    'dataset': 'FACED',
                })
        except Exception as e:
            print(f"  Warning: skip {pkl_file}: {e}")
            import traceback
            traceback.print_exc()

    return segments


def load_ds004572(config):
    """Load ds004572 (real hypnosis BIDS dataset, BrainVision format)."""
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

        # Recursively find all .vhdr (BrainVision) and .bdf files
        eeg_files = (
            sorted(sub_dir.glob('**/*.vhdr'))
            + sorted(sub_dir.glob('**/*.bdf'))
            + sorted(sub_dir.glob('**/*.fif'))
        )

        for eeg_file in eeg_files:
            try:
                if eeg_file.suffix == '.vhdr':
                    raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True, verbose=False)
                elif eeg_file.suffix == '.bdf':
                    raw = mne.io.read_raw_bdf(str(eeg_file), preload=True, verbose=False)
                else:
                    raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
            except Exception as e:
                print(f"  Warning: skip {eeg_file}: {e}")
                continue

            eeg_data = raw.get_data()
            orig_fs = int(raw.info['sfreq'])

            # Extract task name from filename for trial identification
            task_name = ''
            fname = eeg_file.stem
            for keyword in ['baseline', 'induction', 'experience', 'hypnosis']:
                if keyword in fname.lower():
                    task_name = keyword
                    break
            if not task_name:
                task_name = fname

            segments.append({
                'eeg': eeg_data,
                'fs': orig_fs,
                'dataset': 'ds004572',
                'subject_id': f'ds004572_{subj_name}',
                'trial_id': task_name,
                'channels': ds_channels,
            })

    return segments


def load_ds006437(config):
    """Load ds006437 (real hypnosis BIDS dataset, EEGLAB .set format).

    Session-aware loading: each BIDS session (ses-0, ses-1, ses-4, ses-8) is
    treated as a separate trial so that session-specific labels can be assigned
    in prep03. This avoids the previous position-based synthetic split.
    """
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

        # Iterate over BIDS sessions (ses-0, ses-1, ses-4, ses-8)
        session_dirs = sorted([d for d in sub_dir.glob('ses-*') if d.is_dir()])
        if not session_dirs:
            # Fallback: no session subdirs, treat all files as one trial
            session_dirs = [sub_dir]

        for session_dir in session_dirs:
            session_name = session_dir.name  # e.g. 'ses-0'

            eeg_files = (
                sorted(session_dir.glob('**/*.set'))
                + sorted(session_dir.glob('**/*.fif'))
                + sorted(session_dir.glob('**/*.bdf'))
                + sorted(session_dir.glob('**/*.vhdr'))
            )

            for eeg_file in eeg_files:
                try:
                    if eeg_file.suffix == '.set':
                        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
                    elif eeg_file.suffix == '.vhdr':
                        raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True, verbose=False)
                    elif eeg_file.suffix == '.bdf':
                        raw = mne.io.read_raw_bdf(str(eeg_file), preload=True, verbose=False)
                    else:
                        raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
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
                    'trial_id': session_name,
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
        precomputed_features: ndarray or None — (n_windows, 14, 3) DE features for FACED
    """
    logger.info(f"Loading {dataset_name}...")
    t0 = time.time()

    loader = DATASET_LOADERS[dataset_name]
    segments = loader(config)

    if len(segments) == 0:
        logger.warning(f"  {dataset_name}: No segments loaded. Skipping.")
        return [], [], [], None

    logger.info(f"  {dataset_name}: {len(segments)} segments loaded ({time.time() - t0:.1f}s)")

    fs_target = config['features']['fs_target']
    window_sec = config['features']['window_sec']
    step_sec = config['features']['step_sec']

    all_windows = []
    all_subject_ids = []
    all_trial_ids = []
    all_precomputed = []   # for FACED: collect (n_windows, 14, 3) DE arrays

    for i, seg in enumerate(segments):
        try:
            # ── Precomputed-feature fast path (e.g. FACED DE features) ──────────
            if seg.get('_precomputed', False):
                win_feat = seg['_win_feat']  # (n_windows, 14, 3) DE values
                # Placeholder raw-EEG windows (zeros) so save logic works uniformly
                win_len = int(window_sec * fs_target)
                n_pre_windows = win_feat.shape[0]
                for w_idx in range(n_pre_windows):
                    all_windows.append(np.zeros((win_len, 14), dtype=np.float32))
                    all_subject_ids.append(seg['subject_id'])
                    all_trial_ids.append(seg['trial_id'])
                all_precomputed.append(win_feat)  # accumulate DE features

                if (i + 1) % 10 == 0 or i == len(segments) - 1:
                    logger.info(f"  Processed {i + 1}/{len(segments)} segments, "
                                f"{len(all_windows)} windows so far")
                continue

            # ── Standard raw-EEG path ────────────────────────────────────────────
            eeg = seg['eeg']
            orig_fs = seg['fs']
            source_channels = seg['channels']

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

    # Merge precomputed features if any (FACED case)
    precomputed_features = None
    if all_precomputed:
        precomputed_features = np.concatenate(all_precomputed, axis=0)  # (total_windows, 14, 3)

    return all_windows, all_subject_ids, all_trial_ids, precomputed_features


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep01', str(PROJECT_ROOT / config['logs_dir']))

    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_name in DATASET_LOADERS:
        try:
            windows, subj_ids, trial_ids, precomputed_feats = process_dataset(dataset_name, config, logger)

            if len(windows) == 0:
                logger.warning(f"{dataset_name}: No windows. Check data path.")
                summary[dataset_name] = {'status': 'empty', 'n_windows': 0}
                continue

            # Pad/truncate windows to uniform length if needed
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
            save_kwargs = dict(
                windows=windows_array,
                subject_ids=np.array(subj_ids),
                trial_ids=np.array(trial_ids),
            )
            # Save precomputed DE features alongside windows (FACED only)
            if precomputed_feats is not None:
                save_kwargs['precomputed_de'] = precomputed_feats  # (n_windows, 14, 3)

            np.savez_compressed(save_path, **save_kwargs)

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
