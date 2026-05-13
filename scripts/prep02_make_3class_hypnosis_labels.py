#!/usr/bin/env python3
"""
prep02_make_3class_hypnosis_labels.py — Generate 3-class hypnosis depth labels for all datasets.

This script:
  1. Loads raw label information from each dataset
  2. Applies dataset-specific mapping rules (from config.yaml) to produce:
       0 = Awake (清醒)
       1 = Light Hypnosis (浅催眠)
       2 = Deep Hypnosis (深催眠)
  3. Aligns labels with the window-level features from prep01
  4. Saves labels as .npz files to processed/prep02_labels/

IMPORTANT distinction:
  - ds004572 / ds006437: TRUE hypnosis labels (depth scores, protocol phases)
  - DREAMER/DEAP/MAHNOB: PROXY labels via arousal dimension
  - SEED/SEED_IV:        PROXY labels via emotion dimension
  - FACED:               PROXY labels via arousal dimension

Input:  Raw data (for label extraction) + prep01 feature alignment info
Output: processed/prep02_labels/{dataset}_labels.npz
        Each .npz contains: labels (n_windows,), subject_ids, trial_ids,
                            label_type ('true_hypnosis' or 'proxy_*'),
                            class_distribution dict
"""

import sys
import numpy as np
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.label_mapping import LabelMapper
from shared.logger import setup_logger


def load_raw_labels_dreamer(config):
    """Load DREAMER arousal ratings per trial/subject."""
    import scipy.io as sio

    data_path = Path(config['data_paths']['DREAMER'])
    mat = sio.loadmat(str(data_path))
    data = mat['DREAMER'][0, 0]
    arous_labels = data['Arousal'][0, 0]  # (n_subjects, n_trials)

    labels_per_window = []
    subject_ids = []
    trial_ids = []

    n_subjects = arous_labels.shape[0]
    n_trials = arous_labels.shape[1]

    for subj in range(n_subjects):
        for trial in range(n_trials):
            labels_per_window.append(float(arous_labels[subj, trial]))
            subject_ids.append(f'DREAMER_S{subj + 1:02d}')
            trial_ids.append(trial)

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_deap(config):
    """Load DEAP arousal ratings per trial/subject."""
    import pickle

    data_dir = Path(config['data_paths']['DEAP'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    dat_files = sorted(data_dir.glob('s*.dat'))
    for dat_file in dat_files:
        subj_idx = int(dat_file.stem[1:]) - 1
        with open(dat_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        labels = data['labels']
        for trial in range(labels.shape[0]):
            labels_per_window.append(float(labels[trial, 1]))  # arousal
            subject_ids.append(f'DEAP_S{subj_idx + 1:02d}')
            trial_ids.append(trial)

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_mahnob(config):
    """Load MAHNOB arousal ratings."""
    import h5py

    data_dir = Path(config['data_paths']['MAHNOB'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    session_dirs = sorted(data_dir.glob('Sessions/*'))
    for session_dir in session_dirs:
        subj_match = session_dir.name
        h5_files = list(session_dir.glob('*.hdf5')) + list(session_dir.glob('*.h5'))
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    arousal = float(f['arousal'][()]) if 'arousal' in f else 5.0
                labels_per_window.append(arousal)
                subject_ids.append(f'MAHNOB_{subj_match}')
                trial_ids.append(0)
            except Exception:
                continue

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_seed(config):
    """Load SEED emotion labels (positive=1, neutral=0, negative=-1)."""
    data_dir = Path(config['data_paths']['SEED'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    import scipy.io as sio
    # SEED has 3 sessions, each with 15 trials = 45 trials per subject
    # Emotion labels: positive(1), neutral(0), negative(-1) per session
    # This is a simplified loader — adapt to actual directory structure
    label_files = sorted(data_dir.glob('**/*label*.mat'))
    for lf in label_files:
        try:
            mat = sio.loadmat(str(lf))
            labels = mat.get('labels', None) or mat.get('label', None)
            if labels is not None:
                for i in range(len(labels)):
                    labels_per_window.append(int(labels.flatten()[i]))
                    subject_ids.append(f'SEED_{lf.parent.stem if lf.parent.stem else lf.stem}')
                    trial_ids.append(i)
        except Exception:
            continue

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_seed_iv(config):
    """Load SEED-IV emotion labels (4-class: 0,1,2,3)."""
    data_dir = Path(config['data_paths']['SEED_IV'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            labels = mat.get('label', None)
            if labels is not None:
                labels_flat = labels.flatten()
                for i in range(len(labels_flat)):
                    labels_per_window.append(int(labels_flat[i]))
                    subject_ids.append(
                        f'SEED_IV_{mat_file.parent.name}_{mat_file.stem}'
                    )
                    trial_ids.append(i)
        except Exception:
            continue

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_faced(config):
    """Load FACED arousal ratings."""
    data_dir = Path(config['data_paths']['FACED'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            arousal = mat.get('arousal', None)
            if arousal is not None:
                arousal_flat = arousal.flatten()
                for i in range(len(arousal_flat)):
                    labels_per_window.append(float(arousal_flat[i]))
                    subject_ids.append(f'FACED_{mat_file.parent.name}')
                    trial_ids.append(i)
        except Exception:
            continue

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_ds004572(config):
    """Load ds004572 hypnosis depth scores (true labels)."""
    data_dir = Path(config['data_paths']['ds004572'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    try:
        import mne
    except ImportError:
        print("  Warning: MNE not installed. Skipping ds004572 labels.")
        return np.array([]), [], []

    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        # Look for events/annotations with depth scores
        tsv_files = list(sub_dir.glob('**/*events*.tsv')) + \
                    list(sub_dir.glob('**/*depth*.tsv'))
        if tsv_files:
            import csv
            for tsv_file in tsv_files:
                try:
                    with open(tsv_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            depth = row.get('depth', row.get('hypnosis_depth', '5'))
                            labels_per_window.append(float(depth))
                            subject_ids.append(f'ds004572_{subj_name}')
                            trial_ids.append(0)
                except Exception:
                    continue
        else:
            # No depth TSV found — use placeholder
            labels_per_window.append(5.0)
            subject_ids.append(f'ds004572_{subj_name}')
            trial_ids.append(0)

    return np.array(labels_per_window), subject_ids, trial_ids


def load_raw_labels_ds006437(config):
    """Load ds006437 protocol phase labels (true labels)."""
    data_dir = Path(config['data_paths']['ds006437'])
    labels_per_window = []
    subject_ids = []
    trial_ids = []

    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        tsv_files = list(sub_dir.glob('**/*events*.tsv'))
        if tsv_files:
            import csv
            for tsv_file in tsv_files:
                try:
                    with open(tsv_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            phase = row.get('phase', row.get('trigger', 'pre'))
                            labels_per_window.append(phase)
                            subject_ids.append(f'ds006437_{subj_name}')
                            trial_ids.append(0)
                except Exception:
                    continue
        else:
            labels_per_window.append('pre')
            subject_ids.append(f'ds006437_{subj_name}')
            trial_ids.append(0)

    return np.array(labels_per_window), subject_ids, trial_ids


# ============================================================================
# Dataset-specific label loaders registry
# ============================================================================

LABEL_LOADERS = {
    'DREAMER': load_raw_labels_dreamer,
    'DEAP': load_raw_labels_deap,
    'MAHNOB': load_raw_labels_mahnob,
    'SEED': load_raw_labels_seed,
    'SEED_IV': load_raw_labels_seed_iv,
    'FACED': load_raw_labels_faced,
    'ds004572': load_raw_labels_ds004572,
    'ds006437': load_raw_labels_ds006437,
}

LABEL_TYPES = {
    'DREAMER': 'proxy_arousal',
    'DEAP': 'proxy_arousal',
    'MAHNOB': 'proxy_arousal',
    'SEED': 'proxy_emotion',
    'SEED_IV': 'proxy_emotion',
    'FACED': 'proxy_arousal',
    'ds004572': 'true_hypnosis',
    'ds006437': 'true_hypnosis',
}


def process_dataset_labels(dataset_name, config, label_mapper, logger):
    """
    Load raw labels and map to 3-class hypnosis depth.

    Args:
        dataset_name: str
        config: dict
        label_mapper: LabelMapper instance
        logger: logging.Logger

    Returns:
        labels: ndarray (n_windows,) of int (0, 1, 2)
        subject_ids: list of str
        trial_ids: list of int
        class_dist: dict
    """
    logger.info(f"Processing labels for {dataset_name}...")

    loader = LABEL_LOADERS[dataset_name]
    raw_labels, subj_ids, trial_ids = loader(config)

    if len(raw_labels) == 0:
        logger.warning(f"  {dataset_name}: No raw labels loaded.")
        return np.array([], dtype=int), [], [], {}

    # Map raw labels to 3-class
    mapped_labels, success_mask = label_mapper.map_labels(dataset_name, raw_labels)

    # Filter unmapped labels
    valid = success_mask & (mapped_labels >= 0)
    mapped_labels = mapped_labels[valid]
    subj_ids = [s for s, v in zip(subj_ids, valid) if v]
    trial_ids = [t for t, v in zip(trial_ids, valid) if v]

    # Class distribution
    class_dist = label_mapper.get_class_distribution(mapped_labels)

    n_total = len(raw_labels)
    n_mapped = int(np.sum(valid))
    logger.info(f"  {dataset_name}: {n_mapped}/{n_total} labels mapped")
    logger.info(f"  Class distribution: {class_dist}")

    return mapped_labels, subj_ids, trial_ids, class_dist


def align_labels_to_features(prep01_path, trial_labels, trial_subj_ids, trial_trial_ids, logger):
    """
    Align trial-level labels to window-level features from prep01.

    Each window from a trial inherits its trial's label.

    Args:
        prep01_path: Path to the prep01 .npz file
        trial_labels: ndarray, trial-level labels
        trial_subj_ids: list, trial-level subject IDs
        trial_trial_ids: list, trial-level trial IDs
        logger: logging.Logger

    Returns:
        window_labels: ndarray (n_windows,)
    """
    data = np.load(prep01_path, allow_pickle=True)
    n_windows = data['features'].shape[0]
    window_subj_ids = data['subject_ids']
    window_trial_ids = data['trial_ids']

    # Build trial -> label lookup
    trial_label_map = {}
    for s, t, l in zip(trial_subj_ids, trial_trial_ids, trial_labels):
        trial_label_map[(str(s), int(t))] = int(l)

    window_labels = np.full(n_windows, -1, dtype=int)
    for i in range(n_windows):
        key = (str(window_subj_ids[i]), int(window_trial_ids[i]))
        window_labels[i] = trial_label_map.get(key, -1)

    n_valid = np.sum(window_labels >= 0)
    logger.info(f"  Aligned {n_valid}/{n_windows} windows to labels "
                f"({n_windows - n_valid} unmatched)")

    return window_labels


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep02', str(PROJECT_ROOT / config['logs_dir']))

    label_mapper = LabelMapper(config)

    prep01_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_features')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_labels')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_name in LABEL_LOADERS:
        try:
            mapped_labels, subj_ids, trial_ids, class_dist = \
                process_dataset_labels(dataset_name, config, label_mapper, logger)

            if len(mapped_labels) == 0:
                summary[dataset_name] = {'status': 'empty'}
                continue

            # Check if prep01 features exist for alignment
            prep01_path = prep01_dir / f'{dataset_name}_features.npz'
            if prep01_path.exists():
                window_labels = align_labels_to_features(
                    prep01_path, mapped_labels, subj_ids, trial_ids, logger
                )
                save_labels = window_labels
                source = 'aligned_to_prep01'
            else:
                save_labels = mapped_labels
                source = 'trial_level'

            # Save
            save_path = out_dir / f'{dataset_name}_labels.npz'
            np.savez_compressed(
                save_path,
                labels=save_labels,
                subject_ids=np.array(subj_ids[:len(save_labels)]),
                trial_ids=np.array(trial_ids[:len(save_labels)]),
                label_type=LABEL_TYPES[dataset_name],
            )

            summary[dataset_name] = {
                'status': 'ok',
                'n_labels': len(save_labels),
                'n_valid': int(np.sum(save_labels >= 0)),
                'label_type': LABEL_TYPES[dataset_name],
                'source': source,
                'class_distribution': class_dist,
                'save_path': str(save_path),
            }

            logger.info(f"{dataset_name}: Saved {save_labels.shape} labels to {save_path}")

        except Exception as e:
            logger.error(f"{dataset_name}: Failed — {e}")
            import traceback
            traceback.print_exc()
            summary[dataset_name] = {'status': 'error', 'error': str(e)}

    # Save summary
    summary_path = out_dir / 'prep02_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("prep02 complete. Summary:")
    for ds, info in summary.items():
        logger.info(f"  {ds}: {info.get('status', '?')} — "
                     f"type={info.get('label_type', '?')}, "
                     f"n={info.get('n_labels', 0)}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
