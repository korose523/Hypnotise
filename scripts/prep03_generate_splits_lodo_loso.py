#!/usr/bin/env python3
"""
prep03_make_3class_labels_and_stats.py — Generate 3-class hypnosis depth labels.

This script:
  1. Loads raw label information from each dataset
  2. Applies dataset-specific mapping rules to produce:
       0 = Awake (清醒), 1 = Light Hypnosis (浅催眠), 2 = Deep Hypnosis (深催眠)
  3. Aligns labels with window-level features from prep02
  4. Computes class distribution statistics
  5. Saves labels as .npz files to processed/prep03_labels/

IMPORTANT distinction:
  - ds004572 / ds006437: TRUE hypnosis labels (depth scores, protocol phases)
  - DREAMER/DEAP/MAHNOB: PROXY labels via arousal dimension
  - SEED/SEED_IV:        PROXY labels via emotion dimension
  - FACED:               PROXY labels via arousal dimension

Input:  Raw data (for label extraction) + prep02 feature alignment info
Output: processed/prep03_labels/{dataset}_labels.npz
        Each .npz contains: labels (n_windows,), subject_ids, trial_ids,
                            label_type, class_distribution dict
"""

import sys
import numpy as np
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.label_mapping import LabelMapper
from shared.logger import setup_logger


# ============================================================================
# Dataset-specific label loaders
# ============================================================================

def load_raw_labels_dreamer(config):
    """Load DREAMER arousal ratings per trial/subject."""
    import scipy.io as sio

    data_path = Path(config['data_paths']['DREAMER'])
    mat = sio.loadmat(str(data_path))
    data = mat['DREAMER'][0, 0]
    arous_labels = data['Arousal'][0, 0]  # (n_subjects, n_trials)

    rows = []
    n_subjects = arous_labels.shape[0]
    n_trials = arous_labels.shape[1]

    for subj in range(n_subjects):
        for trial in range(n_trials):
            rows.append({
                'subject_id': f'DREAMER_S{subj + 1:02d}',
                'trial_id': trial,
                'raw_value': float(arous_labels[subj, trial]),
            })

    return pd.DataFrame(rows)


def load_raw_labels_deap(config):
    """Load DEAP arousal ratings per trial/subject."""
    import pickle

    data_dir = Path(config['data_paths']['DEAP'])
    rows = []

    dat_files = sorted(data_dir.glob('s*.dat'))
    for dat_file in dat_files:
        subj_idx = int(dat_file.stem[1:]) - 1
        with open(dat_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        labels = data['labels']
        for trial in range(labels.shape[0]):
            rows.append({
                'subject_id': f'DEAP_S{subj_idx + 1:02d}',
                'trial_id': trial,
                'raw_value': float(labels[trial, 1]),  # arousal
            })

    return pd.DataFrame(rows)


def load_raw_labels_mahnob(config):
    """Load MAHNOB arousal ratings."""
    import h5py

    data_dir = Path(config['data_paths']['MAHNOB'])
    rows = []

    session_dirs = sorted(data_dir.glob('Sessions/*'))
    for session_dir in session_dirs:
        subj_match = session_dir.name
        h5_files = list(session_dir.glob('*.hdf5')) + list(session_dir.glob('*.h5'))
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    arousal = float(f['arousal'][()]) if 'arousal' in f else 5.0
                rows.append({
                    'subject_id': f'MAHNOB_{subj_match}',
                    'trial_id': 0,
                    'raw_value': arousal,
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


def load_raw_labels_seed(config):
    """Load SEED emotion labels (positive=1, neutral=0, negative=-1)."""
    data_dir = Path(config['data_paths']['SEED'])
    rows = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            labels = mat.get('labels', None) or mat.get('label', None)
            if labels is not None:
                labels_flat = labels.flatten()
                for i in range(len(labels_flat)):
                    rows.append({
                        'subject_id': f'SEED_{mat_file.stem}',
                        'trial_id': i,
                        'raw_value': int(labels_flat[i]),
                    })
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_raw_labels_seed_iv(config):
    """Load SEED-IV emotion labels (4-class: 0,1,2,3)."""
    data_dir = Path(config['data_paths']['SEED_IV'])
    rows = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            labels = mat.get('label', None)
            if labels is not None:
                labels_flat = labels.flatten()
                for i in range(len(labels_flat)):
                    rows.append({
                        'subject_id': f'SEED_IV_{mat_file.parent.name}_{mat_file.stem}',
                        'trial_id': i,
                        'raw_value': int(labels_flat[i]),
                    })
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_raw_labels_faced(config):
    """Load FACED arousal ratings."""
    data_dir = Path(config['data_paths']['FACED'])
    rows = []

    import scipy.io as sio
    for mat_file in sorted(data_dir.glob('**/*.mat')):
        try:
            mat = sio.loadmat(str(mat_file))
            arousal = mat.get('arousal', None)
            if arousal is not None:
                arousal_flat = arousal.flatten()
                for i in range(len(arousal_flat)):
                    rows.append({
                        'subject_id': f'FACED_{mat_file.parent.name}',
                        'trial_id': i,
                        'raw_value': float(arousal_flat[i]),
                    })
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_raw_labels_ds004572(config):
    """Load ds004572 hypnosis depth scores (true labels)."""
    data_dir = Path(config['data_paths']['ds004572'])
    rows = []

    import csv
    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        tsv_files = list(sub_dir.glob('**/*events*.tsv')) + \
                    list(sub_dir.glob('**/*depth*.tsv'))
        if tsv_files:
            for tsv_file in tsv_files:
                try:
                    with open(tsv_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            depth = row.get('depth', row.get('hypnosis_depth', '5'))
                            rows.append({
                                'subject_id': f'ds004572_{subj_name}',
                                'trial_id': 0,
                                'raw_value': float(depth),
                            })
                except Exception:
                    continue
        else:
            rows.append({
                'subject_id': f'ds004572_{subj_name}',
                'trial_id': 0,
                'raw_value': 5.0,
            })

    return pd.DataFrame(rows)


def load_raw_labels_ds006437(config):
    """Load ds006437 protocol phase labels (true labels)."""
    data_dir = Path(config['data_paths']['ds006437'])
    rows = []

    import csv
    sub_dirs = sorted([d for d in data_dir.glob('sub-*') if d.is_dir()])
    for sub_dir in sub_dirs:
        subj_name = sub_dir.name
        tsv_files = list(sub_dir.glob('**/*events*.tsv'))
        if tsv_files:
            for tsv_file in tsv_files:
                try:
                    with open(tsv_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            phase = row.get('phase', row.get('trigger', 'pre'))
                            rows.append({
                                'subject_id': f'ds006437_{subj_name}',
                                'trial_id': 0,
                                'raw_value': phase,
                            })
                except Exception:
                    continue
        else:
            rows.append({
                'subject_id': f'ds006437_{subj_name}',
                'trial_id': 0,
                'raw_value': 'pre',
            })

    return pd.DataFrame(rows)


# ============================================================================
# Label loaders registry
# ============================================================================

LABEL_LOADERS = {
    'DREAMER': load_raw_labels_dreamer,
    'DEAP': load_raw_labels_deap,
    'MAHNOB': load_raw_labels_mahnob,
    'SEED': load_raw_labels_seed,
    'SEED_IV': load_raw_labels_seed_iv,
    'FACED': load_raw_labels_faced,
    'ds006437': load_raw_labels_ds006437,
}

LABEL_TYPES = {
    'DREAMER': 'proxy_arousal',
    'DEAP': 'proxy_arousal',
    'MAHNOB': 'proxy_arousal',
    'SEED': 'proxy_emotion',
    'SEED_IV': 'proxy_emotion',
    'FACED': 'proxy_arousal',
    'ds006437': 'true_hypnosis',
}


def align_labels_to_features(feat_path, label_df, dataset_name, logger):
    """
    Align trial-level labels to window-level features from prep02.

    Each window from a trial inherits its trial's label.
    """
    feat_data = np.load(feat_path, allow_pickle=True)
    n_windows = feat_data['features'].shape[0]
    window_subj_ids = feat_data['subject_ids']
    window_trial_ids = feat_data['trial_ids']

    # Build trial -> label lookup
    trial_label_map = {}
    for _, row in label_df.iterrows():
        key = (str(row['subject_id']), int(row['trial_id']))
        trial_label_map[key] = row['mapped_label'] if 'mapped_label' in row else -1

    window_labels = np.full(n_windows, -1, dtype=int)
    for i in range(n_windows):
        key = (str(window_subj_ids[i]), int(window_trial_ids[i]))
        window_labels[i] = trial_label_map.get(key, -1)

    n_valid = int(np.sum(window_labels >= 0))
    logger.info(f"  Aligned {n_valid}/{n_windows} windows to labels")

    return window_labels, window_subj_ids, window_trial_ids


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('prep03', str(PROJECT_ROOT / config['logs_dir']))

    label_mapper = LabelMapper(config)

    prep02_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep03_labels')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_name in LABEL_LOADERS:
        try:
            logger.info(f"Processing labels for {dataset_name}...")

            # Load raw labels
            label_df = LABEL_LOADERS[dataset_name](config)
            if len(label_df) == 0:
                logger.warning(f"  {dataset_name}: No raw labels loaded.")
                summary[dataset_name] = {'status': 'empty'}
                continue

            # Map to 3-class
            raw_values = label_df['raw_value'].values
            mapped_labels, success_mask = label_mapper.map_labels(dataset_name, raw_values)
            label_df['mapped_label'] = mapped_labels

            n_mapped = int(np.sum(success_mask))
            logger.info(f"  {dataset_name}: {n_mapped}/{len(raw_values)} labels mapped")

            # Align to prep02 features
            feat_path = prep02_dir / f'{dataset_name}_features.npz'
            if feat_path.exists():
                window_labels, window_subj_ids, window_trial_ids = \
                    align_labels_to_features(feat_path, label_df, dataset_name, logger)
                source = 'aligned_to_prep02'
            else:
                # Fallback: use trial-level labels directly
                valid = success_mask & (mapped_labels >= 0)
                window_labels = mapped_labels[valid]
                window_subj_ids = label_df['subject_id'].values[valid]
                window_trial_ids = label_df['trial_id'].values[valid]
                source = 'trial_level'

            # Class distribution
            class_dist = label_mapper.get_class_distribution(window_labels)
            logger.info(f"  Class distribution: {class_dist}")

            # Save
            save_path = out_dir / f'{dataset_name}_labels.npz'
            np.savez_compressed(
                save_path,
                labels=window_labels,
                subject_ids=window_subj_ids,
                trial_ids=window_trial_ids,
                label_type=LABEL_TYPES[dataset_name],
            )

            # Also save as CSV with metadata for easy loading
            meta_df = pd.DataFrame({
                'subject': window_subj_ids,
                'trial': window_trial_ids,
                'label': window_labels,
            })
            meta_path = out_dir / f'{dataset_name}_meta.csv'
            meta_df.to_csv(meta_path, index=False)

            summary[dataset_name] = {
                'status': 'ok',
                'n_labels': len(window_labels),
                'n_valid': int(np.sum(window_labels >= 0)),
                'label_type': LABEL_TYPES[dataset_name],
                'source': source,
                'class_distribution': class_dist,
                'save_path': str(save_path),
                'meta_path': str(meta_path),
            }

            logger.info(f"{dataset_name}: Saved {len(window_labels)} labels to {save_path}")

        except Exception as e:
            logger.error(f"{dataset_name}: Failed — {e}")
            import traceback
            traceback.print_exc()
            summary[dataset_name] = {'status': 'error', 'error': str(e)}

    # Save summary
    summary_path = out_dir / 'prep03_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("prep03 complete. Summary:")
    for ds, info in summary.items():
        if isinstance(info, dict):
            logger.info(f"  {ds}: {info.get('status', '?')} — "
                        f"type={info.get('label_type', '?')}, "
                        f"n={info.get('n_labels', 0)}, "
                        f"distribution={info.get('class_distribution', {})}")


if __name__ == '__main__':
    main()
