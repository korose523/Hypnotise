"""reprocess_ds006437.py — Re-process ds006437 with session-aware loading.

This is a targeted re-run of prep01/prep02/prep03 for the ds006437 dataset
after fixing the prep01 loader to respect BIDS sessions and the prep03 loader
to assign session-based labels (ses-0=Awake, ses-1=Light, ses-4/8=Deep).
"""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from shared.config_loader import load_config
from shared.feature_extraction import extract_features_window, FEAT_NAMES
from shared.label_mapping import LabelMapper
from scripts.prep01_build_63feat_all_datasets import load_ds006437, process_dataset
from shared.logger import setup_logger


def run_prep01(config, logger):
    """Load raw ds006437 EEG and create windowed segments."""
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('prep01: loading ds006437...')
    segments = load_ds006437(config)
    windows, subj_ids, trial_ids, _ = process_dataset('ds006437', config, logger)

    if len(windows) == 0:
        raise RuntimeError('No windows generated for ds006437')

    fs_target = config['features']['fs_target']
    window_sec = config['features']['window_sec']
    target_len = int(window_sec * fs_target)
    padded = []
    for w in windows:
        if w.shape[0] >= target_len:
            padded.append(w[:target_len])
        else:
            pad = np.zeros((target_len - w.shape[0], w.shape[1]), dtype=w.dtype)
            padded.append(np.vstack([w, pad]))
    windows_array = np.array(padded, dtype=np.float32)

    save_path = out_dir / 'ds006437_windows.npz'
    np.savez_compressed(save_path,
                        windows=windows_array,
                        subject_ids=np.array(subj_ids),
                        trial_ids=np.array(trial_ids))
    logger.info(f'prep01: saved {windows_array.shape} to {save_path}')
    return save_path


def run_prep02(config, logger):
    """Extract 63-dimensional features from prep01 windows."""
    prep01_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_path = prep01_dir / 'ds006437_windows.npz'
    data = np.load(windows_path, allow_pickle=True)
    windows = data['windows']  # (n, samples, 14)
    subj_ids = data['subject_ids']
    trial_ids = data['trial_ids']
    data.close()

    logger.info(f'prep02: extracting features from {len(windows)} windows...')
    t0 = time.time()
    features = []
    for i, win in enumerate(windows):
        feat = extract_features_window(win, config['features']['fs_target'])
        features.append(feat)
        if (i + 1) % 5000 == 0:
            logger.info(f'  processed {i + 1}/{len(windows)} windows')

    features = np.array(features, dtype=np.float32)
    out_path = out_dir / 'ds006437_features.npz'
    np.savez_compressed(out_path,
                        features=features,
                        subject_ids=subj_ids,
                        trial_ids=trial_ids,
                        feature_names=FEAT_NAMES)
    logger.info(f'prep02: saved {features.shape} to {out_path} ({time.time()-t0:.1f}s)')
    return out_path


def run_prep03(config, logger):
    """Generate session-aware 3-class labels for ds006437."""
    from scripts.prep03_generate_splits_lodo_loso import load_raw_labels_ds006437

    prep02_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep03_labels')
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = prep02_dir / 'ds006437_features.npz'
    feat_data = np.load(feat_path, allow_pickle=True)
    n_windows = feat_data['features'].shape[0]
    window_subj_ids = feat_data['subject_ids']
    window_trial_ids = feat_data['trial_ids']
    feat_data.close()

    label_df = load_raw_labels_ds006437(config)
    label_mapper = LabelMapper(config)

    raw_values = label_df['raw_value'].values
    mapped_labels, success_mask = label_mapper.map_labels('ds006437', raw_values)
    label_df['mapped_label'] = mapped_labels

    logger.info(f'prep03: {success_mask.sum()}/{len(raw_values)} session labels mapped')

    # Build trial -> label lookup
    trial_label_map = {}
    for _, row in label_df.iterrows():
        key = (str(row['subject_id']), str(row['trial_id']))
        trial_label_map[key] = row['mapped_label'] if 'mapped_label' in row else -1

    window_labels = np.full(n_windows, -1, dtype=int)
    for i in range(n_windows):
        key = (str(window_subj_ids[i]), str(window_trial_ids[i]))
        window_labels[i] = trial_label_map.get(key, -1)

    n_valid = int(np.sum(window_labels >= 0))
    logger.info(f'prep03: aligned {n_valid}/{n_windows} windows to labels')

    class_dist = label_mapper.get_class_distribution(window_labels)
    logger.info(f'prep03: class distribution {class_dist}')

    out_path = out_dir / 'ds006437_labels.npz'
    np.savez_compressed(out_path,
                        labels=window_labels,
                        subject_ids=window_subj_ids,
                        trial_ids=window_trial_ids,
                        label_type='true_hypnosis_session_proxy')

    # Also save meta CSV
    import pandas as pd
    meta_df = pd.DataFrame({
        'subject': window_subj_ids,
        'trial': window_trial_ids,
        'label': window_labels,
    })
    meta_df.to_csv(out_dir / 'ds006437_meta.csv', index=False)

    logger.info(f'prep03: saved labels to {out_path}')
    return out_path


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('reprocess_ds006437', str(PROJECT_ROOT / config['logs_dir']))

    logger.info('=' * 60)
    logger.info('Reprocessing ds006437 with session-aware labels')
    logger.info('=' * 60)

    run_prep01(config, logger)
    run_prep02(config, logger)
    run_prep03(config, logger)

    logger.info('Done.')


if __name__ == '__main__':
    main()
