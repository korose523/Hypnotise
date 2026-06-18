"""reprocess_ds004572.py — Re-run prep02/prep03 for ds004572.

prep01 windows already contain session/task-level trial_ids. This script
re-extracts 63-dim features and re-generates task-condition labels
(baseline=Awake, induction=Light, experience=Deep) so that features and labels
both carry trial_id for proper alignment.
"""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from shared.config_loader import load_config
from shared.feature_extraction import extract_features_window, FEAT_NAMES
from shared.label_mapping import LabelMapper
from scripts.prep03_generate_splits_lodo_loso import load_raw_labels_ds004572
from shared.logger import setup_logger


def run_prep02(config, logger):
    prep01_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep01_windows')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_path = prep01_dir / 'ds004572_windows.npz'
    data = np.load(windows_path, allow_pickle=True)
    windows = data['windows']
    subj_ids = data['subject_ids']
    trial_ids = data['trial_ids']
    data.close()

    fs = config['features']['fs_target']
    logger.info(f'prep02: extracting features from {len(windows)} ds004572 windows...')
    t0 = time.time()
    features = []
    valid_indices = []
    for i, win in enumerate(windows):
        try:
            feat = extract_features_window(win, fs)
            features.append(feat)
            valid_indices.append(i)
        except Exception as e:
            if i < 5:
                logger.warning(f'  window {i} failed: {e}')
            continue

    features = np.array(features, dtype=np.float32)
    valid_subj_ids = np.array([subj_ids[i] for i in valid_indices])
    valid_trial_ids = np.array([trial_ids[i] for i in valid_indices])

    out_path = out_dir / 'ds004572_features.npz'
    np.savez_compressed(out_path,
                        features=features,
                        subject_ids=valid_subj_ids,
                        trial_ids=valid_trial_ids,
                        feature_names=np.array(FEAT_NAMES))
    logger.info(f'prep02: saved {features.shape} to {out_path} ({time.time()-t0:.1f}s)')
    return out_path


def run_prep03(config, logger):
    prep02_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep02_features')
    out_dir = Path(PROJECT_ROOT / config['processed_dir'] / 'prep03_labels')
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = prep02_dir / 'ds004572_features.npz'
    feat_data = np.load(feat_path, allow_pickle=True)
    n_windows = feat_data['features'].shape[0]
    window_subj_ids = feat_data['subject_ids']
    window_trial_ids = feat_data['trial_ids']
    feat_data.close()

    label_df = load_raw_labels_ds004572(config)
    label_mapper = LabelMapper(config)

    raw_values = label_df['raw_value'].values
    mapped_labels, success_mask = label_mapper.map_labels('ds004572', raw_values)
    label_df['mapped_label'] = mapped_labels

    logger.info(f'prep03: {success_mask.sum()}/{len(raw_values)} task-condition labels mapped')

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

    out_path = out_dir / 'ds004572_labels.npz'
    np.savez_compressed(out_path,
                        labels=window_labels,
                        subject_ids=window_subj_ids,
                        trial_ids=window_trial_ids,
                        label_type='true_hypnosis_task_proxy')

    meta_df = pd.DataFrame({
        'subject': window_subj_ids,
        'trial': window_trial_ids,
        'label': window_labels,
    })
    meta_df.to_csv(out_dir / 'ds004572_meta.csv', index=False)

    logger.info(f'prep03: saved labels to {out_path}')
    return out_path


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('reprocess_ds004572', str(PROJECT_ROOT / config['logs_dir']))

    logger.info('=' * 60)
    logger.info('Reprocessing ds004572 features and labels')
    logger.info('=' * 60)

    run_prep02(config, logger)
    run_prep03(config, logger)
    logger.info('Done.')


if __name__ == '__main__':
    main()
