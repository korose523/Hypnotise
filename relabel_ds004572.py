"""relabel_ds004572.py — Re-generate ds004572 labels with task-condition mapping.

OpenNeuro ds004572 does not expose numeric hypnosis depth scores in the
 downloadable BIDS structure used here. This script assigns labels from the
task-condition filename (baseline/induction/experience), matching prep01's
trial_id.
"""
import sys
import os
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from shared.config_loader import load_config
from shared.label_mapping import LabelMapper
from scripts.prep03_generate_splits_lodo_loso import load_raw_labels_ds004572
from shared.logger import setup_logger


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('relabel_ds004572', str(PROJECT_ROOT / config['logs_dir']))

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

    logger.info(f'{success_mask.sum()}/{len(raw_values)} task-condition labels mapped')

    trial_label_map = {}
    for _, row in label_df.iterrows():
        key = (str(row['subject_id']), str(row['trial_id']))
        trial_label_map[key] = row['mapped_label'] if 'mapped_label' in row else -1

    window_labels = np.full(n_windows, -1, dtype=int)
    for i in range(n_windows):
        key = (str(window_subj_ids[i]), str(window_trial_ids[i]))
        window_labels[i] = trial_label_map.get(key, -1)

    n_valid = int(np.sum(window_labels >= 0))
    logger.info(f'aligned {n_valid}/{n_windows} windows to labels')
    class_dist = label_mapper.get_class_distribution(window_labels)
    logger.info(f'class distribution {class_dist}')

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

    logger.info(f'saved labels to {out_path}')


if __name__ == '__main__':
    main()
