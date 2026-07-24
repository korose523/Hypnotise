#!/usr/bin/env python3
"""
analyze_shap_rf.py — SHAP feature importance for the RF-based classifier.

For each dataset, trains a representative RandomForest on the full feature set
and uses TreeSHAP to estimate per-feature contributions. Saves a JSON summary
and a CSV of top features per dataset.

Output: results/shap_rf/shap_summary.json
        results/shap_rf/top_features.csv
        results/shap_rf/shap_values_{dataset}.npz  (optional, if save_values=True)
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.ensemble import RandomForestClassifier
from shared.config_loader import load_config
from shared.logger import setup_logger


def load_data(dataset_name, prep_dir):
    feat_path = prep_dir / 'prep02_features' / f'{dataset_name}_features.npz'
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'
    if not feat_path.exists() or not label_path.exists():
        return None, None, None

    fd = np.load(feat_path, allow_pickle=True)
    ld = np.load(label_path, allow_pickle=True)
    features = fd['features']
    labels = ld['labels']
    feat_names = fd.get('feature_names', None)
    if feat_names is None:
        feat_names = [f'feat_{i}' for i in range(features.shape[1])]
    valid = labels >= 0
    return features[valid], labels[valid], list(feat_names)


def main():
    if not HAS_SHAP:
        print('SHAP not installed. Run: pip install shap')
        return

    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('shap_rf', str(PROJECT_ROOT / config['logs_dir']))
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'shap_rf')
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV', 'FACED', 'ds006437', 'ds004572']
    # Lightweight RF for SHAP diagnostic (speed matters more than accuracy here)
    rf_params = {
        'n_estimators': 50,
        'max_depth': 15,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'n_jobs': 2,
        'random_state': 42,
    }

    summary = {}
    top_features_rows = []

    for dataset in datasets:
        X, y, feat_names = load_data(dataset, prep_dir)
        if X is None or len(X) < 50:
            logger.warning(f'Skipping {dataset}: insufficient data')
            continue

        logger.info(f'[{dataset}] Training RF on {len(X)} samples...')
        # Subsample for speed if too large
        if len(X) > 20000:
            idx = np.random.choice(len(X), 20000, replace=False)
            X_train, y_train = X[idx], y[idx]
        else:
            X_train, y_train = X, y

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        logger.info(f'[{dataset}] Computing SHAP values...')
        explainer = shap.TreeExplainer(rf)
        # Use a small background sample for SHAP; approximate for multi-class speed
        n_bg = min(100, len(X_train))
        shap_values = explainer.shap_values(
            X_train[:n_bg],
            approximate=True,
            check_additivity=False,
        )

        # shap_values can be list (multi-class) or ndarray (binary/multi-class)
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(np.asarray(sv)).mean(axis=0) for sv in shap_values], axis=0)
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                # (n_classes, n_samples, n_features) -> per-feature mean
                mean_shap = np.abs(sv).mean(axis=(0, 1))
            else:
                mean_shap = np.abs(sv).mean(axis=0)

        # Ensure 1-D array of feature importances
        mean_shap = np.asarray(mean_shap).ravel()

        # Rank features
        ranked = np.argsort(mean_shap)[::-1]
        top = []
        for rank_pos, i in enumerate(ranked[:20], start=1):
            idx = int(np.asarray(i).item())
            top.append({
                'feature': feat_names[idx],
                'mean_abs_shap': float(mean_shap[idx]),
                'rank': rank_pos,
            })
            top_features_rows.append({
                'dataset': dataset,
                'feature': feat_names[idx],
                'mean_abs_shap': float(mean_shap[idx]),
                'rank': rank_pos,
            })

        summary[dataset] = {
            'n_samples': int(len(X)),
            'n_features': int(len(feat_names)),
            'top_20': top,
        }
        logger.info(f'[{dataset}] Top 3 features: {", ".join(t["feature"] for t in top[:3])}')

        # Save per-dataset SHAP values if small
        if n_bg <= 200:
            np.savez_compressed(
                out_dir / f'shap_values_{dataset}.npz',
                shap_values=shap_values if isinstance(shap_values, np.ndarray) else np.array(shap_values),
                background=X_train[:n_bg],
                feature_names=np.array(feat_names)
            )

    with open(out_dir / 'shap_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(top_features_rows).to_csv(out_dir / 'top_features.csv', index=False)
    logger.info(f'Summary saved to {out_dir}')
    logger.info('SHAP analysis complete.')


if __name__ == '__main__':
    main()
