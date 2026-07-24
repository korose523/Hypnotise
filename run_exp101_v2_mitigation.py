#!/usr/bin/env python3
"""
run_exp101_v2_mitigation.py — Mitigation experiments for label collapse and FACED exclusion.

Three modes:
  1. exclude_faced: Run LODO without FACED (7-source variant)
  2. smote: Run LODO with SMOTE oversampling on source domains
  3. all: Both exclude FACED AND apply SMOTE

Output: results/exp101_v2_mitigation/multi_{mode}_ds.json
"""
import sys, os, json, time, warnings, copy
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager, ALL_DATASETS
from shared.logger import setup_logger

# =============================================================================
# Parameters
# =============================================================================
MAX_SRC = 8000
MAX_TGT = 8000
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 5
N_JOBS = -1
ALL_SEEDS = [42, 123, 456, 789, 2024]  # 5 seeds for quick ablation

# Try to import SMOTE
HAS_SMOTE = False
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    pass


def load_dataset(ds_name, prep_dir, max_n=None):
    feat_path = prep_dir / 'prep02_features' / f'{ds_name}_features.npz'
    label_path = prep_dir / 'prep03_labels' / f'{ds_name}_labels.npz'
    if not feat_path.exists() or not label_path.exists():
        raise FileNotFoundError(f'Missing prep data for {ds_name}')
    f = np.load(feat_path, allow_pickle=True)
    l = np.load(label_path, allow_pickle=True)
    X = f['features'].astype(np.float32)
    y = l['labels'].astype(np.int32)
    sids = l['subject_ids']
    f.close(); l.close()
    valid = y >= 0
    X, y, sids = X[valid], y[valid], sids[valid]
    if max_n is not None and len(X) > max_n:
        rng = np.random.RandomState(42)
        groups = np.array([hash(str(s)) % 100000 for s in sids])
        ug = sorted(set(groups))
        n_per = max(1, max_n // len(ug))
        idx = []
        for g in ug:
            gi = np.where(groups == g)[0]
            ni = min(n_per, len(gi))
            idx.extend(rng.choice(gi, ni, replace=False))
        idx = np.array(idx)
        if len(idx) > max_n:
            idx = rng.choice(idx, max_n, replace=False)
        X, y, sids = X[idx], y[idx], sids[idx]
    return X, y, sids


def build_groups(ds_name, subject_ids):
    return np.array([hash(str(s)) % 100000 for s in subject_ids])


def apply_smote(X, y, seed, logger):
    """Apply SMOTE oversampling if available, else return original."""
    if not HAS_SMOTE:
        logger.info('  SMOTE not available (install imbalanced-learn)')
        return X, y
    class_counts = Counter(y)
    if len(set(class_counts.values())) <= 1:
        return X, y  # already balanced
    try:
        sm = SMOTE(random_state=seed, k_neighbors=min(3, min(class_counts.values()) - 1))
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(f'  SMOTE: {dict(class_counts)} -> {dict(Counter(y_res))}')
        return X_res, y_res
    except Exception as e:
        logger.warning(f'  SMOTE failed: {e}')
        return X, y


def run_single_experiment(target, source_domains, seed, prep_dir, sm, logger, use_smote=False):
    X_src_list, y_src_list = [], []
    for src in source_domains:
        Xs, ys, _ = load_dataset(src, prep_dir, max_n=MAX_SRC)
        X_src_list.append(Xs)
        y_src_list.append(ys)
    X_src = np.vstack(X_src_list)
    y_src = np.concatenate(y_src_list)

    X_tgt, y_tgt, tgt_sids = load_dataset(target, prep_dir, max_n=MAX_TGT)
    tgt_groups = build_groups(target, tgt_sids)
    ugs = sorted(set(tgt_groups))
    split = sm.load_subject_split(target, seed)
    calib_subjs = set(str(s) for s in split.get('calib_subjects', []))
    test_subjs = set(str(s) for s in split.get('test_subjects', []))
    calib_mask = np.array([str(s) in calib_subjs for s in tgt_sids])
    test_mask = np.array([str(s) in test_subjs for s in tgt_sids])

    if calib_mask.sum() < 2 or test_mask.sum() < 2:
        logger.warning(f'[{target} s={seed}] SKIP')
        return None

    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)

    # Zero-shot
    if use_smote:
        X_train_zs, y_train_zs = apply_smote(X_src_s, y_src, seed, logger)
    else:
        X_train_zs, y_train_zs = X_src_s, y_src

    rf_zs = RandomForestClassifier(n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF, class_weight='balanced',
        n_jobs=N_JOBS, random_state=seed)
    rf_zs.fit(X_train_zs, y_train_zs)
    pred_zs = rf_zs.predict(X_tgt_s[test_mask])
    acc_zs = accuracy_score(y_tgt[test_mask], pred_zs)
    f1_zs = f1_score(y_tgt[test_mask], pred_zs, average='macro', zero_division=0)
    cm_zs = confusion_matrix(y_tgt[test_mask], pred_zs, labels=[0, 1, 2]).tolist()
    recall_zs = recall_score(y_tgt[test_mask], pred_zs, labels=[0, 1, 2], average=None, zero_division=0).tolist()

    # Calibration
    X_calib = X_tgt_s[calib_mask]
    y_calib = y_tgt[calib_mask]
    X_train_cal = np.vstack([X_train_zs, X_calib])
    y_train_cal = np.concatenate([y_train_zs, y_calib])

    rf_cal = RandomForestClassifier(n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF, class_weight='balanced',
        n_jobs=N_JOBS, random_state=seed)
    rf_cal.fit(X_train_cal, y_train_cal)
    pred_cal = rf_cal.predict(X_tgt_s[test_mask])
    acc_cal = accuracy_score(y_tgt[test_mask], pred_cal)
    f1_cal = f1_score(y_tgt[test_mask], pred_cal, average='macro', zero_division=0)
    cm_cal = confusion_matrix(y_tgt[test_mask], pred_cal, labels=[0, 1, 2]).tolist()
    recall_cal = recall_score(y_tgt[test_mask], pred_cal, labels=[0, 1, 2], average=None, zero_division=0).tolist()

    return {
        'target': target, 'seed': seed,
        'n_src': int(len(X_src)), 'n_calib': int(calib_mask.sum()), 'n_test': int(test_mask.sum()),
        'zs_acc': round(acc_zs, 4), 'zs_f1': round(f1_zs, 4),
        'calib_acc': round(acc_cal, 4), 'calib_f1': round(f1_cal, 4),
        'zs_cm': cm_zs, 'calib_cm': cm_cal,
        'zs_per_class_recall': recall_zs, 'calib_per_class_recall': recall_cal,
        'n_groups': len(ugs), 'n_calib_groups': len(calib_subjs),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mitigation experiments')
    parser.add_argument('--mode', choices=['exclude_faced', 'smote', 'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='*', default=ALL_SEEDS)
    args = parser.parse_args()

    mode = args.mode
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger(f'exp101_v2_{mode}', str(PROJECT_ROOT / config['logs_dir']))

    # Datasets
    if mode in ('exclude_faced', 'all'):
        # Exclude FACED completely
        datasets = [d for d in ALL_DATASETS if d != 'FACED']
    else:
        datasets = ALL_DATASETS[:]

    use_smote = mode in ('smote', 'all')

    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    splits_dir = Path(PROJECT_ROOT / config['splits_dir'])
    sm = SplitManager(str(splits_dir))

    out_dir = PROJECT_ROOT / 'results' / 'exp101_v2_mitigation'
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f'exp101_v2: mode={mode}, smote={use_smote}, datasets={datasets}, seeds={args.seeds}')
    logger.info('=' * 60)

    all_results = []
    t_start = time.time()

    for target in datasets:
        source_domains = [d for d in datasets if d != target]
        logger.info(f'\n--- Target: {target} (sources: {source_domains}) ---')

        for seed in args.seeds:
            try:
                rec = run_single_experiment(target, source_domains, seed, prep_dir, sm, logger, use_smote)
                if rec is not None:
                    all_results.append(rec)
                    logger.info(f'[{target} s={seed}] ZS={rec["zs_acc"]:.4f} Calib={rec["calib_acc"]:.4f}')
            except Exception as e:
                logger.error(f'[{target} s={seed}] FAILED: {e}')
                import traceback; traceback.print_exc()
                continue

    out_path = out_dir / f'multi_{mode}_ds.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    logger.info(f'\nDone. Total records: {len(all_results)} / {len(datasets) * len(args.seeds)}')
    logger.info(f'Results saved to: {out_path}')
    logger.info(f'Elapsed: {time.time() - t_start:.1f}s')

    # Print summary
    results_by_target = {}
    for r in all_results:
        tgt = r['target']
        if tgt not in results_by_target:
            results_by_target[tgt] = {'zs': [], 'cal': []}
        results_by_target[tgt]['zs'].append(r['zs_acc'])
        results_by_target[tgt]['cal'].append(r['calib_acc'])

    print(f'\n=== Summary: {mode} mode ({len(args.seeds)} seeds) ===')
    all_zs, all_cal = [], []
    for tgt in sorted(results_by_target.keys()):
        zs = np.array(results_by_target[tgt]['zs'])
        ca = np.array(results_by_target[tgt]['cal'])
        all_zs.extend(zs)
        all_cal.extend(ca)
        print(f'{tgt:12s}: ZS={zs.mean()*100:.2f}±{zs.std()*100:.2f}% Calib={ca.mean()*100:.2f}±{ca.std()*100:.2f}%')
    print(f'{"Overall":12s}: ZS={np.mean(all_zs)*100:.2f}±{np.std(all_zs)*100:.2f}% Calib={np.mean(all_cal)*100:.2f}±{np.std(all_cal)*100:.2f}%')


if __name__ == '__main__':
    main()
