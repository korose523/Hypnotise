"""run_exp101_reproducible.py — Single reproducible script for multi_8ds.json.

This is the unified, single script that reproduces the main LODO results:
  - 8 target domains (Leave-One-Domain-Out)
  - 20 seeds per target
  - Zero-shot vs 20% target-subject calibration
  - Subsample all domains to MAX_SRC / MAX_TGT windows for fair comparison
  - Outputs confusion matrix, per-class recall, macro-F1, accuracy

Output: results/exp101_lodo_loso/multi_8ds.json
"""
import sys
import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager, ALL_DATASETS
from shared.logger import setup_logger


# =============================================================================
# Reproducibility parameters (fixed and documented)
# =============================================================================
MAX_SRC = 8000          # max windows per source domain (matching paper Table 2)
MAX_TGT = 8000          # max windows for target evaluation
N_ESTIMATORS = 200      # RF trees, matching paper §3.3
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHT = 'balanced'
N_JOBS = -1
ALL_SEEDS = [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555,
             6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789]

OUT_PATH = PROJECT_ROOT / 'results' / 'exp101_lodo_loso' / 'multi_8ds.json'


def stable_hash(s):
    """Deterministic hash (built-in hash() is salted by PYTHONHASHSEED and
    breaks cross-process reproducibility of the group-aware subsampling)."""
    return int(hashlib.md5(str(s).encode('utf-8')).hexdigest(), 16) % 100000


def load_dataset(ds_name, prep_dir, max_n=None):
    """Load features and labels for one dataset, optionally subsample."""
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
        # Group-aware subsampling: keep subject groups intact
        rng = np.random.RandomState(42)
        groups = np.array([stable_hash(s) for s in sids])
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
    """Build numeric group IDs for split-manager alignment."""
    # For most datasets, subject_id is already the real participant ID.
    # Just hash it to a stable integer.
    return np.array([stable_hash(s) for s in subject_ids])


def run_single_experiment(target, source_domains, seed, prep_dir, sm, logger):
    """Run one LODO fold: zero-shot + 20% calibration."""
    # Load source data (all other domains)
    X_src_list, y_src_list = [], []
    for src in source_domains:
        Xs, ys, _ = load_dataset(src, prep_dir, max_n=MAX_SRC)
        X_src_list.append(Xs)
        y_src_list.append(ys)
    X_src = np.vstack(X_src_list)
    y_src = np.concatenate(y_src_list)

    # Load target data
    X_tgt, y_tgt, tgt_sids = load_dataset(target, prep_dir, max_n=MAX_TGT)
    tgt_groups = build_groups(target, tgt_sids)
    ugs = sorted(set(tgt_groups))
    n_grp = len(ugs)

    # Load the inner LOSO split for this target/seed
    split = sm.load_subject_split(target, seed)
    calib_subjs = set(str(s) for s in split.get('calib_subjects', []))
    test_subjs = set(str(s) for s in split.get('test_subjects', []))

    calib_mask = np.array([str(s) in calib_subjs for s in tgt_sids])
    test_mask = np.array([str(s) in test_subjs for s in tgt_sids])

    if calib_mask.sum() < 2 or test_mask.sum() < 2:
        logger.warning(f'[{target} s={seed}] SKIP: calib={calib_mask.sum()} test={test_mask.sum()}')
        return None

    # Standardize: fit on source, transform target
    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)

    # Zero-shot: train on source only, test on target test set
    rf_zs = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
        random_state=seed
    )
    rf_zs.fit(X_src_s, y_src)
    pred_zs = rf_zs.predict(X_tgt_s[test_mask])
    acc_zs = accuracy_score(y_tgt[test_mask], pred_zs)
    f1_zs = f1_score(y_tgt[test_mask], pred_zs, average='macro', zero_division=0)
    cm_zs = confusion_matrix(y_tgt[test_mask], pred_zs, labels=[0, 1, 2]).tolist()
    recall_zs = recall_score(y_tgt[test_mask], pred_zs, labels=[0, 1, 2], average=None, zero_division=0).tolist()

    # Calibration: append 20% target calibration subjects to source, retrain
    X_calib = X_tgt_s[calib_mask]
    y_calib = y_tgt[calib_mask]
    X_train_cal = np.vstack([X_src_s, X_calib])
    y_train_cal = np.concatenate([y_src, y_calib])

    rf_cal = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
        random_state=seed
    )
    rf_cal.fit(X_train_cal, y_train_cal)
    pred_cal = rf_cal.predict(X_tgt_s[test_mask])
    acc_cal = accuracy_score(y_tgt[test_mask], pred_cal)
    f1_cal = f1_score(y_tgt[test_mask], pred_cal, average='macro', zero_division=0)
    cm_cal = confusion_matrix(y_tgt[test_mask], pred_cal, labels=[0, 1, 2]).tolist()
    recall_cal = recall_score(y_tgt[test_mask], pred_cal, labels=[0, 1, 2], average=None, zero_division=0).tolist()

    n_calib_groups = len(calib_subjs)

    rec = {
        'target': target,
        'seed': seed,
        'n_src': int(len(X_src)),
        'n_calib': int(calib_mask.sum()),
        'n_test': int(test_mask.sum()),
        'zs_acc': round(acc_zs, 4),
        'zs_f1': round(f1_zs, 4),
        'calib_acc': round(acc_cal, 4),
        'calib_f1': round(f1_cal, 4),
        'zs_cm': cm_zs,
        'calib_cm': cm_cal,
        'zs_per_class_recall': recall_zs,
        'calib_per_class_recall': recall_cal,
        'n_groups': n_grp,
        'n_calib_groups': n_calib_groups,
    }

    logger.info(f'[{target} s={seed}] ZS={acc_zs:.4f} Calib={acc_cal:.4f} n_calib={calib_mask.sum()} n_test={test_mask.sum()}')
    return rec


CHECKPOINT_PATH = PROJECT_ROOT / 'results' / 'exp101_lodo_loso' / 'multi_8ds_checkpoint.json'


def load_checkpoint():
    """Load existing checkpoint if available."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_checkpoint(results):
    """Save intermediate checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp101_repro', str(PROJECT_ROOT / config['logs_dir']))

    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    splits_dir = Path(PROJECT_ROOT / config['splits_dir'])
    sm = SplitManager(str(splits_dir))
    lodo_splits = sm.load_lodo_splits()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info('exp101 reproducible run: generating multi_8ds.json')
    logger.info(f'Parameters: MAX_SRC={MAX_SRC}, MAX_TGT={MAX_TGT}, n_estimators={N_ESTIMATORS}, seeds={len(ALL_SEEDS)}')
    logger.info('=' * 60)

    all_results = load_checkpoint()
    completed = {(r['target'], r['seed']) for r in all_results}
    logger.info(f'Resumed {len(all_results)} records from checkpoint')

    t_start = time.time()

    for target in ALL_DATASETS:
        source_domains = [d for d in ALL_DATASETS if d != target]
        logger.info(f'\n--- Target: {target} (sources: {source_domains}) ---')

        for seed in ALL_SEEDS:
            if (target, seed) in completed:
                logger.info(f'[{target} s={seed}] SKIP (already in checkpoint)')
                continue
            try:
                rec = run_single_experiment(target, source_domains, seed, prep_dir, sm, logger)
                if rec is not None:
                    all_results.append(rec)
                    completed.add((target, seed))
                    save_checkpoint(all_results)
            except Exception as e:
                logger.error(f'[{target} s={seed}] FAILED: {e}')
                import traceback
                traceback.print_exc()
                continue

    # Save final results
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)

    # Also save to master
    master_path = PROJECT_ROOT / 'multi_8ds_master.json'
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)

    # Clean up checkpoint on successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    logger.info('\n' + '=' * 60)
    logger.info(f'Done. Total records: {len(all_results)} / {len(ALL_DATASETS) * len(ALL_SEEDS)}')
    logger.info(f'Elapsed: {time.time() - t_start:.1f}s')
    logger.info(f'Results saved to: {OUT_PATH}')
    logger.info(f'Master saved to: {master_path}')


if __name__ == '__main__':
    main()
