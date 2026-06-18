"""repair_subject_ids.py — Repair subject IDs in existing preprocessed files.

The advisor identified that MAHNOB/SEED/SEED_IV used session/trial-level
identifiers instead of real participant IDs, causing subject leakage in the
80/20 calibration split. This script repairs the subject_ids in the already
processed prep01/prep02/prep03 files without re-running the signal processing.

Repairs applied:
  - MAHNOB:  session -> real subject via mahnob_session_to_subject.json
  - SEED:    file_trial -> subject number (e.g. SEED_10_20131130_1 -> SEED_10)
  - SEED_IV: file_trial -> subject number (e.g. SEED_IV_1_10_20151014_de_movingAve1 -> SEED_IV_10)

For all repaired datasets, the trial_id is reset to a per-subject trial index.
Original files are backed up to *.bak before overwriting.

Run after any fix to the raw preprocessing code, but before generating splits.
"""
import os
import re
import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

PROCESSED_DIR = PROJECT_ROOT / 'processed'
WINDOWS_DIR = PROCESSED_DIR / 'prep01_windows'
FEATURES_DIR = PROCESSED_DIR / 'prep02_features'
LABELS_DIR = PROCESSED_DIR / 'prep03_labels'


def load_mahnob_session_map():
    with open(PROJECT_ROOT / 'mahnob_session_to_subject.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def repair_mahnob(session_map):
    """Map MAHNOB session-based subject IDs to real subject IDs."""
    for subdir in [WINDOWS_DIR, FEATURES_DIR, LABELS_DIR]:
        path = subdir / 'MAHNOB_windows.npz' if subdir == WINDOWS_DIR else subdir / 'MAHNOB_labels.npz' if subdir == LABELS_DIR else subdir / 'MAHNOB_features.npz'
        if not path.exists():
            print(f'  SKIP: {path} not found')
            continue

        backup = path.with_suffix('.npz.bak')
        if not backup.exists():
            shutil.copy2(path, backup)

        data = np.load(path, allow_pickle=True)
        subj_ids = data['subject_ids'].astype(str)
        trial_ids = data['trial_ids'].astype(str) if 'trial_ids' in data else np.zeros(len(subj_ids), dtype=str)

        new_subj_ids = []
        subj_trial_counters = defaultdict(int)
        new_trial_ids = []

        new_subj_ids = []
        subj_trial_counters = defaultdict(int)
        new_trial_ids = []

        for old_sid in subj_ids:
            session_num = old_sid.replace('MAHNOB_', '')
            real_subj = session_map.get(str(session_num), session_num)
            new_sid = f'MAHNOB_{real_subj}'
            new_subj_ids.append(new_sid)
            new_trial_ids.append(subj_trial_counters[new_sid])
            subj_trial_counters[new_sid] += 1

        new_subj_ids = np.array(new_subj_ids)
        new_trial_ids = np.array(new_trial_ids)

        save_dict = {k: data[k] for k in data.files}
        save_dict['subject_ids'] = new_subj_ids
        save_dict['trial_ids'] = new_trial_ids

        np.savez_compressed(path, **save_dict)
        data.close()

        n_subj = len(set(new_subj_ids))
        print(f'  MAHNOB {subdir.name}: {len(new_subj_ids)} windows -> {n_subj} real subjects')


def repair_seed():
    """Map SEED file+trial subject IDs to real subject IDs."""
    for subdir in [WINDOWS_DIR, FEATURES_DIR, LABELS_DIR]:
        fname = 'SEED_windows.npz' if subdir == WINDOWS_DIR else 'SEED_labels.npz' if subdir == LABELS_DIR else 'SEED_features.npz'
        path = subdir / fname
        if not path.exists():
            print(f'  SKIP: {path} not found')
            continue

        backup = path.with_suffix('.npz.bak')
        if not backup.exists():
            shutil.copy2(path, backup)

        data = np.load(path, allow_pickle=True)
        subj_ids = data['subject_ids'].astype(str)

        new_subj_ids = []
        subj_trial_counters = defaultdict(int)
        new_trial_ids = []

        for old_sid in subj_ids:
            # SEED_10_20131130_1 -> subject number 10
            parts = old_sid.split('_')
            subj_num = parts[1]
            new_sid = f'SEED_{subj_num}'
            new_subj_ids.append(new_sid)
            new_trial_ids.append(subj_trial_counters[new_sid])
            subj_trial_counters[new_sid] += 1

        new_subj_ids = np.array(new_subj_ids)
        new_trial_ids = np.array(new_trial_ids)

        save_dict = {k: data[k] for k in data.files}
        save_dict['subject_ids'] = new_subj_ids
        save_dict['trial_ids'] = new_trial_ids

        np.savez_compressed(path, **save_dict)
        data.close()

        n_subj = len(set(new_subj_ids))
        print(f'  SEED {subdir.name}: {len(new_subj_ids)} windows -> {n_subj} real subjects')


def repair_seed_iv():
    """Map SEED-IV file+trial subject IDs to real subject IDs."""
    for subdir in [WINDOWS_DIR, FEATURES_DIR, LABELS_DIR]:
        fname = 'SEED_IV_windows.npz' if subdir == WINDOWS_DIR else 'SEED_IV_labels.npz' if subdir == LABELS_DIR else 'SEED_IV_features.npz'
        path = subdir / fname
        if not path.exists():
            print(f'  SKIP: {path} not found')
            continue

        backup = path.with_suffix('.npz.bak')
        if not backup.exists():
            shutil.copy2(path, backup)

        data = np.load(path, allow_pickle=True)
        subj_ids = data['subject_ids'].astype(str)

        new_subj_ids = []
        subj_trial_counters = defaultdict(int)
        new_trial_ids = []

        for old_sid in subj_ids:
            # SEED_IV_1_10_20151014_de_movingAve1 -> subject number 10
            parts = old_sid.split('_')
            # parts[0]=SEED, [1]=IV, [2]=session, [3]=subject, [4]=date, [5]=de_movingAve, [6]=trial
            if len(parts) >= 4:
                subj_num = parts[3]
            else:
                subj_num = 'unknown'
            new_sid = f'SEED_IV_{subj_num}'
            new_subj_ids.append(new_sid)
            new_trial_ids.append(subj_trial_counters[new_sid])
            subj_trial_counters[new_sid] += 1

        new_subj_ids = np.array(new_subj_ids)
        new_trial_ids = np.array(new_trial_ids)

        save_dict = {k: data[k] for k in data.files}
        save_dict['subject_ids'] = new_subj_ids
        save_dict['trial_ids'] = new_trial_ids

        np.savez_compressed(path, **save_dict)
        data.close()

        n_subj = len(set(new_subj_ids))
        print(f'  SEED_IV {subdir.name}: {len(new_subj_ids)} windows -> {n_subj} real subjects')


def verify():
    """Print final unique-subject counts for repaired datasets."""
    print('\n--- Verification ---')
    for ds in ['MAHNOB', 'SEED', 'SEED_IV']:
        fpath = FEATURES_DIR / f'{ds}_features.npz'
        if not fpath.exists():
            continue
        d = np.load(fpath, allow_pickle=True)
        n_unique = len(set(d['subject_ids']))
        print(f'{ds}: {len(d["features"])} windows, {n_unique} unique subject IDs')
        d.close()


def main():
    print('Repairing subject IDs in preprocessed files...')
    print('Backups created as *.npz.bak')
    print()

    print('MAHNOB:')
    session_map = load_mahnob_session_map()
    repair_mahnob(session_map)

    print('\nSEED:')
    repair_seed()

    print('\nSEED_IV:')
    repair_seed_iv()

    verify()
    print('\nDone. Run prep04 / exp101 to regenerate splits with real subject IDs.')


if __name__ == '__main__':
    main()
