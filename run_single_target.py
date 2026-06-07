"""run_single_target.py — Run all missing seeds for ONE target only.

Usage: python run_single_target.py <target_name>
Example: python run_single_target.py MAHNOB

Reads existing results from multi_8ds.json, skips completed seeds,
runs missing seeds, saves incrementally after each seed.
"""
import numpy as np, os, json, time, sys, warnings, re
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter

DATASETS = ['DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV', 'FACED', 'ds006437', 'ds004572']
MAX_SRC = 8000
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 5
SEEDS = [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555,
         6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789]
RESULT_PATH = 'results/exp101_lodo_loso/multi_8ds.json'


def build_groups(ds_name, subject_ids, all_ok):
    groups = []
    for sid in subject_ids:
        sid_str = str(sid)
        if ds_name == 'MAHNOB':
            mapping_path = 'mahnob_session_to_subject.json'
            if not hasattr(build_groups, '_mahnob_map'):
                with open(mapping_path, 'r') as f:
                    build_groups._mahnob_map = json.load(f)
            mahnob_map = build_groups._mahnob_map
            m = re.search(r'(\d+)', sid_str)
            sess_id = m.group(1) if m else sid_str
            groups.append(mahnob_map.get(str(sess_id), hash(sid_str) % 10000))
        elif ds_name == 'SEED':
            parts = sid_str.split('_')
            groups.append(int(parts[1]) if len(parts) > 1 else 0)
        elif ds_name == 'SEED_IV':
            parts = sid_str.split('_')
            groups.append(int(parts[3]) if len(parts) > 3 else 0)
        else:
            groups.append(hash(sid_str) % 10000)
    return np.array(groups)


def safe_subsample(X, y, sids, groups, max_n):
    if len(X) <= max_n:
        return X, y, sids, groups
    rng = np.random.RandomState(42)
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
    return X[idx], y[idx], sids[idx], groups[idx]


def run_target(target):
    if target not in DATASETS:
        print('ERROR: target must be one of', DATASETS)
        sys.exit(1)

    # Load all datasets into memory
    print('Loading datasets...')
    X, y, groups = {}, {}, {}
    for ds in DATASETS:
        f = np.load('processed/prep02_features/%s_features.npz' % ds, allow_pickle=True)
        l = np.load('processed/prep03_labels/%s_labels.npz' % ds, allow_pickle=True)
        Xd = f['features'].astype(np.float32)
        yd = l['labels'].astype(np.int32)
        sd = f['subject_ids']
        v = yd >= 0
        Xd, yd, sd = Xd[v], yd[v], sd[v]
        grp = build_groups(ds, sd, True)
        Xd, yd, sd, grp = safe_subsample(Xd, yd, sd, grp, MAX_SRC)
        X[ds] = Xd; y[ds] = yd; groups[ds] = grp
        f.close(); l.close()
        print('  %-12s: %d win, %d groups' % (ds, len(yd), len(set(grp))))

    # Load existing results
    results = []
    done = set()
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, 'r') as f:
            results = json.load(f)
        done = set((r['target'], r['seed']) for r in results)
        print('\nLoaded %d existing experiments' % len(results))

    src = [d for d in DATASETS if d != target]
    Xs = np.concatenate([X[d] for d in src])
    ys = np.concatenate([y[d] for d in src])
    Xt, yt, gt = X[target], y[target], groups[target]
    ugs = sorted(set(gt))
    n_grp = len(ugs)

    print('\nTarget=%s: src=%d tgt=%d groups=%d' % (target, len(ys), len(yt), n_grp))

    missing = [s for s in SEEDS if (target, s) not in done]
    if not missing:
        print('All seeds already completed for %s!' % target)
        return

    print('Missing seeds: %s' % missing)
    t0 = time.time()

    for seed in missing:
        if n_grp < 2:
            print('  s=%-4d: SKIP (<2 groups)' % seed)
            continue

        rng = np.random.RandomState(seed)
        perm = rng.permutation(ugs)
        nc = max(1, int(n_grp * 0.2))
        cs = set(perm[:nc]); ts = set(perm[nc:])
        cm = np.array([g in cs for g in gt])
        tm = np.array([g in ts for g in gt])

        if cm.sum() < 2 or tm.sum() < 2:
            print('  s=%-4d: SKIP (cal=%d test=%d)' % (seed, cm.sum(), tm.sum()))
            continue

        scaler = StandardScaler()
        Xs_s = scaler.fit_transform(Xs)
        Xt_s = scaler.transform(Xt)

        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
                                    class_weight='balanced', n_jobs=-1, random_state=seed)
        rf.fit(Xs_s, ys)
        zp = rf.predict(Xt_s[tm])
        za = accuracy_score(yt[tm], zp)
        zf = f1_score(yt[tm], zp, average='macro', zero_division=0)
        zcm = confusion_matrix(yt[tm], zp, labels=[0, 1, 2])

        rf2 = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
                                     class_weight='balanced', n_jobs=-1, random_state=seed)
        rf2.fit(np.vstack([Xs_s, Xt_s[cm]]), np.concatenate([ys, yt[cm]]))
        wp = rf2.predict(Xt_s[tm])
        wa = accuracy_score(yt[tm], wp)
        wf = f1_score(yt[tm], wp, average='macro', zero_division=0)
        wcm = confusion_matrix(yt[tm], wp, labels=[0, 1, 2])

        results.append({
            'target': target, 'seed': seed,
            'n_src': len(ys), 'n_calib': int(cm.sum()), 'n_test': int(tm.sum()),
            'zs_acc': round(za, 4), 'zs_f1': round(zf, 4),
            'calib_acc': round(wa, 4), 'calib_f1': round(wf, 4),
            'zs_cm': zcm.tolist(), 'calib_cm': wcm.tolist(),
            'n_groups': n_grp, 'n_calib_groups': nc,
        })
        print('  s=%-4d: ZS=%.4f  Calib=%.4f  (cal=%d test=%d)  [%ds]' %
              (seed, za, wa, cm.sum(), tm.sum(), time.time() - t0))

        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        with open(RESULT_PATH, 'w') as f:
            json.dump(results, f, indent=1)

    print('\nTarget %s complete. Total time: %.0fs' % (target, time.time() - t0))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_single_target.py <target_name>')
        print('Targets:', DATASETS)
        sys.exit(1)
    run_target(sys.argv[1])
