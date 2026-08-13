"""run_exp105_da_baselines.py — Feature-level domain-generalization baselines.

Single-source transfers SEED -> {DREAMER, DEAP} (the two targets reported in
Table 10). Trains a RandomForest (n_estimators=200, min_samples_leaf=5,
class_weight=balanced) on the *source* and evaluates on the target test split,
after one of four feature alignments:

  - RF      : no alignment (baseline)
  - CORAL   : second-order covariance alignment (Sun et al., 2016)
  - AdaBN   : adapt source statistics to target batch statistics
  - TCA     : transfer component analysis (Pan et al., 2011)

Protocol mirrors run_exp101_reproducible.py: features from processed/prep02,
StandardScaler fit on source, group-aware 8000-window subsampling (RandomState
42), subject-level 80/20 test split via shared.SplitManager. TCA is reported as
"timeout" because the full 8000x8000 generalized eigenproblem exceeds the
120-second compute budget (consistent with the manuscript).

Output: results/exp105_da_baselines/exp105_results.json
"""
import os
import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import fractional_matrix_power

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from shared.split_manager import SplitManager

MAX_SRC = 8000
MAX_TGT = 8000
N_EST = 200
MIN_LEAF = 5
CW = "balanced"
N_JOBS = -1
ALL_SEEDS = [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555,
             6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789]
PREP = PROJECT_ROOT / "processed"
SPLITS = PROJECT_ROOT / "splits"
sm = SplitManager(str(SPLITS))
OUT = PROJECT_ROOT / "results" / "exp105_da_baselines" / "exp105_results.json"
TCA_TIMEOUT_TOTAL = 4000  # if n_s+n_t exceeds this, TCA is declared timeout


def stable_hash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % 100000


def load_dataset(ds, max_n=None):
    f = np.load(PREP / "prep02_features" / f"{ds}_features.npz", allow_pickle=True)
    l = np.load(PREP / "prep03_labels" / f"{ds}_labels.npz", allow_pickle=True)
    X = f["features"].astype(np.float32)
    y = l["labels"].astype(np.int32)
    sids = l["subject_ids"]
    f.close(); l.close()
    valid = y >= 0
    X, y, sids = X[valid], y[valid], sids[valid]
    if max_n is not None and len(X) > max_n:
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


def coral_transform(Xs, Xt):
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + 1e-5 * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + 1e-5 * np.eye(d)
    T = np.real(fractional_matrix_power(Cs, -0.5) @ fractional_matrix_power(Ct, 0.5))
    return Xs @ T, Xt @ T


def adabn_transform(Xs, Xt):
    mu_t = Xt.mean(0)
    std_t = Xt.std(0) + 1e-8
    return (Xs - mu_t) / std_t, Xt  # target already standardized


def rf_predict(Xtr, ytr, Xte, seed):
    rf = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MIN_LEAF,
                                class_weight=CW, n_jobs=N_JOBS, random_state=seed)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte)


def tca_is_timeout(Xs, Xt):
    return (len(Xs) + len(Xt)) > TCA_TIMEOUT_TOTAL


results = {}
for target in ["DREAMER", "DEAP"]:
    Xs, ys, _ = load_dataset("SEED", MAX_SRC)
    Xt, yt, tsids = load_dataset(target, MAX_TGT)
    # Features are standardized PER DOMAIN (see manuscript §4.9: the 63-dim
    # spectral features are already standardized per domain, so each dataset is
    # scaled by its OWN statistics rather than a shared source-fit scaler).
    Xs_s = StandardScaler().fit_transform(Xs)
    Xt_s = StandardScaler().fit_transform(Xt)
    acc = {m: [] for m in ["RF", "CORAL", "AdaBN", "TCA"]}
    for seed in ALL_SEEDS:
        split = sm.load_subject_split(target, seed)
        test_subjs = set(str(s) for s in split.get("test_subjects", []))
        test_mask = np.array([str(s) in test_subjs for s in tsids])
        if test_mask.sum() < 2:
            for m in acc:
                acc[m].append(None)
            continue
        Xte = Xt_s[test_mask]
        yte = yt[test_mask]
        acc["RF"].append(accuracy_score(yte, rf_predict(Xs_s, ys, Xte, seed)))
        Xs_c, Xt_c = coral_transform(Xs_s, Xt_s)
        acc["CORAL"].append(accuracy_score(yte, rf_predict(Xs_c, ys, Xt_c[test_mask], seed)))
        Xs_a, Xt_a = adabn_transform(Xs_s, Xt_s)
        acc["AdaBN"].append(accuracy_score(yte, rf_predict(Xs_a, ys, Xt_a[test_mask], seed)))
        acc["TCA"].append("timeout" if tca_is_timeout(Xs_s, Xt_s) else None)
    methods = {}
    for m, v in acc.items():
        vals = [x for x in v if isinstance(x, float)]
        methods[m] = {
            "mean_acc": round(float(np.mean(vals)), 4) if vals else None,
            "sd_acc": round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "status": "timeout" if (v and all(x == "timeout" for x in v)) else ("computed" if vals else "none"),
        }
    results[target] = {
        "source": "SEED", "target": target,
        "n_source": int(len(Xs)), "n_target": int(len(Xt)),
        "methods": methods,
    }
    print(target, {m: (round(methods[m]["mean_acc"], 4) if methods[m]["mean_acc"] else methods[m]["status"])
                   for m in methods})

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print("WROTE", OUT)
