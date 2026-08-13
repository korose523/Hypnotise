#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-domain upper-bound experiment (Advisor §9).
For each of the 8 datasets, train a RandomForest on the WITHIN-dataset
calibration split and test on the held-out (LOSO / group) split, using the
PRE-COMPUTED partitions in splits/loso_<DS>_seed<SEED>.json and the processed
features/labels in processed/. Same 63-dim features and same 20 seeds as the
cross-dataset LODO run. Reports Acc / BAcc / Cohen's kappa / per-class recall,
averaged over 20 seeds, as the within-domain upper bound.

RF config matches paper Table 2: n_estimators=200, min_samples_leaf=5,
class_weight='balanced', n_jobs=-1.

Output: results/exp101_within_domain/within_domain_results.json
"""
import json, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, recall_score

BASE = "E:/universal_bci_hypnosis"
PROC = os.path.join(BASE, "processed")
SPLITS = os.path.join(BASE, "splits")
OUT = os.path.join(BASE, "results", "exp101_within_domain")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["DEAP", "DREAMER", "MAHNOB", "SEED", "SEED_IV", "FACED", "ds004572", "ds006437"]
SEEDS = [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555,
          6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789]


def load(ds):
    ft = np.load(os.path.join(PROC, "prep02_features", f"{ds}_features.npz"), allow_pickle=True)
    lb = np.load(os.path.join(PROC, "prep03_labels", f"{ds}_labels.npz"), allow_pickle=True)
    X = ft["features"].astype(np.float64)
    y = lb["labels"].astype(int)
    # IDs live in features.npz for ALL datasets; some labels.npz lack trial_ids.
    sub = np.array(ft["subject_ids"]).astype(str)
    tri = np.array(ft["trial_ids"]).astype(str)
    return X, y, sub, tri


results = {}
for ds in DATASETS:
    X, y, sub, tri = load(ds)
    N = len(y)
    per_seed = []
    for seed in SEEDS:
        sp = os.path.join(SPLITS, f"loso_{ds}_seed{seed}.json")
        if not os.path.exists(sp):
            print("MISSING", sp)
            continue
        d = json.load(open(sp))
        cal = set(d.get("calib_subjects", []))
        te = set(d.get("test_subjects", []))
        # a window belongs to a split unit if its subject_id OR trial_id is in the set
        cal_idx = np.isin(sub, list(cal)) | np.isin(tri, list(cal))
        te_idx = np.isin(sub, list(te)) | np.isin(tri, list(te))
        if cal_idx.sum() == 0 or te_idx.sum() == 0:
            print(f"  {ds} seed{seed}: EMPTY split (cal={cal_idx.sum()} te={te_idx.sum()})")
            continue
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
        rf.fit(X[cal_idx], y[cal_idx])
        pred = rf.predict(X[te_idx])
        yt = y[te_idx]
        acc = float((pred == yt).mean())
        bacc = float(balanced_accuracy_score(yt, pred))
        kappa = float(cohen_kappa_score(yt, pred))
        rec = recall_score(yt, pred, labels=[0, 1, 2], average=None, zero_division=0)
        per_seed.append(dict(seed=seed, acc=acc, bacc=bacc, kappa=kappa,
                             recall=[float(x) for x in rec],
                             n_calib=int(cal_idx.sum()), n_test=int(te_idx.sum())))
    if not per_seed:
        print(f"{ds}: NO valid seeds")
        continue
    accs = np.array([p["acc"] for p in per_seed])
    baccs = np.array([p["bacc"] for p in per_seed])
    kapps = np.array([p["kappa"] for p in per_seed])
    cov = (sum(p["n_calib"] + p["n_test"] for p in per_seed) / (len(per_seed) * N))
    results[ds] = dict(
        n_seeds=len(per_seed), n_windows=N, coverage=float(cov),
        acc_mean=float(accs.mean()), acc_std=float(accs.std()),
        bacc_mean=float(baccs.mean()), bacc_std=float(baccs.std()),
        kappa_mean=float(kapps.mean()), kappa_std=float(kapps.std()),
        per_seed=per_seed,
    )
    print(f"{ds}: within Acc={accs.mean()*100:5.2f}±{accs.std()*100:4.2f}  "
          f"BAcc={baccs.mean()*100:5.2f}±{baccs.std()*100:4.2f}  "
          f"kappa={kapps.mean():+.3f}  coverage={cov*100:.1f}%")

json.dump(results, open(os.path.join(OUT, "within_domain_results.json"), "w"), indent=2)
print("SAVED", os.path.join(OUT, "within_domain_results.json"))
