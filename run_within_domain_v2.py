"""
Within-domain upper bound (v2) -- faithful to the paper's split protocol.

Protocol (per advisor checklist C1 + config.yaml):
  - Split function: sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2)
  - groups = participant identifier (subject_id); for ds006437 = session id (sub-XXX_ses-Y)
  - random_state = experiment seed (the SAME 20 seeds as config.yaml)
  - Classifier: RandomForest(n_estimators=200, min_samples_leaf=5, class_weight=balanced)
    NOTE: config.yaml sets n_estimators=500; paper Table 2 / Appendix A state 200.
    We follow the paper's stated value (200) for comparability and flag the drift.

For each dataset, the within-domain upper bound trains on 80% of its own
participants and tests on the held-out 20% (no cross-dataset source). This
mirrors the paper's cross-dataset calibration split (target 80% used for
training/adaptation, 20% held out for test) and is therefore directly
comparable to the cross-dataset *calibrated* results in multi_8ds.json.

Bootstrap 95% CI: 1000 resamples of the 20 seed-level metrics (percentile).
"""
import os, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

PROC = "E:/universal_bci_hypnosis/processed"
OUT  = "E:/universal_bci_hypnosis/results/exp101_within_domain/within_domain_results_v2.json"
DS   = ["DEAP", "DREAMER", "MAHNOB", "SEED", "SEED_IV", "FACED", "ds004572", "ds006437"]
SEEDS = [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555,
         6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789]
N_EST = 200  # paper-stated; config.yaml says 500 (flagged)

def load(ds):
    ft = np.load(os.path.join(PROC, "prep02_features", f"{ds}_features.npz"), allow_pickle=True)
    lb = np.load(os.path.join(PROC, "prep03_labels", f"{ds}_labels.npz"), allow_pickle=True)
    X = ft["features"].astype(np.float64)
    y = lb["labels"].astype(int)
    sub = np.array(ft["subject_ids"]).astype(str)
    tri = np.array(ft["trial_ids"]).astype(str)
    grp = tri if ds == "ds006437" else sub   # C2: ds006437 partitioned by session id
    return X, y, grp

def boot_ci(vals, B=1000):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(42)
    bs = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(B)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

res = {}
for ds in DS:
    X, y, grp = load(ds)
    n = len(y)
    accs, baccs, kappas = [], [], []
    n_train = n_test = 0
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, grp))
        n_train, n_test = len(tr), len(te)
        clf = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=seed)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
        kappas.append(cohen_kappa_score(y[te], pred))
    res[ds] = {
        "n_windows": int(n),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "acc_ci": boot_ci(accs),
        "bacc_mean": float(np.mean(baccs)), "bacc_std": float(np.std(baccs)),
        "bacc_ci": boot_ci(baccs),
        "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
        "kappa_ci": boot_ci(kappas),
    }
    print(f"{ds:10s} n={n:7d}  Acc={res[ds]['acc_mean']*100:5.2f}  "
          f"BAcc={res[ds]['bacc_mean']*100:5.2f}  kappa={res[ds]['kappa_mean']:+.3f}")

json.dump(res, open(OUT, "w"), indent=2)
print("WROTE", OUT)
