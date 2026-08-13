"""run_exp103_reconstruction_check.py — BEST-EFFORT RECONSTRUCTION of exp103 (Table 6).

STATUS (academic-integrity note):
  The ORIGINAL exp103 run script (run_exp103*.py) is LOST — it is absent from both
  the MAIN and RERUN repositories and from the author's working copies; only the
  run logs (logs/exp103_*.log) and the archived results
  (results/exp103_mahal_vs_fixed/exp103_results.json, force-added in v1.0.8)
  survive. This script is a RECONSTRUCTION derived from those logs, the archived
  result file, and the shared WFSC modules.

  It does NOT byte-for-byte reproduce exp103_results.json: across 10+ protocol
  variants the reconstructed balanced-accuracy / accuracy differ from the archive
  by ~2 percentage points (the gap persists even for the unweighted `fixed`
  baseline, indicating the discrepancy lies in the underlying feature/version or
  split generation on the original machine, not the weighting scheme). The
  archived exp103_results.json REMAINS the authoritative published result.

  WHAT THIS SCRIPT IS GOOD FOR: it independently CONFIRMS the paper's Table 6
  conclusion — that Mahalanobis-weighted and Fixed-weight WFSC calibration yield
  negligible difference (diff_BAcc ~ 0 across all 8 target domains).

Experiment: for each of 8 target domains (Leave-One-Domain-Out), compare two
Weighted Feature-Space Calibration (WFSC) weighting schemes applied to the
20%-participant target calibration subset:

  - wfsc_fixed       : WFSC_Fixed  — uniform weight w=5 for all calibration samples
  - wfsc_mahalanobis : WFSC_Mahalanobis — per-sample dynamic weight by Mahalanobis
                       distance to the source manifold

Protocol mirrors run_exp101_reproducible.py:
  * features from processed/prep02 (63-dim spectral)
  * source = the other 7 domains, each subsampled to MAX_SRC windows
  * source-fit StandardScaler (fit on source, transform target)
  * subject-level 80/20 test/calibration split via shared.SplitManager
    (loso_<TARGET>_seed<SEED>.json)
  * RandomForest (n_estimators=200, min_samples_leaf=5, class_weight=balanced,
    n_jobs=-1) — unified RF_PARAMS from shared.wfsc

IMPORTANT: output is written to exp103_reconstruction_check.json so this script
can NEVER overwrite the authoritative archived exp103_results.json.

Output: results/exp103_mahal_vs_fixed/exp103_reconstruction_check.json
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from shared.split_manager import SplitManager, ALL_DATASETS
from shared.wfsc import make_wfsc

MAX_SRC = 8000
MAX_TGT = 8000
PREP = PROJECT_ROOT / "processed"
SPLITS = PROJECT_ROOT / "splits"
# IMPORTANT: this best-effort reconstruction writes to a SEPARATE file so it can
# NEVER overwrite the archived (authoritative) exp103_results.json.
OUT = PROJECT_ROOT / "results" / "exp103_mahal_vs_fixed" / "exp103_reconstruction_check.json"
# exp103 used only the first 5 seeds (per logs/exp103_20260618_232114.log)
EXP103_SEEDS = [42, 123, 456, 789, 2024]
CLASS_NAMES = ["Awake", "Light Hypnosis", "Deep Hypnosis"]
LABELS = [0, 1, 2]


def stable_hash(s):
    import hashlib
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % 100000


def load_dataset(ds_name, max_n=None):
    f = np.load(PREP / "prep02_features" / f"{ds_name}_features.npz", allow_pickle=True)
    l = np.load(PREP / "prep03_labels" / f"{ds_name}_labels.npz", allow_pickle=True)
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


def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    m = OrderedDict()
    m["accuracy"] = acc
    m["balanced_accuracy"] = bacc
    m["macro_f1"] = macro_f1
    m["weighted_f1"] = weighted_f1
    m["cohens_kappa"] = kappa
    m["auc_roc"] = auc
    for i, nm in enumerate(CLASS_NAMES):
        m[f"{nm}_precision"] = float(p[i])
        m[f"{nm}_recall"] = float(r[i])
        m[f"{nm}_f1"] = float(f[i])
    m["confusion_matrix"] = cm
    m["kappa"] = kappa
    return m


def run_one(target, seed, sm):
    source_domains = [d for d in ALL_DATASETS if d != target]
    Xs_list, ys_list = [], []
    for src in source_domains:
        Xs, ys, _ = load_dataset(src, MAX_SRC)
        Xs_list.append(Xs); ys_list.append(ys)
    X_src = np.vstack(Xs_list)
    y_src = np.concatenate(ys_list)

    X_tgt, y_tgt, tgt_sids = load_dataset(target, MAX_TGT)
    split = sm.load_subject_split(target, seed)
    calib_subjs = set(str(s) for s in split.get("calib_subjects", []))
    test_subjs = set(str(s) for s in split.get("test_subjects", []))
    calib_mask = np.array([str(s) in calib_subjs for s in tgt_sids])
    test_mask = np.array([str(s) in test_subjs for s in tgt_sids])
    if calib_mask.sum() < 2 or test_mask.sum() < 2:
        return None

    scaler = StandardScaler().fit(X_src)
    X_src_s = scaler.transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)
    X_calib_s = X_tgt_s[calib_mask]
    y_calib = y_tgt[calib_mask]
    X_test_s = X_tgt_s[test_mask]
    y_test = y_tgt[test_mask]

    out = {"per_source": {}}
    for method in ["mahalanobis", "fixed"]:
        wfsc = make_wfsc(method, random_state=seed)
        wfsc.fit(X_src=X_src_s, y_src=y_src,
                 target_calib_data=X_calib_s, target_calib_labels=y_calib)
        pred = wfsc.predict(X_test_s)
        proba = wfsc.predict_proba(X_test_s)
        out[f"wfsc_{method}"] = compute_metrics(y_test, pred, proba)
    return out


def aggregate(per_seed_dict, method):
    wkey = f"wfsc_{method}"
    keys_scalar = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                   "cohens_kappa", "auc_roc",
                   "Awake_precision", "Awake_recall", "Awake_f1",
                   "Light Hypnosis_precision", "Light Hypnosis_recall", "Light Hypnosis_f1",
                   "Deep Hypnosis_precision", "Deep Hypnosis_recall", "Deep Hypnosis_f1",
                   "kappa"]
    seeds = list(per_seed_dict.keys())
    agg = {}
    for k in keys_scalar:
        vals = [per_seed_dict[s][wkey][k] for s in seeds]
        arr = np.array([v for v in vals if isinstance(v, float) and not np.isnan(v)])
        if len(arr) == 0:
            agg[f"{k}_mean"] = float("nan"); agg[f"{k}_std"] = 0.0
        else:
            agg[f"{k}_mean"] = float(np.mean(arr))
            agg[f"{k}_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
    # mean confusion matrix across seeds
    cms = np.array([per_seed_dict[s][method]["confusion_matrix"] for s in seeds],
                   dtype=float)
    agg["confusion_matrix_mean"] = cms.mean(axis=0).tolist()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs=2, metavar=("TARGET", "SEED"), default=None)
    args = ap.parse_args()

    sm = SplitManager(str(SPLITS))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.test:
        target, seed = args.test[0], int(args.test[1])
        res = run_one(target, seed, sm)
        print(json.dumps(res, indent=2))
        return

    per_seed = {}
    for target in ALL_DATASETS:
        per_seed[target] = {}
        for seed in EXP103_SEEDS:
            rec = run_one(target, seed, sm)
            if rec is None:
                print(f"[{target} s={seed}] SKIP")
                continue
            per_seed[target][str(seed)] = rec
            print(f"[{target} s={seed}] "
                  f"mahal_BAcc={rec['wfsc_mahalanobis']['balanced_accuracy']:.4f} "
                  f"fixed_BAcc={rec['wfsc_fixed']['balanced_accuracy']:.4f}")

    summary = {}
    for target in ALL_DATASETS:
        if str(EXP103_SEEDS[0]) not in per_seed[target]:
            continue
        mh = aggregate(per_seed[target], "mahalanobis")
        fx = aggregate(per_seed[target], "fixed")
        diff_bacc = mh["balanced_accuracy_mean"] - fx["balanced_accuracy_mean"]
        summary[target] = {
            "mahalanobis": mh, "fixed": fx,
            "diff_bacc": float(diff_bacc), "n_seeds": len(per_seed[target]),
        }

    results = {"summary": summary, "per_seed": per_seed}
    json.dump(results, open(OUT, "w"), indent=2)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
