"""compute_cross_bacc_kappa.py
Read results/exp101_lodo_loso/multi_8ds.json (each record stores zs_cm / calib_cm
3x3 confusion matrices) and compute, per target dataset (cross-dataset LODO):
  - Zero-shot (ZS) and Calibrated (Cal) Balanced Accuracy (BAcc)
  - Cohen's kappa (kappa)
  - mean over 20 seeds + bootstrap 95% CI (resampling seeds, 2000 resamples)
Output: results/exp101_lodo_loso/cross_domain_bacc_kappa.json
This is the cross-domain counterpart needed to re-report Table 3 in balanced
metrics (framework B from advisor plan), without re-running the experiment.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("E:/universal_bci_hypnosis")
SRC = ROOT / "results" / "exp101_lodo_loso" / "multi_8ds.json"
OUT = ROOT / "results" / "exp101_lodo_loso" / "cross_domain_bacc_kappa.json"
N_BOOT = 2000
RNG = np.random.RandomState(20260807)

records = json.load(open(SRC))
by_target = {}
for r in records:
    by_target.setdefault(r["target"], []).append(r)


def metrics_from_cm(cm):
    cm = np.array(cm, dtype=float)
    N = cm.sum()
    if N == 0:
        return None
    # per-class recall (row-normalized) -> BAcc
    row = cm.sum(axis=1)
    recalls = [cm[i, i] / row[i] if row[i] > 0 else 0.0 for i in range(cm.shape[0])]
    bacc = float(np.mean(recalls))
    # Cohen's kappa from CM
    po = np.trace(cm) / N
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col) / (N * N))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0
    return bacc, kappa


def boot_ci(vals, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < 2:
        return float(vals.mean()), float(vals.mean())
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.randint(0, n, size=n)
        means[b] = vals[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


out = {}
for target, recs in by_target.items():
    zs_b, zs_k, cal_b, cal_k = [], [], [], []
    for r in recs:
        mz = metrics_from_cm(r["zs_cm"])
        mc = metrics_from_cm(r["calib_cm"])
        if mz:
            zs_b.append(mz[0]); zs_k.append(mz[1])
        if mc:
            cal_b.append(mc[0]); cal_k.append(mc[1])
    zs_b = np.array(zs_b); zs_k = np.array(zs_k)
    cal_b = np.array(cal_b); cal_k = np.array(cal_k)
    out[target] = {
        "n_seeds": len(recs),
        "zs_bacc_mean": float(zs_b.mean()), "zs_bacc_ci": list(boot_ci(zs_b)),
        "zs_kappa_mean": float(zs_k.mean()), "zs_kappa_ci": list(boot_ci(zs_k)),
        "cal_bacc_mean": float(cal_b.mean()), "cal_bacc_ci": list(boot_ci(cal_b)),
        "cal_kappa_mean": float(cal_k.mean()), "cal_kappa_ci": list(boot_ci(cal_k)),
    }

# overall (macro over targets, then mean over seeds already inside target)
all_zs_b = np.array([out[t]["zs_bacc_mean"] for t in out])
all_cal_b = np.array([out[t]["cal_bacc_mean"] for t in out])
all_zs_k = np.array([out[t]["zs_kappa_mean"] for t in out])
all_cal_k = np.array([out[t]["cal_kappa_mean"] for t in out])
out["__OVERALL__"] = {
    "zs_bacc_macro_mean": float(all_zs_b.mean()), "zs_bacc_ci": list(boot_ci(all_zs_b)),
    "cal_bacc_macro_mean": float(all_cal_b.mean()), "cal_bacc_ci": list(boot_ci(all_cal_b)),
    "zs_kappa_macro_mean": float(all_zs_k.mean()), "zs_kappa_ci": list(boot_ci(all_zs_k)),
    "cal_kappa_macro_mean": float(all_cal_k.mean()), "cal_kappa_ci": list(boot_ci(all_cal_k)),
}

json.dump(out, open(OUT, "w"), indent=1)
print("Saved", OUT)
order = ["DEAP", "DREAMER", "MAHNOB", "SEED", "SEED_IV", "FACED", "ds004572", "ds006437"]
print(f"{'target':10s} {'ZS_BAcc':>9s} {'95%CI':>16s} {'ZS_k':>7s} {'Cal_BAcc':>9s} {'95%CI':>16s} {'Cal_k':>7s}")
for t in order:
    o = out[t]
    print(f"{t:10s} {o['zs_bacc_mean']*100:8.2f}% [{o['zs_bacc_ci'][0]*100:5.1f},{o['zs_bacc_ci'][1]*100:5.1f}] "
          f"{o['zs_kappa_mean']:+6.3f} {o['cal_bacc_mean']*100:8.2f}% [{o['cal_bacc_ci'][0]*100:5.1f},{o['cal_bacc_ci'][1]*100:5.1f}] {o['cal_kappa_mean']:+6.3f}")
o = out["__OVERALL__"]
print(f"{'OVERALL':10s} {o['zs_bacc_macro_mean']*100:8.2f}% [{o['zs_bacc_ci'][0]*100:5.1f},{o['zs_bacc_ci'][1]*100:5.1f}] "
      f"{o['zs_kappa_macro_mean']:+6.3f} {o['cal_bacc_macro_mean']*100:8.2f}% [{o['cal_bacc_ci'][0]*100:5.1f},{o['cal_bacc_ci'][1]*100:5.1f}] {o['cal_kappa_macro_mean']:+6.3f}")
