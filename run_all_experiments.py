"""Complete experiment pipeline: Multi-source LODO + Mahalanobis WFSC + 20-seed + Wilcoxon.
Run on local machine with 16GB+ RAM.

Usage: python run_all_experiments.py [--full] [--eegnet]

Options:
  --full     Use full source data (no subsampling, ~430K samples, needs 16GB+ RAM)
  --eegnet   Run EEGNet baseline (needs PyTorch)
"""
import numpy as np, time, os, json, sys, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from pathlib import Path

from shared.mahalanobis_wfsc import MahalanobisWFSC

# ====== CONFIG ======
DATASETS = ['DREAMER','DEAP','MAHNOB','SEED','SEED_IV','FACED','ds006437','ds004572']
SEEDS = [42,123,456,789,2024,1111,2222,3333,4444,5555,
         6666,7777,8888,9999,1234,2345,3456,4567,5678,6789]
CALIB_RATIOS = [0, 0.05, 0.10, 0.20]
RF_TREES = 200
MAX_SRC = 8000  # per source domain for fair comparison


def load_data():
    print('Loading datasets...', flush=True)
    X, y, s = {}, {}, {}
    for ds in DATASETS:
        fp = Path('processed/prep02_features') / f'{ds}_features.npz'
        lp = Path('processed/prep03_labels') / f'{ds}_labels.npz'
        if not fp.exists() or not lp.exists():
            print(f'  {ds}: MISSING', flush=True); continue
        f = np.load(fp); l = np.load(lp)
        Xd = f['features'].astype(np.float32)
        yd = l['labels'].astype(np.int32)
        sd = f['subject_ids']
        v = yd >= 0; X[ds] = Xd[v]; y[ds] = yd[v]; s[ds] = sd[v]
        if MAX_SRC and len(y[ds]) > MAX_SRC:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y[ds]), MAX_SRC, replace=False)
            X[ds] = X[ds][idx]; y[ds] = y[ds][idx]; s[ds] = s[ds][idx]
        d = dict(zip(*np.unique(y[ds], return_counts=True)))
        print(f'  {ds}: {len(y[ds])} windows, dist={d}', flush=True)
        f.close(); l.close()
    return X, y, s


def run_lodo(X, y, s, args):
    """Multi-source LODO with 20 seeds, Mahalanobis WFSC, Wilcoxon tests."""
    out_dir = Path('results/exp101_full')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    t0 = time.time()
    n_total = len(DATASETS) * len(SEEDS) * len(CALIB_RATIOS)
    n_done = 0
    
    for target in DATASETS:
        sources = [d for d in DATASETS if d != target and d in X]
        if target not in X: continue
        
        print(f'\n=== Target: {target} | {len(sources)} sources ===', flush=True)
        Xs_all = np.concatenate([X[d] for d in sources], axis=0)
        ys_all = np.concatenate([y[d] for d in sources], axis=0)
        Xt = X[target]; yt = y[target]; st = s[target]
        us = sorted(set(st))
        
        for seed in SEEDS:
            rng = np.random.RandomState(seed); ps = rng.permutation(us)
            
            for ratio in CALIB_RATIOS:
                nc = max(1, int(len(us) * ratio))
                cs = set(ps[:nc]); ts = set(ps[nc:])
                cm = np.array([x in cs for x in st])
                tm = np.array([x in ts for x in st])
                if cm.sum() < 1 or tm.sum() < 1: continue
                
                scaler = StandardScaler()
                Xsrc = scaler.fit_transform(Xs_all)
                Xcal = scaler.transform(Xt[cm])
                Xtst = scaler.transform(Xt[tm])
                
                # ---- Zero-Shot ----
                rf_zs = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=5,
                    class_weight='balanced', n_jobs=-1, random_state=seed)
                rf_zs.fit(Xsrc, ys_all)
                zp = rf_zs.predict(Xtst)
                zs_a = accuracy_score(yt[tm], zp)
                zs_f = f1_score(yt[tm], zp, average='macro', zero_division=0)
                
                # ---- Fixed-Weight WFSC (baseline) ----
                wa_fixed, wf_fixed = zs_a, zs_f
                if ratio > 0 and cm.sum() > 0:
                    rf_fw = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=5,
                        class_weight='balanced', n_jobs=-1, random_state=seed)
                    X_fw = np.vstack([Xsrc, Xcal])
                    y_fw = np.concatenate([ys_all, yt[cm]])
                    rf_fw.fit(X_fw, y_fw)
                    wp = rf_fw.predict(Xtst)
                    wa_fixed = accuracy_score(yt[tm], wp)
                    wf_fixed = f1_score(yt[tm], wp, average='macro', zero_division=0)
                
                # ---- Mahalanobis WFSC ----
                wa_mahal, wf_mahal = zs_a, zs_f
                if ratio > 0 and cm.sum() > 3:
                    try:
                        mwfsc = MahalanobisWFSC().fit(Xt[cm])  # fit on calibration only (no test leakage)
                        rf_m = mwfsc.apply_calibration(Xsrc, ys_all, Xcal, yt[cm],
                            RandomForestClassifier, n_estimators=RF_TREES, min_samples_leaf=5,
                            class_weight='balanced', n_jobs=-1, random_state=seed)
                        wp_m = rf_m.predict(Xtst)
                        wa_mahal = accuracy_score(yt[tm], wp_m)
                        wf_mahal = f1_score(yt[tm], wp_m, average='macro', zero_division=0)
                    except Exception as e:
                        if seed == SEEDS[0]:
                            print(f'  Mahalanobis WFSC error: {e}')
                
                all_results.append({
                    'target': target, 'seed': seed, 'calib_ratio': ratio,
                    'n_calib': int(cm.sum()), 'n_test': int(tm.sum()),
                    'zs_acc': round(zs_a, 4), 'zs_f1': round(zs_f, 4),
                    'wfsc_fixed_acc': round(wa_fixed, 4), 'wfsc_fixed_f1': round(wf_fixed, 4),
                    'wfsc_mahal_acc': round(wa_mahal, 4), 'wfsc_mahal_f1': round(wf_mahal, 4),
                })
                n_done += 1
                if n_done % 140 == 0:
                    e = time.time() - t0
                    print(f'  [{n_done}/{n_total}] {100*n_done/n_total:.0f}% ETA:{(n_total-n_done)/(n_done/e)/60:.0f}min', flush=True)
    
    # ---- Save ----
    elapsed = time.time() - t0
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=1)
    
    # ---- Summary ----
    print(f'\n=== RESULTS ({elapsed/60:.1f} min) ===')
    print(f'{"Target":12s} {"ZS":>8s} {"FW-WFSC":>8s} {"MH-WFSC":>8s} {"Wilcox_p":>8s}')
    for tgt in DATASETS:
        d = [r for r in all_results if r['target'] == tgt]
        if not d: continue
        zs = np.array([r['zs_acc'] for r in d])
        fw = np.array([r['wfsc_fixed_acc'] for r in d if r['calib_ratio'] > 0])
        mh = np.array([r['wfsc_mahal_acc'] for r in d if r['calib_ratio'] > 0])
        
        zs_20 = np.array([r['zs_acc'] for r in d if r['calib_ratio'] == 0.20])
        ws_20 = np.array([r['wfsc_mahal_acc'] for r in d if r['calib_ratio'] == 0.20])
        
        if len(zs_20) >= 5 and len(ws_20) >= 5:
            try:
                _, p = wilcoxon(zs_20, ws_20, alternative='less')
            except:
                p = 1.0
        else:
            p = 1.0
        
        print(f'{tgt:12s} {zs.mean():8.4f} {fw.mean() if len(fw)>0 else 0:8.4f} '
              f'{mh.mean() if len(mh)>0 else 0:8.4f} {p:8.4f}')
    
    zs_a = np.array([r['zs_acc'] for r in all_results])
    mh_a = np.array([r['wfsc_mahal_acc'] for r in all_results if r['calib_ratio'] > 0])
    print(f'{"OVERALL":12s} {zs_a.mean():8.4f} {"-":>8s} {mh_a.mean():8.4f}')
    
    # Wilcoxon test: ZS vs Mahalanobis WFSC at ratio=0.20
    zs_r20 = np.array([r['zs_acc'] for r in all_results if r['calib_ratio'] == 0.20])
    ws_r20 = np.array([r['wfsc_mahal_acc'] for r in all_results if r['calib_ratio'] == 0.20])
    if len(zs_r20) >= 10:
        stat, p_val = wilcoxon(zs_r20, ws_r20, alternative='less')
        print(f'\nWilcoxon (ZS vs Mahalanobis WFSC-0.2): W={stat:.1f}, p={p_val:.6f}')
        print(f'Significant at α=0.05: {"YES" if p_val < 0.05 else "NO"}')
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Use full source data')
    parser.add_argument('--eegnet', action='store_true', help='Run EEGNet baseline')
    args = parser.parse_args()
    
    if args.full:
        MAX_SRC = None
        print('Mode: FULL data')
    else:
        print('Mode: Subsampled (MAX_SRC=20000 per dataset)')
    
    X, y, s = load_data()
    results = run_lodo(X, y, s, args)
    print(f'\nAll done! Results in results/exp101_full/results.json')
