#!/usr/bin/env python3
"""
exp108_shap_cross_domain_stability.py — SHAP interpretability and cross-domain stability.

Experiment 108 (Paper 1, Interpretability):
  - Use SHAP to explain WFSC predictions across LODO folds
  - Analyze which features are most important for hypnosis depth classification
  - Study feature importance stability across domains (cross-domain consistency)
  - Correlate feature importance with known EEG/hypnosis literature
  - Visualize feature importance patterns

Output: results/exp108_shap/exp108_shap_values.npz
        results/exp108_shap/exp108_feature_importance.json
        results/exp108_shap/exp108_stability_analysis.json
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.split_manager import SplitManager
from shared.seed_manager import SeedManager
from shared.wfsc import WFSC_Mahalanobis
from shared.metrics import compute_all_metrics
from shared.feature_extraction import FEATURE_ORDER
from shared.logger import setup_logger


def load_prep_data(dataset_name, prep_dir):
    """Load features and labels from prep01/prep02 output."""
    feat_path = prep_dir / 'prep01_features' / f'{dataset_name}_features.npz'
    label_path = prep_dir / 'prep02_labels' / f'{dataset_name}_labels.npz'

    if not feat_path.exists() or not label_path.exists():
        return None, None, None

    feat_data = np.load(feat_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)

    features = feat_data['features']
    labels = label_data['labels']
    subj_ids = label_data['subject_ids']
    valid = labels >= 0
    return features[valid], labels[valid], subj_ids[valid]


def compute_shap_values(model, X, feature_names=None):
    """
    Compute SHAP values for a trained model.

    Uses TreeExplainer for Random Forest models.

    Args:
        model: trained sklearn RF model
        X: ndarray (n_samples, n_features)
        feature_names: list of str

    Returns:
        shap_values: ndarray (n_samples, n_features)
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For multi-class: shap_values is a list of arrays (one per class)
        # We aggregate by absolute mean across classes
        if isinstance(shap_values, list):
            # Average absolute SHAP across all classes
            n_classes = len(shap_values)
            shap_abs_mean = np.zeros_like(shap_values[0])
            for cls_shap in shap_values:
                shap_abs_mean += np.abs(cls_shap)
            shap_abs_mean /= n_classes
            return shap_abs_mean, explainer
        else:
            return np.abs(shap_values), explainer

    except ImportError:
        print("  Warning: shap not installed. Using feature importances as fallback.")
        return None, None
    except Exception as e:
        print(f"  Warning: SHAP computation failed: {e}")
        return None, None


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp108', str(PROJECT_ROOT / config['logs_dir']))

    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
        logger.warning("SHAP not installed. Using sklearn feature importances as fallback.")
        logger.warning("Install with: pip install shap")

    seed_mgr = SeedManager(config['experiment']['seeds'])
    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp108_shap')
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])

    # Use first 5 seeds for SHAP (computationally expensive)
    shap_seeds = list(seed_mgr)[:5]

    all_feature_importance = {}  # {target: {seed: {feature: importance}}}
    all_shap_values = {}

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n--- Target: {target_domain} ---")

        X_target, y_target, target_subj_ids = load_prep_data(target_domain, prep_dir)
        if X_target is None or len(X_target) < 10:
            logger.warning(f"  Skipping {target_domain}")
            continue

        source_data = {}
        for src in source_domains:
            X_s, y_s, _ = load_prep_data(src, prep_dir)
            if X_s is not None and len(X_s) > 0:
                source_data[src] = (X_s, y_s)

        if len(source_data) == 0:
            continue

        all_feature_importance[target_domain] = {}

        for seed in shap_seeds:
            seed_mgr.set_seed(seed)

            # Split
            np.random.seed(seed)
            unique_subjs = list(set(str(s) for s in target_subj_ids))
            np.random.shuffle(unique_subjs)
            n_calib = max(1, int(len(unique_subjs) * 0.2))
            calib_subjs = set(unique_subjs[:n_calib])

            calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])
            X_calib = X_target[calib_mask]
            y_calib = y_target[calib_mask]
            X_test = X_target[~calib_mask]
            y_test = y_target[~calib_mask]

            if len(X_test) == 0:
                continue

            # Train WFSC-Mahalanobis
            wfsc = WFSC_Mahalanobis(random_state=seed, n_jobs=-1)
            wfsc.fit(source_data, X_calib, y_calib)

            # For SHAP, we need individual source models
            feature_imp = {}

            for src_name, model in wfsc.source_models.items():
                scaler = wfsc.source_scalers[src_name]
                X_test_scaled = scaler.transform(X_test)

                # Get sklearn feature importance
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                else:
                    imp = np.zeros(X_test.shape[1])

                feature_imp[src_name] = imp

                # SHAP values (if available)
                if HAS_SHAP:
                    # Subsample for efficiency
                    n_shap_samples = min(200, len(X_test_scaled))
                    indices = np.random.choice(len(X_test_scaled), n_shap_samples, replace=False)
                    X_shap = X_test_scaled[indices]

                    shap_vals, explainer = compute_shap_values(model, X_shap, FEATURE_ORDER)
                    if shap_vals is not None:
                        mean_imp = np.mean(shap_vals, axis=0)
                        feature_imp[f'{src_name}_shap'] = mean_imp.tolist()

            all_feature_importance[target_domain][seed] = feature_imp

            logger.info(f"  [{seed}] {target_domain}: computed importance for "
                        f"{len(feature_imp)} source models")

    # ==================================================================
    # Cross-domain stability analysis
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Domain Feature Importance Stability Analysis")
    logger.info("=" * 60)

    # Compute mean importance per feature across all targets and seeds
    feature_names = FEATURE_ORDER
    n_features = len(feature_names)

    # Aggregate: for each target, compute mean importance across seeds
    target_mean_importance = {}
    for target, seed_data in all_feature_importance.items():
        # Average across seeds and source models
        all_imps = []
        for seed, src_data in seed_data.items():
            for src_name, imp in src_data.items():
                if '_shap' not in src_name and len(imp) == n_features:
                    all_imps.append(np.array(imp))

        if all_imps:
            target_mean_importance[target] = np.mean(all_imps, axis=0)

    # Compute cross-target rank correlation (Spearman)
    stability = {}
    targets = list(target_mean_importance.keys())

    if len(targets) >= 2:
        from scipy.stats import spearmanr

        rank_correlations = []
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                r, p = spearmanr(
                    target_mean_importance[targets[i]],
                    target_mean_importance[targets[j]]
                )
                rank_correlations.append({
                    'target_a': targets[i],
                    'target_b': targets[j],
                    'spearman_r': float(r),
                    'p_value': float(p),
                })
                stability[f'{targets[i]}_vs_{targets[j]}'] = {
                    'spearman_r': float(r),
                    'p_value': float(p),
                }

        mean_corr = np.mean([c['spearman_r'] for c in rank_correlations])
        logger.info(f"  Mean pairwise Spearman rank correlation: {mean_corr:.4f}")
        logger.info(f"  Range: [{min(c['spearman_r'] for c in rank_correlations):.4f}, "
                    f"{max(c['spearman_r'] for c in rank_correlations):.4f}]")

    # Global feature importance ranking
    if target_mean_importance:
        global_imp = np.mean(list(target_mean_importance.values()), axis=0)
        ranked_features = sorted(zip(feature_names, global_imp), key=lambda x: -x[1])

        logger.info("\n  Top-10 Most Important Features (global average):")
        for rank, (fname, fimp) in enumerate(ranked_features[:10], 1):
            logger.info(f"    {rank:2d}. {fname:<40s} {fimp:.6f}")

        logger.info("\n  Bottom-5 Least Important Features:")
        for rank, (fname, fimp) in enumerate(ranked_features[-5:], n_features - 4):
            logger.info(f"    {rank:2d}. {fname:<40s} {fimp:.6f}")

        # DASM vs logBP importance comparison
        dasm_imp = np.mean(global_imp[:42])
        bp_imp = np.mean(global_imp[42:])
        logger.info(f"\n  DASM block mean importance: {dasm_imp:.6f}")
        logger.info(f"  logBP block mean importance: {bp_imp:.6f}")
        logger.info(f"  DASM/logBP ratio: {dasm_imp / (bp_imp + 1e-10):.2f}")

    # Save results
    imp_path = out_dir / 'exp108_feature_importance.json'
    with open(imp_path, 'w') as f:
        json.dump({
            'per_target_per_seed': {
                k: {str(sk): sv for sk, sv in v.items()}
                for k, v in all_feature_importance.items()
            },
            'target_mean_importance': {
                k: v.tolist() for k, v in target_mean_importance.items()
            },
            'global_ranking': [(fn, float(fi)) for fn, fi in ranked_features]
            if target_mean_importance else [],
        }, f, indent=2, default=str)

    stability_path = out_dir / 'exp108_stability_analysis.json'
    with open(stability_path, 'w') as f:
        json.dump(stability, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info("exp108 complete.")


if __name__ == '__main__':
    main()
