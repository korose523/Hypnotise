"""
wfsc.py — Weighted Feature-Source Combination (WFSC) with Mahalanobis dynamic weights.

Two implementations:
  - WFSC_Fixed:      Fixed weight w (default 5.0) for all calibration samples
  - WFSC_Mahalanobis: Per-sample dynamic weights based on Mahalanobis distance

[FIX M1]: Original code used fixed w=5 for all calibration samples.
          This module implements the paper's claimed Mahalanobis dynamic weighting.
          Both variants available for ablation comparison (exp103).

[FIX C2]: All experiments use unified RF_PARAMS from this module.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
import warnings


# ===========================================================================
# Unified RF parameters (FIX C2: all experiments must use same parameters)
# ===========================================================================
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'n_jobs': -1,
}


def make_rf(random_state=42):
    """Create a standard RF classifier (unified parameter entry point)."""
    params = RF_PARAMS.copy()
    params['random_state'] = random_state
    return RandomForestClassifier(**params)


# ===========================================================================
# Helper: Compute Mahalanobis distance
# ===========================================================================
def _compute_mahalanobis(X_calib, X_source, robust=True):
    """
    Compute Mahalanobis distance from calibration samples to source centroid.

    Args:
        X_calib:  (n_calib, d) target domain calibration samples
        X_source: (n_source, d) source domain samples (for covariance estimation)
        robust:   True = LedoitWolf (more stable for high-dim), False = Empirical

    Returns:
        distances: (n_calib,) Mahalanobis distances
    """
    # Limit source samples for computational efficiency
    if len(X_source) > 10000:
        idx = np.random.choice(len(X_source), 10000, replace=False)
        X_src_sub = X_source[idx]
    else:
        X_src_sub = X_source

    mu = X_src_sub.mean(axis=0)  # (d,)

    # Covariance estimation
    try:
        if robust:
            cov_est = LedoitWolf().fit(X_src_sub)
        else:
            cov_est = EmpiricalCovariance().fit(X_src_sub)
        VI = cov_est.precision_  # Precision matrix (inverse covariance)
    except Exception:
        # Fallback: diagonal covariance
        warnings.warn("Covariance estimation failed, using diagonal approximation")
        var = X_src_sub.var(axis=0) + 1e-8
        VI = np.diag(1.0 / var)

    # Mahalanobis distance: sqrt((x-mu)^T VI (x-mu))
    diff = X_calib - mu  # (n_calib, d)
    dist = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, VI, diff), 0))
    return dist


# ===========================================================================
# WFSC - Fixed weight version (original implementation, for ablation baseline)
# ===========================================================================
class WFSC_Fixed(BaseEstimator):
    """
    Fixed-weight WFSC: all calibration samples receive equal weight w.

    Corresponds to the "actual code implementation" pointed out by reviewer (w=5).
    Used as ablation baseline in exp103.
    """

    def __init__(self, w=5.0, random_state=42):
        self.w = w
        self.random_state = random_state
        self.model_ = None

    def fit_source(self, X_src, y_src):
        """Train base RF on full source domain data."""
        self.model_ = make_rf(self.random_state)
        self.model_.fit(X_src, y_src)
        self.X_src_ = X_src
        self.y_src_ = y_src
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Retrain RF with fixed-weight calibration samples (merged source + calib).
        Fixed weight: all source samples weight=1, calibration samples weight=w.
        """
        check_is_fitted(self, 'model_')

        # Merge source + calibration
        X_all = np.vstack([self.X_src_, X_calib])
        y_all = np.concatenate([self.y_src_, y_calib])
        w_src = np.ones(len(self.X_src_))
        w_cal = np.full(len(X_calib), self.w)
        sample_weight = np.concatenate([w_src, w_cal])

        self.model_.fit(X_all, y_all, sample_weight=sample_weight)
        return self

    def fit(self, source_data=None, target_calib_data=None,
            target_calib_labels=None, X_src=None, y_src=None):
        """
        Unified fit interface for backward compatibility.

        Args:
            source_data: dict {source_name: (X, y)} or None
            target_calib_data: ndarray or None
            target_calib_labels: ndarray or None
            X_src: ndarray, direct source features (when source_data is None)
            y_src: ndarray, direct source labels
        """
        if source_data is not None and X_src is None:
            # Multi-source: merge all source data
            Xs, ys = [], []
            for src_name, (X, y) in source_data.items():
                Xs.append(X)
                ys.append(y)
            X_src = np.vstack(Xs)
            y_src = np.concatenate(ys)

        self.fit_source(X_src, y_src)

        if target_calib_data is not None and len(target_calib_data) > 0:
            self.calibrate(target_calib_data, target_calib_labels)
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)

    def get_weights(self):
        """Return uniform weights for all calibration samples."""
        return {'method': 'fixed', 'w': self.w}


# ===========================================================================
# WFSC - Mahalanobis dynamic weight version (paper's claimed implementation)
# ===========================================================================
class WFSC_Mahalanobis(BaseEstimator):
    """
    Dynamic-weight WFSC: per-sample weights based on Mahalanobis distance.

    Principle (Paper Section 3.4):
      - Calibration samples closer to source manifold (smaller distance) get
        higher weights — they are more "reliable" anchors
      - Outlier calibration samples get lower weights
      - Weight formula: w_i = exp(-alpha * dist_i / median_dist)

    This provides more precise "anchoring" than uniform weighting.
    """

    def __init__(self, alpha=1.0, w_min=1.0, w_max=10.0,
                 robust_cov=True, random_state=42):
        """
        Args:
            alpha:      Distance decay coefficient (higher = faster decay)
            w_min:      Weight lower bound
            w_max:      Weight upper bound
            robust_cov: True = LedoitWolf covariance estimation
            random_state: RF random seed
        """
        self.alpha = alpha
        self.w_min = w_min
        self.w_max = w_max
        self.robust_cov = robust_cov
        self.random_state = random_state
        self.model_ = None

    def fit_source(self, X_src, y_src):
        """Train base RF on full source domain data."""
        self.model_ = make_rf(self.random_state)
        self.model_.fit(X_src, y_src)
        self.X_src_ = X_src
        self.y_src_ = y_src
        return self

    def _compute_weights(self, X_calib):
        """Compute dynamic per-sample weights based on Mahalanobis distance."""
        dist = _compute_mahalanobis(X_calib, self.X_src_, robust=self.robust_cov)

        median_dist = np.median(dist)
        if median_dist < 1e-10:
            median_dist = 1.0

        # Weight: smaller distance -> higher weight
        raw_w = np.exp(-self.alpha * dist / median_dist)

        # Linear scale to [w_min, w_max]
        raw_min, raw_max = raw_w.min(), raw_w.max()
        if raw_max - raw_min < 1e-10:
            weights = np.full(len(X_calib), (self.w_min + self.w_max) / 2)
        else:
            weights = self.w_min + (self.w_max - self.w_min) * \
                      (raw_w - raw_min) / (raw_max - raw_min)

        return weights, dist

    def calibrate(self, X_calib, y_calib):
        """
        Retrain RF with Mahalanobis dynamic weights (merged source + calib).
        """
        check_is_fitted(self, 'model_')

        weights, dist = self._compute_weights(X_calib)
        self.calib_weights_ = weights
        self.calib_distances_ = dist

        # Merge source (weight=1) + calibration (dynamic weight)
        X_all = np.vstack([self.X_src_, X_calib])
        y_all = np.concatenate([self.y_src_, y_calib])
        w_all = np.concatenate([np.ones(len(self.X_src_)), weights])

        self.model_.fit(X_all, y_all, sample_weight=w_all)

        # Record weight statistics (for paper figures)
        self.weight_stats_ = {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'median': float(np.median(weights)),
            'dist_mean': float(dist.mean()),
            'dist_median': float(np.median(dist)),
        }
        self.weight_details = {
            'method': 'mahalanobis',
            'alpha': self.alpha,
            'w_min': self.w_min,
            'w_max': self.w_max,
            'weights_mean': float(weights.mean()),
            'weights_std': float(weights.std()),
        }
        return self

    def fit(self, source_data=None, target_calib_data=None,
            target_calib_labels=None, X_src=None, y_src=None):
        """
        Unified fit interface for backward compatibility.

        Args:
            source_data: dict {source_name: (X, y)} or None
            target_calib_data: ndarray or None
            target_calib_labels: ndarray or None
            X_src: ndarray, direct source features (when source_data is None)
            y_src: ndarray, direct source labels
        """
        if source_data is not None and X_src is None:
            Xs, ys = [], []
            for src_name, (X, y) in source_data.items():
                Xs.append(X)
                ys.append(y)
            X_src = np.vstack(Xs)
            y_src = np.concatenate(ys)

        self.fit_source(X_src, y_src)

        if target_calib_data is not None and len(target_calib_data) > 0:
            self.calibrate(target_calib_data, target_calib_labels)
        else:
            # No calibration — record as uniform fallback
            self.weight_details = {'method': 'uniform_fallback'}

        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)

    def get_weights(self):
        """Return source domain weights (for backward compatibility)."""
        if hasattr(self, 'weight_details'):
            return self.weight_details
        return {'method': 'unknown'}

    def get_weight_details(self):
        """Return detailed weight computation info (for analysis)."""
        return getattr(self, 'weight_details', {})

    def get_weight_stats(self):
        """Return weight statistics (for ablation experiment output)."""
        if not hasattr(self, 'weight_stats_'):
            return {}
        return self.weight_stats_


# ===========================================================================
# Factory: create WFSC instance by method name
# ===========================================================================
def make_wfsc(method='mahalanobis', random_state=42, **kwargs):
    """
    Create a WFSC instance.

    Args:
        method: 'mahalanobis' | 'fixed'
        random_state: RF random seed
        **kwargs: additional parameters passed to the specific implementation

    Returns:
        WFSC instance
    """
    if method == 'mahalanobis':
        return WFSC_Mahalanobis(random_state=random_state, **kwargs)
    elif method == 'fixed':
        w = kwargs.pop('w', 5.0)
        return WFSC_Fixed(w=w, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown WFSC method: {method}. Available: 'mahalanobis', 'fixed'"
        )
