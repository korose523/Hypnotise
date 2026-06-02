"""Mahalanobis WFSC - Dynamic weight calibration for cross-domain EEG classification.

Implements the actual Mahalanobis-distance-based dynamic weighting that the
original paper describes but the current code doesn't implement.
"""
import numpy as np
from sklearn.covariance import LedoitWolf

class MahalanobisWFSC:
    """
    Weighted Feature Space Calibration using Mahalanobis distance.
    
    For each calibration sample, computes its Mahalanobis distance from 
    the target domain centroid. Samples closer to the centroid get higher 
    weights when added to the source training set.
    
    Args:
        reg_covar: float, regularization for LedoitWolf covariance (default 0.1)
    """
    def __init__(self, reg_covar=0.1):
        self.reg_covar = reg_covar
        self.target_centroid_ = None
        self.inv_cov_ = None
        
    def fit(self, X_target):
        """Compute target domain centroid and precision matrix."""
        self.target_centroid_ = X_target.mean(axis=0)
        lw = LedoitWolf().fit(X_target)
        # Regularize the covariance matrix
        cov = lw.covariance_ + np.eye(X_target.shape[1]) * self.reg_covar
        self.inv_cov_ = np.linalg.inv(cov)
        return self
    
    def compute_weights(self, X_calib):
        """Compute Mahalanobis distance-based weights for calibration samples.
        
        Returns:
            weights: ndarray (n_calib,) - higher weight = closer to centroid
        """
        if self.target_centroid_ is None:
            raise ValueError("Must call fit() before compute_weights()")
        
        # Batch Mahalanobis: (x - μ)^T Σ^{-1} (x - μ)
        delta = X_calib - self.target_centroid_
        md = np.sum(delta @ self.inv_cov_ * delta, axis=1)  # (n_calib,)
        
        # Convert distance to weight: w = exp(-λ * d)
        # Scale such that max distance gets weight ~1, min gets weight ~5
        md = np.maximum(md, 1e-10)  # avoid div by zero
        weights = np.exp(-0.5 * md / np.median(md))
        
        # Scale to [1, 5] range
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = 1 + 4 * (weights - w_min) / (w_max - w_min)
        else:
            weights = np.ones_like(weights) * 3
        
        return weights
    
    def apply_calibration(self, X_source, y_source, X_calib, y_calib, model_cls, **model_kwargs):
        """Train a model with Mahalanobis-weighted calibration samples.
        
        Args:
            X_source, y_source: source domain data
            X_calib, y_calib: target domain calibration data  
            model_cls: sklearn classifier class (e.g. RandomForestClassifier)
            **model_kwargs: passed to model constructor
            
        Returns:
            Trained model with calibration applied
        """
        weights = self.compute_weights(X_calib)
        
        # Train RF with sample_weight
        X_all = np.vstack([X_source, X_calib])
        y_all = np.concatenate([y_source, y_calib])
        w_all = np.concatenate([np.ones(len(X_source)), weights])
        
        model = model_cls(**model_kwargs)
        model.fit(X_all, y_all, sample_weight=w_all)
        return model
