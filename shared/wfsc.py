"""
wfsc.py — Weighted Feature-Source Combination with Mahalanobis dynamic weights.

WFSC combines predictions from multiple source-domain models using
Mahalanobis distance-based weighting to estimate domain similarity.

Reference (modified):
    Pan, S. J., & Yang, Q. "A Survey on Transfer Learning." IEEE TKDE 2010.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class WFSC_Mahalanobis:
    """
    Weighted Feature-Source Combination with Mahalanobis distance weighting.

    For each source domain, trains a classifier and computes Mahalanobis
    distance between source and target feature distributions. Closer
    distributions receive higher weights in the ensemble.
    """

    def __init__(self, base_clf=None, n_jobs=-1, random_state=None):
        """
        Args:
            base_clf: classifier instance. Default: RandomForest with config params.
            n_jobs: int, parallel jobs for base classifier
            random_state: int, random seed
        """
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=5,
                class_weight='balanced',
                n_jobs=n_jobs,
                random_state=random_state
            )
        self.base_clf_template = base_clf
        self.source_models = {}     # {source_name: fitted_classifier}
        self.source_scalers = {}    # {source_name: StandardScaler}
        self.weights = {}           # {source_name: weight}
        self.inv_cov = None         # Inverse pooled covariance for Mahalanobis
        self.pooled_mean = None     # Pooled mean vector
        self.n_classes = 3
        self.class_names = ['Awake', 'Light Hypnosis', 'Deep Hypnosis']

    def fit(self, source_data, target_calib_data, target_calib_labels=None):
        """
        Train source models and compute Mahalanobis weights.

        Args:
            source_data: dict {source_name: (X_train, y_train)}
                X_train: ndarray (n_samples, n_features)
                y_train: ndarray (n_samples,)
            target_calib_data: dict {source_name: X_calib}
                X_calib: ndarray (n_calib, n_features) — calibration set from target
            target_calib_labels: ndarray, optional — for meta-learning refinement

        Returns:
            self
        """
        self.source_models = {}
        self.source_scalers = {}

        # Train a model for each source domain
        for src_name, (X_train, y_train) in source_data.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            clf = type(self.base_clf_template)(**self.base_clf_template.get_params())
            clf.fit(X_scaled, y_train)

            self.source_models[src_name] = clf
            self.source_scalers[src_name] = scaler

        # Compute Mahalanobis distances and weights
        self._compute_mahalanobis_weights(source_data, target_calib_data)

        return self

    def _compute_mahalanobis_weights(self, source_data, target_calib_data):
        """
        Compute weights based on Mahalanobis distance between source
        and target feature distributions.

        Closer sources get higher weights.
        """
        source_means = {}
        source_covs = {}
        n_features = None

        for src_name, (X_train, _) in source_data.items():
            scaler = self.source_scalers[src_name]
            X_scaled = scaler.transform(X_train)
            source_means[src_name] = np.mean(X_scaled, axis=0)
            source_covs[src_name] = np.cov(X_scaled, rowvar=False)
            if n_features is None:
                n_features = X_scaled.shape[1]

        # Compute pooled covariance from all sources
        cov_pooled = np.zeros((n_features, n_features))
        for cov in source_covs.values():
            cov_pooled += cov
        cov_pooled /= len(source_covs)
        cov_pooled += np.eye(n_features) * 1e-6  # Regularization

        # Inverse covariance for Mahalanobis distance
        self.inv_cov = np.linalg.inv(cov_pooled)

        # Compute Mahalanobis distance for each source-target pair
        distances = {}
        for src_name in source_data:
            src_mean = source_means[src_name]
            X_calib = target_calib_data.get(src_name)
            if X_calib is None:
                # Use pooled target estimate
                target_mean = np.mean(
                    [source_means[s] for s in source_data], axis=0)
            else:
                scaler = self.source_scalers[src_name]
                X_calib_scaled = scaler.transform(X_calib)
                target_mean = np.mean(X_calib_scaled, axis=0)

            diff = src_mean - target_mean
            dist = np.sqrt(diff @ self.inv_cov @ diff)
            distances[src_name] = dist

        # Convert distances to weights (inverse distance, normalized)
        inv_distances = {k: 1.0 / (v + 1e-10) for k, v in distances.items()}
        total = sum(inv_distances.values())
        self.weights = {k: v / total for k, v in inv_distances.items()}

    def predict_proba(self, X, source_name=None):
        """
        Get weighted ensemble prediction probabilities.

        Args:
            X: ndarray (n_samples, n_features)
            source_name: str, if provided, use only this source's model

        Returns:
            proba: ndarray (n_samples, n_classes) — weighted probabilities
        """
        if source_name is not None:
            scaler = self.source_scalers[source_name]
            X_scaled = scaler.transform(X)
            return self.source_models[source_name].predict_proba(X_scaled)

        # Weighted ensemble prediction
        proba_sum = np.zeros((X.shape[0], self.n_classes))
        for src_name, model in self.source_models.items():
            scaler = self.source_scalers[src_name]
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            proba_sum += proba * self.weights[src_name]

        return proba_sum

    def predict(self, X, source_name=None):
        """
        Get weighted ensemble predictions.

        Args:
            X: ndarray (n_samples, n_features)

        Returns:
            predictions: ndarray (n_samples,) — predicted class labels
        """
        proba = self.predict_proba(X, source_name)
        return np.argmax(proba, axis=1)

    def get_weights(self):
        """Return the current source domain weights."""
        return dict(self.weights)


class WFSC_Fixed:
    """
    Weighted Feature-Source Combination with fixed (uniform) weights.

    Simple baseline: all source domains contribute equally.
    """

    def __init__(self, base_clf=None, n_jobs=-1, random_state=None):
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=5,
                class_weight='balanced',
                n_jobs=n_jobs,
                random_state=random_state
            )
        self.base_clf_template = base_clf
        self.source_models = {}
        self.source_scalers = {}
        self.n_classes = 3

    def fit(self, source_data, target_calib_data=None, target_calib_labels=None):
        """
        Train source models with uniform weighting.

        Args:
            source_data: dict {source_name: (X_train, y_train)}
            target_calib_data: ignored (kept for API compatibility)
            target_calib_labels: ignored

        Returns:
            self
        """
        self.source_models = {}
        self.source_scalers = {}

        for src_name, (X_train, y_train) in source_data.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            clf = type(self.base_clf_template)(**self.base_clf_template.get_params())
            clf.fit(X_scaled, y_train)

            self.source_models[src_name] = clf
            self.source_scalers[src_name] = scaler

        return self

    def predict_proba(self, X, source_name=None):
        """Get ensemble prediction probabilities (uniform weights)."""
        if source_name is not None:
            scaler = self.source_scalers[source_name]
            X_scaled = scaler.transform(X)
            return self.source_models[source_name].predict_proba(X_scaled)

        n_sources = len(self.source_models)
        proba_sum = np.zeros((X.shape[0], self.n_classes))
        for src_name, model in self.source_models.items():
            scaler = self.source_scalers[src_name]
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            proba_sum += proba

        return proba_sum / n_sources

    def predict(self, X, source_name=None):
        """Get ensemble predictions (uniform weights)."""
        proba = self.predict_proba(X, source_name)
        return np.argmax(proba, axis=1)
