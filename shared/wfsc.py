"""
wfsc.py — Weighted Feature-Source Combination with Mahalanobis dynamic weights.

WFSC combines predictions from multiple source-domain models using
Mahalanobis distance-based weighting to estimate domain similarity.

Each source domain trains its own RF model independently. Predictions are
combined via weighted probability averaging.

Two variants:
  - WFSC_Mahalanobis: dynamic weights via inverse Mahalanobis distance
  - WFSC_Fixed: uniform weights baseline
"""

import numpy as np
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
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_leaf=5,
                class_weight='balanced', n_jobs=n_jobs, random_state=random_state
            )
        self.base_clf_template = base_clf
        self.source_models = {}
        self.source_scalers = {}
        self.weights = {}
        self.inv_cov = None
        self.n_classes = 3
        self.class_names = ['Awake', 'Light Hypnosis', 'Deep Hypnosis']
        self.weight_details = {}

    def fit(self, source_data, target_calib_data=None, target_calib_labels=None):
        """
        Train source models and compute Mahalanobis weights.

        Args:
            source_data: dict {source_name: (X_train, y_train)}
            target_calib_data: ndarray (n_calib, n_features) — calibration set from target
            target_calib_labels: ndarray, optional — for meta-learning refinement

        Returns:
            self
        """
        self.source_models = {}
        self.source_scalers = {}
        self.weights = {}

        # Train a model for each source domain
        for src_name, (X_train, y_train) in source_data.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            clf = type(self.base_clf_template)(**self.base_clf_template.get_params())
            clf.fit(X_scaled, y_train)

            self.source_models[src_name] = clf
            self.source_scalers[src_name] = scaler

        # Compute Mahalanobis weights if calibration data is provided
        if target_calib_data is not None and len(target_calib_data) > 0:
            self._compute_mahalanobis_weights(source_data, target_calib_data)
        else:
            # Fallback to uniform weights if no calibration
            n_src = len(source_data)
            self.weights = {s: 1.0 / n_src for s in source_data}
            self.weight_details['method'] = 'uniform_fallback'

        return self

    def _compute_mahalanobis_weights(self, source_data, target_calib_data):
        """Compute weights based on Mahalanobis distance."""
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

        # Pooled covariance
        cov_pooled = np.zeros((n_features, n_features))
        for cov in source_covs.values():
            cov_pooled += cov
        cov_pooled /= len(source_covs)
        cov_pooled += np.eye(n_features) * 1e-6

        self.inv_cov = np.linalg.inv(cov_pooled)

        # Target mean (from calibration data)
        target_scaled_parts = []
        for src_name in source_data:
            scaler = self.source_scalers[src_name]
            target_scaled_parts.append(scaler.transform(target_calib_data))

        # Use average across source scalers for target representation
        target_mean = np.mean(
            [np.mean(tsp, axis=0) for tsp in target_scaled_parts], axis=0
        )

        # Compute Mahalanobis distances
        distances = {}
        for src_name in source_data:
            src_mean = source_means[src_name]
            diff = src_mean - target_mean
            dist = np.sqrt(diff @ self.inv_cov @ diff)
            distances[src_name] = dist

        # Inverse distance weighting
        inv_distances = {k: 1.0 / (v + 1e-10) for k, v in distances.items()}
        total = sum(inv_distances.values())
        self.weights = {k: v / total for k, v in inv_distances.items()}

        self.weight_details = {
            'method': 'mahalanobis',
            'distances': {k: float(v) for k, v in distances.items()},
            'inv_distances': {k: float(v) for k, v in inv_distances.items()},
            'weights': {k: float(v) for k, v in self.weights.items()},
        }

    def predict_proba(self, X, source_name=None):
        """Get weighted ensemble prediction probabilities."""
        if source_name is not None:
            scaler = self.source_scalers[source_name]
            X_scaled = scaler.transform(X)
            return self.source_models[source_name].predict_proba(X_scaled)

        proba_sum = np.zeros((X.shape[0], self.n_classes))
        for src_name, model in self.source_models.items():
            scaler = self.source_scalers[src_name]
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            proba_sum += proba * self.weights[src_name]

        return proba_sum

    def predict(self, X, source_name=None):
        """Get weighted ensemble predictions."""
        proba = self.predict_proba(X, source_name)
        return np.argmax(proba, axis=1)

    def get_weights(self):
        """Return the current source domain weights."""
        return dict(self.weights)

    def get_weight_details(self):
        """Return detailed weight computation info (for analysis)."""
        return dict(self.weight_details)


class WFSC_Fixed:
    """
    Weighted Feature-Source Combination with fixed (uniform) weights.
    """

    def __init__(self, base_clf=None, n_jobs=-1, random_state=None):
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_leaf=5,
                class_weight='balanced', n_jobs=n_jobs, random_state=random_state
            )
        self.base_clf_template = base_clf
        self.source_models = {}
        self.source_scalers = {}
        self.n_classes = 3

    def fit(self, source_data, target_calib_data=None, target_calib_labels=None):
        """Train source models with uniform weighting."""
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
