"""
domain_adaptation.py — Real implementations of CORAL, TCA, and AdaBN.

These methods align feature distributions between source and target domains
to enable cross-dataset transfer learning for BCI classification.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class CORAL:
    """
    Correlation Alignment (CORAL).

    Aligns second-order statistics (covariance) of source and target features.

    Reference:
        Sun, B., et al. "Return of Frustratingly Easy Domain Adaptation." AAAI 2016.
    """

    def __init__(self):
        self.source_scaler = StandardScaler()
        self.transform_matrix = None

    def fit(self, X_source, X_target):
        """
        Compute the CORAL transformation from source and target features.

        Args:
            X_source: ndarray (n_source, n_features) — source domain features
            X_target: ndarray (n_target, n_features) — target domain features

        Returns:
            self
        """
        # Standardize both domains
        Xs = self.source_scaler.fit_transform(X_source)
        Xt = StandardScaler().fit_transform(X_target)

        # Compute covariance matrices
        Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
        Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-6

        # CORAL alignment: transform source to match target covariance
        # A = Cs^{-1/2} * Ct^{1/2}
        Cs_sqrt_inv = self._matrix_sqrt_inv(Cs)
        Ct_sqrt = self._matrix_sqrt(Ct)
        self.transform_matrix = Cs_sqrt_inv @ Ct_sqrt

        return self

    def transform(self, X):
        """
        Apply CORAL transformation to align features.

        Args:
            X: ndarray (n_samples, n_features)

        Returns:
            X_aligned: ndarray (n_samples, n_features) — aligned features
        """
        X_scaled = self.source_scaler.transform(X)
        return X_scaled @ self.transform_matrix

    def fit_transform(self, X_source, X_target):
        """Fit and transform in one step."""
        self.fit(X_source, X_target)
        return self.transform(X_source)

    @staticmethod
    def _matrix_sqrt(M):
        """Compute matrix square root via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    @staticmethod
    def _matrix_sqrt_inv(M):
        """Compute inverse matrix square root via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class TCA:
    """
    Transfer Component Analysis (TCA).

    Learns a shared latent subspace that minimizes distribution divergence
    between source and target using Maximum Mean Discrepancy (MMD).

    Reference:
        Pan, S. J., et al. "Domain Adaptation via Transfer Component Analysis."
        IEEE TNN 2011.
    """

    def __init__(self, n_components=10, kernel_type='rbf', kernel_param=1.0, mu=1.0):
        """
        Args:
            n_components: int, number of transfer components
            kernel_type: str, 'rbf' or 'linear'
            kernel_param: float, kernel bandwidth for RBF
            mu: float, trade-off parameter (0=same as PCA, 1=full TCA)
        """
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.mu = mu
        self.scaler = StandardScaler()
        self.components_ = None

    def _compute_kernel(self, X, Y=None):
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        if self.kernel_type == 'linear':
            return X @ Y.T
        elif self.kernel_type == 'rbf':
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
            dist_sq = X_sq + Y_sq.T - 2 * X @ Y.T
            return np.exp(-dist_sq / (2 * self.kernel_param ** 2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def fit(self, X_source, X_target):
        """
        Learn transfer components from source and target.

        Args:
            X_source: ndarray (n_source, n_features)
            X_target: ndarray (n_target, n_features)

        Returns:
            self
        """
        n_s = X_source.shape[0]
        n_t = X_target.shape[0]
        n = n_s + n_t

        # Scale features
        X_s = self.scaler.fit_transform(X_source)
        X_t = self.scaler.transform(X_target)
        X_all = np.vstack([X_s, X_t])

        # Compute kernel matrix
        K = self._compute_kernel(X_all)
        K = K / np.trace(K + 1e-10)  # Normalize kernel

        # Construct MMD matrix (HKH formulation)
        e = np.ones((n, 1)) / n
        H = np.eye(n) - e @ e.T  # Centering matrix

        # Domain label matrix
        domain_labels = np.hstack([
            np.ones(n_s) / n_s,
            -np.ones(n_t) / n_t
        ])
        M = np.outer(domain_labels, domain_labels)

        # Objective: min_K  tr(K L K) + mu tr(K)
        # L = (I - W W^T) where W are transfer components
        L = H @ M @ H + self.mu * np.eye(n)

        # Solve generalized eigenvalue problem: K L K alpha = lambda K alpha
        # Equivalent to: L^{1/2} K L^{1/2} beta = lambda beta
        L_sqrt = CORAL._matrix_sqrt(L + 1e-6 * np.eye(n))
        KLK = L_sqrt @ K @ L_sqrt

        eigvals, eigvecs = np.linalg.eigh(KLK)

        # Select top components (smallest eigenvalues = maximum alignment)
        idx = np.argsort(eigvals)[:self.n_components]
        alpha = L_sqrt @ eigvecs[:, idx]

        # Normalize
        alpha = alpha / (np.linalg.norm(alpha, axis=0, keepdims=True) + 1e-10)
        self.components_ = alpha

        return self

    def transform(self, X):
        """
        Project features into transfer component space.

        Args:
            X: ndarray (n_samples, n_features)

        Returns:
            X_projected: ndarray (n_samples, n_components)
        """
        X_scaled = self.scaler.transform(X)
        K = self._compute_kernel(X_scaled, self.scaler.transform(X_scaled))
        return K @ self.components_

    def fit_transform(self, X_source, X_target):
        """Fit and return projected source features."""
        self.fit(X_source, X_target)
        return self.transform(X_source)


class AdaBN:
    """
    Adaptive Batch Normalization (AdaBN).

    Adapts source-trained model to target domain by recalculating batch
    normalization statistics on target data. For non-neural-network models,
    this is approximated via feature standardization.

    Reference:
        Li, Y., et al. "Revisiting Batch Normalization for Practical Domain Adaptation." ICLR 2017.
    """

    def __init__(self):
        self.source_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit(self, X_source, X_target):
        """
        Compute normalization statistics from source and target.

        Args:
            X_source: ndarray (n_source, n_features)
            X_target: ndarray (n_target, n_features) — calibration set

        Returns:
            self
        """
        self.source_scaler.fit(X_source)
        self.target_scaler.fit(X_target)
        return self

    def transform(self, X):
        """
        Apply target-domain normalization.

        Args:
            X: ndarray (n_samples, n_features)

        Returns:
            X_adapted: ndarray (n_samples, n_features)
        """
        return self.target_scaler.transform(X)

    def transform_source(self, X):
        """
        Apply source-domain normalization (for source training data).

        Args:
            X: ndarray (n_samples, n_features)

        Returns:
            X_normalized: ndarray (n_samples, n_features)
        """
        return self.source_scaler.transform(X)

    def fit_transform(self, X_source, X_target):
        """Fit and return adapted features."""
        self.fit(X_source, X_target)
        return self.transform(X_source)
