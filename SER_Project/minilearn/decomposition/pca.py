"""
minilearn/decomposition/pca.py

Principal Component Analysis from scratch via SVD.

Pipeline:
  1. Center the data (subtract per-feature mean).
  2. Compute SVD: X_centered = U · diag(S) · V^T.
  3. The rows of V^T are the principal components, ordered by importance.
  4. The variance explained by component k is S_k^2 / (n_samples - 1).
  5. To project new data: subtract the same mean, multiply by V^T.
"""

import numpy as np


class PCA:
    """Principal Component Analysis.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. None keeps all (= min(n_samples, n_features)).

    Attributes (set by fit)
    ----------
    mean_                     : per-feature mean of training data
    components_               : (n_components, n_features) — rows are principal axes
    explained_variance_       : (n_components,) — variance along each axis
    explained_variance_ratio_ : (n_components,) — fraction of total variance
    singular_values_          : (n_components,) — raw SVD singular values
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_ratio = explained_variance / total_variance

        n = self.n_components if self.n_components is not None else len(S)
        self.components_               = Vt[:n]
        self.singular_values_          = S[:n]
        self.explained_variance_       = explained_variance[:n]
        self.explained_variance_ratio_ = explained_ratio[:n]
        return self

    def transform(self, X):
        """Project X into the principal-component space."""
        if self.components_ is None:
            raise RuntimeError("PCA must be fit before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca):
        """Project back from PCA space to original feature space.

        With n_components < n_features, this is a reconstruction with
        compression loss — useful for measuring how much information is kept.
        """
        return X_pca @ self.components_ + self.mean_