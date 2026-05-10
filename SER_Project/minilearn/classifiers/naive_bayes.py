"""
Gaussian Naive Bayes classifier from scratch.

Bayes' theorem:  P(class | x) ∝ P(class) · P(x | class)


Here I used  log space to avoid underflow when multiplying many small
probabilities together.
"""

import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Small fraction of the largest feature variance added to all variances
        for numerical stability (prevents division by 0 on near-constant features).

    Attributes
    ----------
    classes_       : ndarray, unique class labels.
    class_priors_  : ndarray, log P(class) for each class.
    means_         : ndarray of shape (n_classes, n_features).
    variances_     : ndarray of shape (n_classes, n_features).
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = None  
        self.means_ = None
        self.variances_ = None

    def fit(self, X, y):
        """Estimate priors, means, and variances per class."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.class_priors_ = np.zeros(n_classes)
        self.means_      = np.zeros((n_classes, n_features))
        self.variances_  = np.zeros((n_classes, n_features))

        # Smoothing: same thing sklearn uses
        epsilon = self.var_smoothing * X.var(axis=0).max()

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_priors_[i] = np.log(len(X_cls) / len(X))
            self.means_[i]        = X_cls.mean(axis=0)
            self.variances_[i]    = X_cls.var(axis=0) + epsilon
        return self

    def _log_likelihood(self, X):
        """Compute unnormalized log P(class | x) for each (sample, class).

        For a Gaussian feature:
            log P(x_f | class) = -0.5 * [ log(2π σ²) + (x_f − μ)² / σ² ]
        Summed over features, plus the log prior, gives our score per class.
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            mean, var = self.means_[i], self.variances_[i]
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * var))
            sq_dist  = -0.5 * np.sum((X - mean) ** 2 / var, axis=1)
            scores[:, i] = self.class_priors_[i] + log_norm + sq_dist
        return scores

    def predict(self, X):
        """Pick the class with the highest log-posterior score."""
        return self.classes_[np.argmax(self._log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        """Convert log scores to proper probabilities via softmax."""
        log_probs = self._log_likelihood(X)
        # Subtract row max for numerical stability before exponentiating
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))