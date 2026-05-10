"""
Simplified linear Support Vector Machine, one-vs-rest for multiclass.

Trains k binary SVMs (one per class) via subgradient descent on
hinge loss + L2 regularization. Prediction: highest decision score wins.
"""

import numpy as np


class LinearSVM:
    """Linear SVM with one-vs-rest multiclass.

    For each class k, learns weights w_k and bias b_k such that
    points in class k get score w_k·x + b_k > 0 and others get < 0,
    with as wide a margin as possible.

    Hinge loss for one binary classifier:
        L = mean(max(0, 1 - y·(w·x + b))) + (λ/2)·||w||²
    where y ∈ {-1, +1}, λ = 1/(C·n_samples).

    Parameters
    ----------
    C             : float, default=1.0  - inverse L2 regularization
    learning_rate : float, default=0.01
    max_iter      : int,   default=1000 - subgradient steps per binary classifier
    tol           : float, default=1e-5 - early stop when |ΔL| < tol
    random_state  : reserved (currently unused)

    Attributes
    ----------
    classes_   : (n_classes,) - unique class labels
    coef_      : (n_classes, n_features) - one weight vector per class
    intercept_ : (n_classes,)
    n_iter_    : list[int] - iterations used per binary classifier
    """

    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000,
                 tol=1e-5, random_state=None):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fit_binary(self, X, y_binary):
        """Train one binary SVM. y_binary in {-1, +1}."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        lambda_ = 1.0 / self.C / n_samples
        prev_loss = np.inf

        for it in range(self.max_iter):
            margins = y_binary * (X @ w + b)
            # Mask of points within or beyond margin (these contribute to gradient)
            violating = margins < 1

            # Subgradient of hinge: -y·x for violators, 0 otherwise
            # Plus L2 gradient: λ·w
            grad_w = -np.mean((y_binary * violating)[:, None] * X, axis=0) + lambda_ * w
            grad_b = -np.mean(y_binary * violating)

            # Loss (just for convergence check)
            hinge = np.mean(np.maximum(0, 1 - margins))
            loss  = hinge + 0.5 * lambda_ * np.sum(w ** 2)

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return w, b, it + 1

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.coef_      = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        self.n_iter_    = []

        for i, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1.0, -1.0)
            w, b, n_iter = self._fit_binary(X, y_binary)
            self.coef_[i]      = w
            self.intercept_[i] = b
            self.n_iter_.append(n_iter)
        return self

    def decision_function(self, X):
        """Raw decision scores: one per class. Higher = more confident."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        """Pseudo-probabilities via softmax on decision scores.
        Not calibrated probabilities, but useful for AUC and ranking."""
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)  # stability
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))