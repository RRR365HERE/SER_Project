"""
Multinomial Logistic Regression with softmax + cross-entropy loss,
trained from scratch via gradient descent. Supports L2 regularization.

For each class k we learn a weight vector w_k. The score for class k
on input x is w_k . x + b_k. Softmax turns scores into probabilities.
Gradient descent on the cross-entropy loss finds the weights.
"""

import numpy as np


class LogisticRegression:
    """Multinomial logistic regression (softmax) with L2 regularization.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength (smaller C = stronger regularization).
    learning_rate : float, default=0.1
        Gradient descent step size. Assumes inputs are roughly standardized.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations.
    tol : float, default=1e-5
        Stop early when |loss change| < tol between consecutive iterations.
    random_state : int or None
        Seed for reproducibility (only used if we add random init in the future).

    Attributes
    ----------
    coef_       : ndarray of shape (n_classes, n_features)
    intercept_  : ndarray of shape (n_classes,)
    classes_    : ndarray of unique class labels
    loss_history_ : list of loss values per iteration (for diagnosing convergence)
    n_iter_     : actual number of iterations run
    """

    def __init__(self, C=1.0, learning_rate=0.1, max_iter=1000,
                 tol=1e-5, random_state=None):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _softmax(logits):
        """Numerically stable softmax: subtract row max before exponentiating."""
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y])
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y_idx] = 1.0

        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)

        lambda_ = 1.0 / self.C   # L2 regularization strength
        self.loss_history_ = []
        prev_loss = np.inf

        for it in range(self.max_iter):
            logits = X @ W + b              # (n_samples, n_classes)
            P = self._softmax(logits)       # (n_samples, n_classes)

            ce = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
            l2 = 0.5 * lambda_ * np.sum(W ** 2) / n_samples

            loss = ce + l2
            self.loss_history_.append(loss)

            # ---- Backward: gradients ----
            #   dL/dW = X.T @ (P - Y) / n + lambda * W
            #   dL/db = mean(P - Y, axis=0)
            error = P - Y
            grad_W = X.T @ error / n_samples + (lambda_ / n_samples) * W

            grad_b = error.mean(axis=0)

            W -= self.learning_rate * grad_W
            b -= self.learning_rate * grad_b

            # ---- Convergence check ----
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.coef_      = W.T          # (n_classes, n_features) - sklearn convention
        self.intercept_ = b
        self.n_iter_    = it + 1
        return self

    def _decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        return self._softmax(self._decision_function(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))