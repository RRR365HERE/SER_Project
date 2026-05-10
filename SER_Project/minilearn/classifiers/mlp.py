"""
Multi-Layer Perceptron classifier from scratch.

Architecture: Input -> Linear(W1) -> ReLU -> Linear(W2) -> Softmax -> Output

Trained with mini-batch stochastic gradient descent on cross-entropy loss.
Implements forward AND backward passes (backpropagation) from scratch
using only NumPy.
"""

import numpy as np


class MLPClassifier:
    """One-hidden-layer feedforward neural network.

    Parameters
    ----------
    hidden_size   : int, default=64    — number of neurons in the hidden layer
    learning_rate : float, default=0.01
    max_iter      : int, default=200    — number of training epochs (full passes)
    batch_size    : int, default=32     — mini-batch size for SGD
    C             : float, default=1.0  — inverse L2 strength
    random_state  : int or None         — seed for weight initialization

    Attributes
    ----------
    classes_       : ndarray of unique class labels
    W1, b1, W2, b2 : trained weights and biases
    loss_history_  : list of mean training loss per epoch
    n_iter_        : number of epochs actually run
    """

    def __init__(self, hidden_size=64, learning_rate=0.01, max_iter=200,
                 batch_size=32, C=1.0, random_state=None):
        self.hidden_size   = hidden_size
        self.learning_rate = learning_rate
        self.max_iter      = max_iter
        self.batch_size    = batch_size
        self.C             = C
        self.random_state  = random_state


    @staticmethod
    def _relu(z):       return np.maximum(z, 0)
    @staticmethod
    def _relu_grad(z):  return (z > 0).astype(float)

    @staticmethod
    def _softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)  # stability
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, X):
        """Compute predictions and cache intermediate values for backprop."""
        z1 = X @ self.W1 + self.b1            # pre-activation hidden
        a1 = self._relu(z1)                   # post-activation hidden
        z2 = a1 @ self.W2 + self.b2           # logits
        y_hat = self._softmax(z2)             # probabilities
        return y_hat, (z1, a1)

    def _backward(self, X, Y, y_hat, cache, lambda_):
        """Backprop: chain rule from loss back to W1, b1, W2, b2."""
        z1, a1 = cache
        n = X.shape[0]

        dz2 = (y_hat - Y) / n                 # (batch, n_classes)
        dW2 = a1.T @ dz2 + lambda_ * self.W2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T                 # (batch, hidden)
        dz1 = da1 * self._relu_grad(z1)       # ReLU derivative gates the gradient
        dW1 = X.T @ dz1 + lambda_ * self.W1
        db1 = dz1.sum(axis=0)

        return dW1, db1, dW2, db2

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

        rng = np.random.default_rng(self.random_state)
        self.W1 = rng.standard_normal((n_features, self.hidden_size)) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.standard_normal((self.hidden_size, n_classes)) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(n_classes)

        lambda_ = 1.0 / self.C / n_samples
        batch_size = self.batch_size if self.batch_size else n_samples
        self.loss_history_ = []

        for epoch in range(self.max_iter):
            order = rng.permutation(n_samples)
            X_shuf, Y_shuf = X[order], Y[order]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                Y_batch = Y_shuf[start : start + batch_size]

                # Forward
                y_hat, cache = self._forward(X_batch)

                # Loss (for monitoring)
                ce = -np.mean(np.sum(Y_batch * np.log(y_hat + 1e-12), axis=1))
                l2 = 0.5 * lambda_ * (np.sum(self.W1**2) + np.sum(self.W2**2))
                epoch_loss += ce + l2
                n_batches += 1

                # Backward
                dW1, db1, dW2, db2 = self._backward(X_batch, Y_batch, y_hat, cache, lambda_)

                # Update
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2

            self.loss_history_.append(epoch_loss / n_batches)

        self.n_iter_ = self.max_iter
        return self

    def predict_proba(self, X):
        y_hat, _ = self._forward(np.asarray(X, dtype=float))
        return y_hat

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))