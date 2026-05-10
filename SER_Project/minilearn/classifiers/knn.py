"""
k-Nearest Neighbors classifier from scratch.

In KNN: fit() just stores the training data; all the
real work happens at predict time, where we find the k closest training
points to each query point and vote on the most common class.
"""

import numpy as np


class KNeighborsClassifier:
    """k-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors used for each prediction.
    weights : {'uniform', 'distance'}, default='uniform'
        - 'uniform': each neighbor casts an equal vote.
        - 'distance': closer neighbors get larger votes (weight = 1 / distance).

    Attributes
    ----------
    X_train_ : ndarray of shape (n_train, n_features)
    y_train_ : ndarray of shape (n_train,)
    classes_ : ndarray of unique class labels (sorted)
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Store training data. KNN does no real learning at fit-time."""
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _pairwise_distances(self, X):
        """Euclidean distances from each row of X to each row of X_train_.

        Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b,
        which lets us replace a slow double loop with a single matrix
        multiplication that NumPy handles very efficiently.

        Returns
        -------
        ndarray of shape (n_query, n_train)
        """
        X = np.asarray(X, dtype=float)
        test_sq  = np.sum(X ** 2, axis=1, keepdims=True)              # (n_query, 1)
        train_sq = np.sum(self.X_train_ ** 2, axis=1, keepdims=True).T  # (1, n_train)
        d_sq = test_sq + train_sq - 2 * X @ self.X_train_.T
        # Tiny negatives can appear from floating-point error; clip to 0.
        return np.sqrt(np.maximum(d_sq, 0))

    def predict(self, X):
        """Predict class labels for each row of X."""
        distances = self._pairwise_distances(X)
        # argpartition is faster than full sort: brings the k smallest to the front
        k_idx = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

        predictions = np.empty(len(X), dtype=self.y_train_.dtype)
        for i in range(len(X)):
            neighbor_labels = self.y_train_[k_idx[i]]
            if self.weights == "uniform":
                vals, counts = np.unique(neighbor_labels, return_counts=True)
                predictions[i] = vals[np.argmax(counts)]
            else:  # distance-weighted
                neighbor_dists = distances[i, k_idx[i]]
                w = 1 / np.maximum(neighbor_dists, 1e-10)
                tally = {}
                for label, weight in zip(neighbor_labels, w):
                    tally[label] = tally.get(label, 0) + weight
                predictions[i] = max(tally, key=tally.get)
        return predictions

    def predict_proba(self, X):
        """Estimate class probabilities (fraction of neighbors in each class)."""
        distances = self._pairwise_distances(X)
        k_idx = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((len(X), len(self.classes_)))

        for i in range(len(X)):
            neighbor_labels = self.y_train_[k_idx[i]]
            if self.weights == "uniform":
                for label in neighbor_labels:
                    proba[i, class_to_idx[label]] += 1
                proba[i] /= self.n_neighbors
            else:
                neighbor_dists = distances[i, k_idx[i]]
                w = 1 / np.maximum(neighbor_dists, 1e-10)
                for label, weight in zip(neighbor_labels, w):
                    proba[i, class_to_idx[label]] += weight
                proba[i] /= proba[i].sum()
        return proba

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == y))