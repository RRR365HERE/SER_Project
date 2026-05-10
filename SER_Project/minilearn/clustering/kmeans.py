"""
K-Means clustering from scratch with k-means++ initialization
and multiple random restarts.
"""

import numpy as np


class KMeans:
    """K-Means clustering.

    Algorithm:
      1. Initialize k centroids (default: k-means++).
      2. Repeat:
         (E step) Assign each point to its nearest centroid.
         (M step) Move each centroid to the mean of its assigned points.
         Stop when centroid shift < tol or max_iter reached.

    To reduce sensitivity to initialization, run the whole thing `n_init`
    times with different seeds and keep the run with the lowest inertia.

    Parameters
    ----------
    n_clusters   : int, default=8
    init         : 'k-means++' or 'random', default='k-means++'
    n_init       : int, number of independent runs, default=10
    max_iter     : int, default=300
    tol          : float, convergence tolerance on centroid shift, default=1e-4
    random_state : int or None

    Attributes (set by fit)
    ----------
    cluster_centers_ : (n_clusters, n_features)
    labels_          : (n_samples,) — cluster assignment per training point
    inertia_         : float — sum of squared distances to assigned centroids
    n_iter_          : int — iterations of the best run
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _pairwise_distances_sq(X, centroids):
        """Squared Euclidean distances: shape (n_samples, n_centroids).
        Same algebraic trick as in KNN — avoids a Python loop."""
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        return np.maximum(x_sq + c_sq - 2 * X @ centroids.T, 0.0)

    def _init_centroids(self, X, rng):
        """Pick initial centroids using k-means++ (or uniform random)."""
        n = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++: spread centroids out
        centroids = np.empty((self.n_clusters, X.shape[1]))
        centroids[0] = X[rng.integers(n)]
        for k in range(1, self.n_clusters):
            d2 = self._pairwise_distances_sq(X, centroids[:k]).min(axis=1)
            total = d2.sum()
            if total == 0:
                centroids[k] = X[rng.integers(n)]
            else:
                next_idx = rng.choice(n, p=d2 / total)
                centroids[k] = X[next_idx]
        return centroids

    def _single_run(self, X, rng):
        """One full Lloyd's-algorithm run from one initialization."""
        centroids = self._init_centroids(X, rng)
        for it in range(self.max_iter):
       
            dists_sq = self._pairwise_distances_sq(X, centroids)
            labels = np.argmin(dists_sq, axis=1)

    
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster — re-seed with a random data point
                    new_centroids[k] = X[rng.integers(X.shape[0])]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

    
        dists_sq = self._pairwise_distances_sq(X, centroids)
        inertia = float(np.sum(np.min(dists_sq, axis=1)))
        return centroids, labels, inertia, it + 1

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        master_rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best = None
        for _ in range(self.n_init):
            sub_rng = np.random.default_rng(master_rng.integers(2**31))
            centroids, labels, inertia, n_iter = self._single_run(X, sub_rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best = (centroids, labels, n_iter)

        self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists_sq = self._pairwise_distances_sq(X, self.cluster_centers_)
        return np.argmin(dists_sq, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def transform(self, X):
        """Euclidean distance from each row of X to each centroid."""
        X = np.asarray(X, dtype=float)
        return np.sqrt(self._pairwise_distances_sq(X, self.cluster_centers_))