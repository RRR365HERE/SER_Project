"""
Decision Tree classifier using the CART algorithm with Gini impurity.

Recursively splits the feature space on (feature, threshold) pairs that
minimize weighted Gini impurity in the children. Stops when nodes are
pure or when stopping criteria (max_depth, min_samples_split,
min_samples_leaf) are reached.
"""

import numpy as np


class _Node:
    """A tree node. Internal nodes have feature/threshold/left/right;
    leaves have prediction/class_counts."""
    __slots__ = ("feature", "threshold", "left", "right",
                 "prediction", "class_counts")

    def __init__(self):
        self.feature = None       
        self.threshold = None     
        self.left = None         
        self.right = None         
        self.prediction = None   
        self.class_counts = None  


class DecisionTreeClassifier:
    """CART decision tree with Gini impurity.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree (None = grow until pure).
    min_samples_split : int, default=2
        Minimum samples required to attempt a split.
    min_samples_leaf : int, default=1
        Minimum samples required in each child of a split.

    Attributes
    ----------
    classes_     : ndarray of unique class labels
    root_        : root _Node
    n_features_  : number of features
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.classes_ = None
        self.root_ = None
        self.n_features_ = None

   
    @staticmethod
    def _gini_from_counts(counts):
        """Gini = 1 - Σ p_k² for a vector of class counts."""
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p * p)

    def _class_counts(self, y_encoded, n_classes):
        """Vector of per-class counts, using bincount for speed."""
        return np.bincount(y_encoded, minlength=n_classes)

    # ---- split selection ----
    def _best_split(self, X, y_encoded, n_classes):
        """Find the (feature, threshold) split with the largest Gini gain.

        Walks the sorted values of each feature once, maintaining incremental
        left/right class counts to evaluate every threshold cheaply.
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        parent_counts = self._class_counts(y_encoded, n_classes)
        parent_gini = self._gini_from_counts(parent_counts)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            x = X[:, feature]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y_encoded[order]

            left_counts = np.zeros(n_classes, dtype=int)
            right_counts = parent_counts.copy()

            for i in range(n_samples - 1):
                cls = y_sorted[i]
                left_counts[cls] += 1
                right_counts[cls] -= 1

                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                n_left = i + 1
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left  = self._gini_from_counts(left_counts)
                gini_right = self._gini_from_counts(right_counts)
                weighted   = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = parent_gini - weighted

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y_encoded, depth, n_classes):
        node = _Node()
        node.class_counts = self._class_counts(y_encoded, n_classes)
        node.prediction = self.classes_[np.argmax(node.class_counts)]

        # Stopping criteria
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y_encoded) < self.min_samples_split:
            return node
        if len(np.unique(y_encoded)) == 1:  # pure node
            return node

        feature, threshold, gain = self._best_split(X, y_encoded, n_classes)
        if feature is None or gain <= 0:
            return node

        node.feature = feature
        node.threshold = threshold
        left_mask = X[:, feature] <= threshold
        node.left  = self._build_tree(X[left_mask],  y_encoded[left_mask],
                                       depth + 1, n_classes)
        node.right = self._build_tree(X[~left_mask], y_encoded[~left_mask],
                                       depth + 1, n_classes)
        return node

    # ---- public API ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_classes = len(self.classes_)

        # Encode y as integer class indices for fast counting
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_to_idx[v] for v in y])

        self.root_ = self._build_tree(X, y_encoded, depth=0, n_classes=n_classes)
        return self

    def _walk(self, x, node):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x, self.root_).prediction for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            leaf = self._walk(x, self.root_)
            total = leaf.class_counts.sum()
            out[i] = leaf.class_counts / total if total > 0 else 0.0
        return out

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def get_depth(self):
        def _depth(node):
            if node.feature is None:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root_)

    def get_n_leaves(self):
        def _count(node):
            if node.feature is None:
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root_)