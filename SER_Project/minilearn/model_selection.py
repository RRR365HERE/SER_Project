"""
K-fold and stratified K-fold cross-validation utilities.

`cross_val_score` accepts a `scale` flag that fits a StandardScaler
inside each fold's training data - the no-leakage way to combine
CV with standardization.
"""

from copy import deepcopy
import numpy as np

from minilearn.preprocessing import StandardScaler
from minilearn.metrics import accuracy_score, f1_score


class KFold:
    """Splits indices into k consecutive folds (with optional shuffling)."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(idx)

        # Build fold sizes (handles non-divisible n by adding 1 to the first remainder folds)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        for size in fold_sizes:
            test_idx = idx[current : current + size]
            train_idx = np.concatenate([idx[:current], idx[current + size :]])
            yield train_idx, test_idx
            current += size


class StratifiedKFold:
    """Stratified K-fold: every fold has the same class proportions as the full dataset.

    Important for imbalanced data like RAVDESS where `neutral` is half-size.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        # Split each class's indices into n_splits chunks
        class_chunks = {}
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                cls_idx = rng.permutation(cls_idx)
            class_chunks[cls] = np.array_split(cls_idx, self.n_splits)

        for k in range(self.n_splits):
            test_idx  = np.concatenate([class_chunks[c][k] for c in class_chunks])
            train_idx = np.concatenate([
                np.concatenate(class_chunks[c][:k] + class_chunks[c][k + 1 :])
                for c in class_chunks
            ])
            if self.shuffle:
                test_idx  = rng.permutation(test_idx)
                train_idx = rng.permutation(train_idx)
            yield train_idx, test_idx


def _get_scorer(scoring):
    if callable(scoring):
        return scoring
    if scoring == "accuracy":
        return accuracy_score
    if scoring == "f1_macro":
        return lambda y, p: f1_score(y, p, average="macro")
    if scoring == "f1_weighted":
        return lambda y, p: f1_score(y, p, average="weighted")
    raise ValueError(f"Unknown scoring: {scoring!r}")


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy",
                    stratified=True, scale=False, random_state=None):
    """Evaluate an estimator with k-fold cross-validation.

    Parameters
    ----------
    estimator : object with fit/predict (UNFITTED — will be deep-copied per fold)
    X, y      : data and labels
    cv        : number of folds
    scoring   : 'accuracy', 'f1_macro', 'f1_weighted', or a callable
    stratified: if True, use StratifiedKFold (recommended for classification)
    scale     : if True, fit a StandardScaler INSIDE each fold's training set
                and apply it to the test fold. Prevents data leakage.
    random_state : seed for fold shuffling

    Returns
    -------
    ndarray of shape (cv,) — score on each held-out fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    scorer = _get_scorer(scoring)

    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(n_splits=cv, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)

        model = deepcopy(estimator)         # fresh, unfitted clone
        model.fit(X_tr, y_tr)
        scores.append(scorer(y_te, model.predict(X_te)))

    return np.array(scores)