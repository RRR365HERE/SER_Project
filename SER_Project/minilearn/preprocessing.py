"""
This function is used in as data preprocessing utilities: feature standardization and train/test splitting.
"""

import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    for each feature column j:
        z_j = (x_j - mean_j) / std_j

    The mean and std are infered from the training data via fit(), then applied
    unchanged to any other data via transform(). This important in cases of spliting the data
    to preven data leakage.

    Attributes
    ----------
    mean_: ndarray of shape (n_features,)
        per-feature mean computed during fit().
    std_ : ndarray of shape (n_features,)
        per-feature standard deviation computed during fit().
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit ( self, X ):
        """ Compute per-feature mean and std from X."""
        X = np.asarray( X, dtype=float )
        self.mean_ = X.mean(axis = 0)
        self.std_ = X.std(axis = 0)
        self.std_[self.std_ == 0 ] = 1.0
        return self

    def transform( self, X ):
        """ Apply the infered standardization to X. """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fit before transform")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit then transform in one call. Use for training data."""
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size = 0.2, random_state = None, stratify = None):
    """Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    test_size : float (0,1)
        Proportion of the dataset to include in the test split.
    random_state : int or None
        Seed for reproducibility.
    startify : ndarray of shape (n_samples,) or None
        If provided, splits are prduced such that each class in `startify`
        keeps the same proportion in train and test. Pass y here for startified splits.


    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1" )
    rng = np.random.default_rng(random_state)

    if stratify is None:
        # Simple shuffle-and-cut
        idx = rng.permutation(n)
        n_test = int(round(n * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
    else:
        # Stratified: split each class proportionally
        stratify = np.asarray(stratify)
        train_idx_list = []
        test_idx_list = []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            cls_idx = rng.permutation(cls_idx)
            n_test_cls = int(round(len(cls_idx) * test_size))
            test_idx_list.append(cls_idx[:n_test_cls])
            train_idx_list.append(cls_idx[n_test_cls:])
        test_idx = np.concatenate(test_idx_list)
        train_idx = np.concatenate(train_idx_list)
        # Re-shuffle so classes aren't grouped
        test_idx = rng.permutation(test_idx)
        train_idx = rng.permutation(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]






    

    
