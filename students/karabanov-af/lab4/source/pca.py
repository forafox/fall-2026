import numpy as np


def fit(X):
    """PCA through the SVD of the centered matrix: X - mean = U @ diag(s) @ Vt.

    The rows of Vt are the principal axes, ordered by the singular values s.
    """
    mean = X.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt, singular_values


def explained_variance(singular_values, n_objects):
    """Variance along every principal axis: s_j^2 / (n - 1)."""
    return singular_values ** 2 / (n_objects - 1)


def explained_variance_ratio(singular_values):
    """Share of the total variance that every principal axis carries."""
    return singular_values ** 2 / np.sum(singular_values ** 2)


def transform(X, mean, components, k=None):
    """Coordinates of the objects in the basis of the first k principal axes."""
    return (X - mean) @ components[:k].T


def inverse_transform(P, mean, components):
    """Back to the feature space from the k coordinates kept in P."""
    return P @ components[:P.shape[1]] + mean


def effective_dimension(ratio, threshold=0.95):
    """How many axes are needed to keep the given share of the total variance."""
    return int(np.searchsorted(np.cumsum(ratio), threshold) + 1)


def broken_stick(d):
    """Expected share of the j-th axis if the variance were split between d axes at random."""
    return np.array([np.sum(1 / np.arange(j, d + 1)) / d for j in range(1, d + 1)])
