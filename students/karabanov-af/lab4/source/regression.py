import numpy as np


def fit(X, y, alpha=0.0, n_components=None):
    """Linear regression through the SVD of X: w = V @ diag(f / s) @ U.T @ y.

    The filter f_j = s_j^2 / (s_j^2 + alpha) is what makes the three methods differ:
    alpha = 0 and all components give plain least squares, alpha > 0 gives ridge regression,
    and keeping only the first n_components gives regression on the principal components.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    filters = s ** 2 / (s ** 2 + alpha)
    if n_components is not None:
        filters[n_components:] = 0
    intercept = y.mean()
    return Vt.T @ (filters / s * (U.T @ (y - intercept))), intercept


def predict(X, weights, intercept):
    return X @ weights + intercept


def r2(y_true, y_pred):
    """Share of the target variance explained by the model."""
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
