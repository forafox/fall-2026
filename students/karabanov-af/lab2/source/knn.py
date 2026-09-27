import numpy as np


def distances(X_train, X):
    """Euclidean distance from every object of X to every object of X_train: shape (len(X), len(X_train))."""
    return np.linalg.norm(X[:, None, :] - X_train[None, :, :], axis=2)


def gaussian(r):
    """Gaussian kernel K(r) = exp(-r^2 / 2): weight of a neighbour standing at r window widths away."""
    return np.exp(-0.5 * r ** 2)


def predict(X_train, y_train, X, k=5):
    """Majority vote of the k nearest neighbours."""
    neighbours = np.argsort(distances(X_train, X), axis=1)[:, :k]
    votes = y_train[neighbours]
    return np.array([np.bincount(row, minlength=y_train.max() + 1).argmax() for row in votes])


def predict_parzen(X_train, y_train, X, k=5):
    """Parzen window of variable width: the window h(x) is the distance to the (k + 1)-th neighbour.

    Every one of the k nearest neighbours votes with the weight K(rho / h) instead of a plain 1.
    """
    D = distances(X_train, X)
    order = np.argsort(D, axis=1)
    neighbours = order[:, :k]
    rho = np.take_along_axis(D, neighbours, axis=1)
    h = np.take_along_axis(D, order[:, k:k + 1], axis=1)
    weights = gaussian(rho / h)
    scores = np.array([(weights * (y_train[neighbours] == c)).sum(axis=1) for c in range(y_train.max() + 1)])
    return scores.argmax(axis=0)
