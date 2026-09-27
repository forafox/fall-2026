import numpy as np


def distances(X_train, X):
    """Euclidean distance from every object of X to every object of X_train: shape (len(X), len(X_train))."""
    return np.linalg.norm(X[:, None, :] - X_train[None, :, :], axis=2)


def predict(X_train, y_train, X, k=5):
    """Majority vote of the k nearest neighbours."""
    neighbours = np.argsort(distances(X_train, X), axis=1)[:, :k]
    votes = y_train[neighbours]
    return np.array([np.bincount(row).argmax() for row in votes])
