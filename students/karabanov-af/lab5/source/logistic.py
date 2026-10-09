import numpy as np


def sigmoid(z):
    """Logistic function 1 / (1 + exp(-z)), written through tanh so that exp() never overflows."""
    return 0.5 * (1 + np.tanh(0.5 * z))


def signed(y):
    """Labels {0, 1} as the margins {-1, +1} the lecture uses."""
    return 2 * y - 1


def margins(F, y, w):
    """M_i = y_i <w, x_i>: positive when the object is on its own side of the plane."""
    return y * (F @ w)


def without_bias(w):
    """Zero out the free term: it is the one weight that stays unregularized."""
    return np.concatenate([[0.0], w[1:]])


def log_loss(F, y, w, tau=0.0):
    """Q(w) = sum log(1 + exp(-M_i)) + tau/2 ||w||^2, computed without overflow on large margins."""
    return float(np.logaddexp(0, -margins(F, y, w)).sum() + 0.5 * tau * w[1:] @ w[1:])


def gradient(F, y, w, tau=0.0):
    """dQ/dw_j = -sum (1 - sigma_i) y_i f_j(x_i), where sigma_i = sigma(M_i)."""
    return -F.T @ ((1 - sigmoid(margins(F, y, w))) * y) + tau * without_bias(w)


def hessian(F, y, w, tau=0.0):
    """d2Q/dw_j dw_k = sum (1 - sigma_i) sigma_i f_j(x_i) f_k(x_i) = F.T D F."""
    sigma = sigmoid(margins(F, y, w))
    return F.T @ (((1 - sigma) * sigma)[:, None] * F) + tau * np.diag(without_bias(np.ones(len(w))))


def newton_raphson(F, y, tau=1.0, step=1.0, n_iterations=50, tolerance=1e-10):
    """w := w - h * inv(Q'') Q', the step the lecture writes out for the log-loss.

    Returns the whole trajectory of weights, starting from zeros: the caller reads
    the answer off its last row and the convergence off the rest.
    """
    trajectory = [np.zeros(F.shape[1])]
    for _ in range(n_iterations):
        w = trajectory[-1]
        trajectory.append(w - step * np.linalg.solve(hessian(F, y, w, tau), gradient(F, y, w, tau)))
        if abs(log_loss(F, y, trajectory[-2], tau) - log_loss(F, y, trajectory[-1], tau)) < tolerance:
            break
    return np.array(trajectory)


def probability(F, w):
    """Posterior probability of the positive class: P(y = +1 | x) = sigma(<w, x>)."""
    return sigmoid(F @ w)


def predict(F, w):
    return np.where(F @ w >= 0, 1, -1)


def accuracy(y_true, y_predicted):
    return float(np.mean(y_true == y_predicted))
