import numpy as np

# helpers
def calculate_mse(y, tx, w):
    """Calculate the mean squared error."""
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Returns:
        An array (same shape as w) containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad


# Methods
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
    return w, calculate_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)
        w = w - gamma * compute_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, calculate_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, calculate_mse(y, tx, w)
