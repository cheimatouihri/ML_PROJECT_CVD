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


def ridge_regression(y, tx, lambda_):
    """ Implement ridge regression using normal equations. """
    # Lecture 03d, slide 6
    # Solve system of linear equations to get optimal w:
    # (X^T * X + lamba * I) * w = X^T * y
    # Aw = b
    # A = (X^T * X + lamba * I) * w
    # b = X^T * y

    N = tx.shape[0]
    D = tx.shape[1]
    tx_t = np.transpose(tx)  # (D, N)
    I = np.identity(D)
    lambda_prime = lambda_ * 2 * N
    A = (tx_t @ tx) + (lambda_prime * I)  # (D, D)
    b = tx_t @ y  # (D, 1)
    # vector/array of optimal weights from ridge regression
    w = np.linalg.solve(A, b)
    # per project instructions, penalty term NOT included in loss calculation
    loss = calculate_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """ Compute output of logistic (sigmoid) function on t. """
    # (1 + e^-x)^-1
    sigma = (1 / 1 + np.exp(-t))**-1
    return sigma


def calculate_neg_ll_loss(y, tx, w):
    """ Compute negative log-likelihood loss. """
    sigma = sigmoid(np.dot(tx, w))
    # internal negative log-likelihood calculation for each sample n
    neg_ll = np.multiply(y, np.log(sigma)) + np.multiply(1 - y, np.log(1 - sigma))
    # take average and make negative
    neg_ll = np.mean(neg_ll) * -1
    return neg_ll.item()


def calculate_neg_ll_gradient(y, tx, w):
    """ Compute the gradient of negative log-likelihood loss, w.r.t. w
        (lecture 05b, slide 18). """
    N = tx.shape[0]
    sigma = sigmoid(np.dot(tx, w))  # N x 1
    tx_t = np.transpose(tx)  # D x N
    gradient = (np.dot(tx_t, (sigma - y))) / N  # D x 1
    return gradient


def learning_by_gradient_descent(y, tx, w, gamma, reg=False, lambda_=0):
    """ Do one step of gradient descent using logistic regression. Return the loss and the updated w. """
    gradient = calculate_neg_ll_gradient(y, tx, w)
    loss = calculate_neg_ll_loss(y, tx, w)
    # reg argument indicates whether to use regularization
    if reg:  # yes, regularize
        gradient += (2 * lambda_ * w)
    w -= (gamma * gradient)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Perform logistic regression using gradient descent.
        No regularization on this version. """
    w = initial_w
    # iterative logistic regression by gradient descent
    for _ in range(max_iters):
        # get loss and update weights w
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Perform logistic regression using gradient descent, with regularization term
        to help avoid overfitting. """
    w = initial_w
    # iterative logistic regression by gradient descent
    for _ in range(max_iters):
        # get loss and update weights w
        # per project instructions, penalty term NOT included in loss calculation
        w, loss = learning_by_gradient_descent(y, tx, w, gamma, reg=True, lambda_=lambda_)
    return w, loss
