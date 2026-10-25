import os
import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess
from implementations import sigmoid

DATA_DIR = "data"

def load_preprocess(use_cache=True):
    """Return tx_train, tx_test, y_train (labels -1/1), test_ids, feature names.
    """
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)
    names = open(f"{DATA_DIR}/x_train.csv").readline().strip().split(",")[1:]
    tx_train, tx_test, features = preprocess(x_train, x_test, names)
    return tx_train, tx_test, y_train, test_ids, features

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold (from lab 4).

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold.
        If N is not divisible by k_fold, the last N % k_fold shuffled rows are left out.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def f1_score(y_true, y_pred):
    """F1 score of the positive class (1)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true != 1))
    fn = np.sum((y_pred != 1) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0

def predict(scores, threshold):
    """Labels -1/1 from model scores (tx @ w): 1 if the score is above the threshold."""
    return np.where(scores > threshold, 1, -1)

def threshold_curve(scores, y):
    """F1 for a range of thresholds
       Returns (thresholds, f1 values)
    """
    thresholds = np.quantile(scores, np.linspace(0.50, 0.99, 99))
    return thresholds, np.array([f1_score(y, predict(scores, t)) for t in thresholds])

def step_size(tx, loss, lambda_=0.0, y01=None, pos_weight=1.0, iters=30, seed=0):
    """Safe gradient descent step size: gamma = 1 / (largest curvature of the loss).
    """
    v = np.random.default_rng(seed).normal(size=tx.shape[1])
    
    for _ in range(iters):  # power iteration
        v = tx.T @ (tx @ v) / len(tx)
        L = np.linalg.norm(v)
        v /= L

    if loss == "mse":
        curvature = L
    elif loss == "logistic":
        # largest sample weight after normalizing the weights to mean 1 (1 without class weights)
        c_max = (
            1.0
            if pos_weight == 1
            else pos_weight / (1 + (pos_weight - 1) * np.mean(y01))
        )
        curvature = c_max * L / 4
    else:
        raise ValueError(f"unknown loss {loss!r}")
    return 1 / (curvature + 2 * lambda_)

def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, pos_weight
):
    """reg_logistic_regression (y in {0, 1}) where each positive sample counts pos_weight
    times in the loss. Counters the class imbalance (only ~9% positives).
    c_n = pos_weight for positives, 1 for negatives (normalized to mean 1).
    pos_weight = 1 gives exactly reg_logistic_regression."""
    c = np.where(y == 1, pos_weight, 1.0)
    c = c / c.mean()
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        gradient = tx.T @ (c * (sigmoid(tx @ w) - y)) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w