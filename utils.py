import os
import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess
from implementations import sigmoid

DATA_DIR = "data"
INTERACTION_FEATURES = [
    "_AGEG5YR",
    "SEX",
    "GENHLTH",
    "PHYSHLTH",
    "_BMI5",
    "_RFHYPE5",
    "BPHIGH4=1",
    "TOLDHI2",
    "DIABETE3=1",
    "_SMOKER3",
    "CVDSTRK3",
    "CHCCOPD1",
    "CHCKIDNY",
    "DIFFWALK",
    "USEEQUIP",
    "QLACTLM2",
    "HAVARTH3",
    "EMPLOY1=7",
]

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
    
    for _ in range(iters):  
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

def random_undersample(x, y, und_ratio, seed):
    """Randomly undersample the majority class to reduce class imbalance.

    Keeps all minority samples plus und_ratio * (number of minority samples) majority
    samples drawn at random without replacement (capped at the size of the majority
    class). und_ratio = 1 gives a balanced dataset.
    """
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    min_label = labels[counts.argmin()]
    min_indices = np.where(y == min_label)[0]
    maj_indices = np.where(y != min_label)[0]

    n_keep = min(int(und_ratio * len(min_indices)), len(maj_indices))
    maj_keep = rng.choice(maj_indices, n_keep, replace=False)

    idx = rng.permutation(np.concatenate([maj_keep, min_indices]))
    return x[idx], y[idx]

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

def add_powers(tx_tr, features, degree, tx_te=None):
    """Polynomial feature expansion (as build_poly in lab 4), only for non-binary columns.

    Adds x^2, ..., x^degree for every column with more than 2 distinct values (numeric and
    ordinal features). 0/1 columns (one-hot, missing indicators) are skipped, since x^p = x.
    Each new column is standardized with the mean / std of the training data.

    Returns:
        tx_tr_poly, tx_te_poly (None if tx_te is None), new feature names
    """
    cols = [j for j in range(1, tx_tr.shape[1]) if len(np.unique(tx_tr[:, j])) > 2]
    parts_tr, parts_te, names = [tx_tr], [tx_te], list(features)
    for p in range(2, degree + 1):
        power_tr = tx_tr[:, cols] ** p
        mean, std = power_tr.mean(axis=0), power_tr.std(axis=0)
        std[std == 0] = 1
        parts_tr.append((power_tr - mean) / std)
        if tx_te is not None:
            parts_te.append((tx_te[:, cols] ** p - mean) / std)
        names += [f"{features[j]}^{p}" for j in cols]
    tx_te_poly = np.hstack(parts_te) if tx_te is not None else None
    return np.hstack(parts_tr), tx_te_poly, names

# Features combined pairwise by add_interactions(): age, sex, general health and the main
# cardiovascular risk factors / comorbidities (chosen from the EDA). 
# Names missing after preprocessing are skipped.

def add_interactions(tx_tr, features, names=INTERACTION_FEATURES, tx_te=None):
    """Interaction features: the product x_i * x_j for every pair of the listed features.

    A linear model can then learn effects that depend on another feature, e.g. that high
    blood pressure matters more at older ages. Each product is standardized with the mean /
    std of the training data.

    Returns:
        tx_tr_inter, tx_te_inter (None if tx_te is None), new feature names
    """
    idx = [features.index(n) for n in names if n in features]
    parts_tr, parts_te, new_names = [tx_tr], [tx_te], list(features)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            prod_tr = tx_tr[:, i] * tx_tr[:, j]
            mean, std = prod_tr.mean(), prod_tr.std()
            std = std if std > 0 else 1.0
            parts_tr.append(((prod_tr - mean) / std)[:, None])
            if tx_te is not None:
                parts_te.append(((tx_te[:, i] * tx_te[:, j] - mean) / std)[:, None])
            new_names.append(f"{features[i]}*{features[j]}")
    tx_te_inter = np.hstack(parts_te) if tx_te is not None else None
    return np.hstack(parts_tr), tx_te_inter, new_names