"""Diagnostics of the best logistic model: why is the F1 score stuck around 0.43?

Reads the best setting from results/cv_results.csv and makes
four figures in results/:
  convergence.png                does gradient descent stop too early?
  scores_precision_recall.png    can the features separate the two classes?
  learning_curve.png             would more data help?
  by_age.png                     which people does the model miss?

Uses one train / validation split: validation = fold 1, training = folds 2-5
(same folds as the notebook).

Usage:  python diagnostics.py           (full data, about 10-20 min)
        python diagnostics.py --quick   (50,000 rows, for testing)
"""

import csv
import os
import sys
import numpy as np
import matplotlib

#matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt

from implementations import calculate_neg_ll_loss, sigmoid
from utils import (
    load_preprocess,
    build_k_indices,
    f1_score,
    predict,
    threshold_curve,
    step_size,
    weighted_reg_logistic_regression,
    add_powers,
)

K = 5  # same folds as the notebook
MAX_ITERS = 1000 
RESULTS = "results/cv_results.csv"
QUICK = "--quick" in sys.argv


def best_logistic_setting(path):
    """Best (lambda, pos_weight, degree) among the logistic models in cv_results.csv.

    Undersampled models are skipped (they use a different training set). Falls back to
    lambda=1e-4, pos_weight=2, degree=1 (best full-data run) if the file is missing."""
    if not os.path.exists(path):
        print(f"{path} not found, using lambda=1e-4, pos_weight=2, degree=1")
        return 1e-4, 2.0, 1
    with open(path) as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if "logistic" in r["model"] and "undersampled" not in r["model"]
        ]
    best = max(rows, key=lambda r: float(r["val_f1"]))
    s = dict(p.split("=") for p in best["param"].split() if "=" in p)
    print(
        f"best logistic setting in {path}: {best['model']} {best['param']} (val F1 {best['val_f1']})"
    )
    return float(s.get("lambda", 0)), float(s.get("pw", 1)), int(s.get("degree", 1))


os.makedirs("results", exist_ok=True)
LAM, PW, DEGREE = best_logistic_setting(RESULTS)

tx, _, y, _, features = load_preprocess()
if QUICK:
    keep = np.random.default_rng(0).choice(len(y), min(50_000, len(y)), replace=False)
    tx, y = tx[keep], y[keep]
base_features = features
if DEGREE > 1:
    tx, _, features = add_powers(tx, features, DEGREE)
y01 = (y == 1).astype(float)
N, D = tx.shape
k_indices = build_k_indices(y, K, seed=1)
te = k_indices[0]
tr = k_indices[1:].reshape(-1)
g = step_size(tx, "logistic", LAM, y01, PW)
print(
    f"N={N}  D={D}  lambda={LAM:g}  pos_weight={PW:g}  degree={DEGREE}  quick mode: {QUICK}"
)

# ---------- 1. convergence ----------
# Train in blocks of 100 steps (same updates as one long run) and record the losses and the
# validation F1 (best threshold) after each block.
w = np.zeros(D)
# at w = 0 every probability is 0.5, so the loss is log(2) and everyone is predicted positive
steps, loss_tr, loss_te = [0], [np.log(2)], [np.log(2)]
f1_te = [f1_score(y[te], np.ones(len(te)))]
for block in range(30):  # 30 x 100 = 3000 steps
    w = weighted_reg_logistic_regression(y01[tr], tx[tr], LAM, w, 100, g, PW)
    steps.append((block + 1) * 100)
    loss_tr.append(calculate_neg_ll_loss(y01[tr], tx[tr], w))
    loss_te.append(calculate_neg_ll_loss(y01[te], tx[te], w))
    f1_te.append(threshold_curve(tx[te] @ w, y[te])[1].max())

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(steps, loss_tr, label="train")
ax[0].plot(steps, loss_te, label="validation")
ax[0].set_xlabel("gradient descent steps")
ax[0].set_ylabel("negative log-likelihood")
ax[0].legend()
ax[1].plot(steps, f1_te, color="tab:orange")
ax[1].axvline(MAX_ITERS, color="gray", linestyle="--", label=f"MAX_ITERS = {MAX_ITERS}")
ax[1].set_xlabel("gradient descent steps")
ax[1].set_ylabel("validation F1 (best threshold)")
ax[1].legend()
fig.suptitle("Convergence of gradient descent")
fig.tight_layout()
fig.savefig("results/convergence.png", dpi=150)
plt.close(fig)
print(
    f"\n1. validation F1 after {MAX_ITERS} steps: {f1_te[MAX_ITERS // 100]:.4f}, after 3000: {f1_te[-1]:.4f}"
)

# ---------- 2. predicted probabilities and precision-recall (model w after 3000 steps) ----------
scores = tx[te] @ w
p = sigmoid(scores)
thresholds, f1s = threshold_curve(scores, y[te])
best = np.argmax(f1s)
t_best = thresholds[best]

precision, recall = [], []
for t in thresholds:
    pred = predict(scores, t)
    tp = np.sum((pred == 1) & (y[te] == 1))
    precision.append(tp / max(np.sum(pred == 1), 1))
    recall.append(tp / np.sum(y[te] == 1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0, 1, 41)
ax[0].hist(p[y[te] == -1], bins=bins, density=True, alpha=0.6, label="no CVD")
ax[0].hist(p[y[te] == 1], bins=bins, density=True, alpha=0.6, label="CVD")
ax[0].axvline(
    sigmoid(t_best),
    color="black",
    linestyle="--",
    label=f"best threshold ({sigmoid(t_best):.2f})",
)
ax[0].set_xlabel("predicted probability of CVD")
ax[0].set_ylabel("density")
ax[0].set_title("Predicted probabilities by true class")
ax[0].legend()
ax[1].plot(recall, precision)
ax[1].scatter(
    recall[best],
    precision[best],
    color="tab:red",
    zorder=3,
    label=f"best F1 = {f1s[best]:.3f}",
)
ax[1].set_xlabel("recall (share of CVD cases caught)")
ax[1].set_ylabel("precision (share of alarms that are CVD)")
ax[1].set_title("Precision-recall trade-off")
ax[1].legend()
fig.tight_layout()
fig.savefig("results/scores_precision_recall.png", dpi=150)
plt.close(fig)
print(
    f"2. at the best threshold: precision {precision[best]:.1%}, recall {recall[best]:.1%}"
)

# ---------- 3. learning curve ----------
# Train on growing subsets of the training folds. The threshold is tuned on the training subset
# itself, so the validation F1 is honest.
sizes = [n for n in [2_000, 5_000, 10_000, 25_000, 50_000, 100_000] if n < len(tr)] + [
    len(tr)
]
rng = np.random.default_rng(0)
lc_tr, lc_te = [], []
print("3. learning curve")
for n in sizes:
    sub = rng.choice(tr, n, replace=False)
    w_n = weighted_reg_logistic_regression(
        y01[sub], tx[sub], LAM, np.zeros(D), MAX_ITERS, g, PW
    )
    th, f1_sub = threshold_curve(tx[sub] @ w_n, y[sub])
    lc_tr.append(f1_sub.max())
    lc_te.append(f1_score(y[te], predict(tx[te] @ w_n, th[np.argmax(f1_sub)])))
    print(f"   n={n:>7}  train F1 {lc_tr[-1]:.4f}  validation F1 {lc_te[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(sizes, lc_tr, "o-", label="train")
ax.semilogx(sizes, lc_te, "o-", label="validation")
ax.set_xlabel("number of training samples")
ax.set_ylabel("F1")
ax.set_title("Learning curve")
ax.legend()
fig.tight_layout()
fig.savefig("results/learning_curve.png", dpi=150)
plt.close(fig)

# ---------- 4. performance by age group ----------
age = tx[te, base_features.index("_AGEG5YR")]
groups = np.unique(age)  # standardized values, in increasing age order
labels = ["18-24"] + [f"{a}-{a + 4}" for a in range(25, 80, 5)] + ["80+"]
labels = (
    labels if len(groups) == len(labels) else [str(i + 1) for i in range(len(groups))]
)
pred = predict(scores, t_best)

rec, prec, share = [], [], []
for a in groups:
    m = age == a
    tp = np.sum((pred[m] == 1) & (y[te][m] == 1))
    rec.append(tp / max(np.sum(y[te][m] == 1), 1))
    prec.append(tp / max(np.sum(pred[m] == 1), 1))
    share.append(np.mean(y[te][m] == 1))

x = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - 0.27, rec, 0.27, label="recall")
ax.bar(x, prec, 0.27, label="precision")
ax.bar(x + 0.27, share, 0.27, label="share of CVD in the group")
ax.set_xticks(x, labels, rotation=45, ha="right")
ax.set_xlabel("age group")
ax.set_title("Model performance by age group (validation fold)")
ax.legend()
fig.tight_layout()
fig.savefig("results/by_age.png", dpi=150)
plt.close(fig)
print(
    "4. recall by age group: " + "  ".join(f"{l} {r:.0%}" for l, r in zip(labels, rec))
)

print(
    "\nsaved results/convergence.png, scores_precision_recall.png, learning_curve.png, by_age.png"
)