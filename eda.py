import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data

os.makedirs("eda_output", exist_ok=True)
names = open("data/x_train.csv").readline().strip().split(",")[1:]

x_train, x_test, y_train, _, _ = load_csv_data("data")
y = (y_train == 1).astype(float)

# survey codes 7/77/777 - don't know, 9/99/999 - refused
X = x_train.copy()
for col in range(X.shape[1]):
    values = X[~np.isnan(X[:, col]), col]
    if len(values) and np.all(values == np.round(values)):
        digits = len(str(int(values.max())))
        X[np.isin(X[:, col], [int("7" * digits), int("9" * digits)]), col] = np.nan

missing = np.isnan(X).mean(axis=0)
n_unique = np.array([len(np.unique(column[~np.isnan(column)])) for column in X.T])
corr = np.array(
    [
        np.corrcoef(column[~np.isnan(column)], y[~np.isnan(column)])[0, 1] if count > 1 else 0
        for column, count in zip(X.T, n_unique)
    ]
)

print("train shape:", x_train.shape)
print(f"positives: {y.mean():.1%}")
print("features with survey codes:", int((missing > np.isnan(x_train).mean(axis=0)).sum()))
print("features missing > 50% / 80%:", int((missing > 0.5).sum()), "/", int((missing > 0.8).sum()))
print("constant:", int((n_unique <= 1).sum()), " binary:", int((n_unique == 2).sum()),
      " discrete (3-15):", int(((n_unique > 2) & (n_unique <= 15)).sum()), " continuous:", int((n_unique > 15).sum()))  # fmt: skip

top = np.argsort(-np.abs(np.where(missing < 0.5, corr, 0)))[:15]
print("\ntop 15 features (< 50% missing) by correlation with the label:")
for col in top:
    print(f"  {names[col]:<10} r={corr[col]:+.3f}  missing {missing[col]:.0%}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(missing * 100, bins=20, range=(0, 100))
ax[0].set_xlabel("% missing")
ax[0].set_ylabel("number of features")
ax[1].barh([names[col] for col in top[::-1]], np.abs(corr[top[::-1]]))
ax[1].set_xlabel("|correlation| with MICHD")
fig.tight_layout()
fig.savefig("eda_output/eda.png", dpi=150)
