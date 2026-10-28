import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data
from preprocessing import EXTRA_CODES, NOMINAL

os.makedirs("eda_output", exist_ok=True)
names = open("data/x_train.csv").readline().strip().split(",")[1:]
col = {name: i for i, name in enumerate(names)}

x_train, x_test, y_train, _, _ = load_csv_data("data")
y = (y_train == 1).astype(float)


def replace_codes(x, names):
    """ 7s/9s -> NaN (don't know / refused), 88 -> 0 (none),
    calculated variables (_X, X_) only lose a 1-digit 9, plus the EXTRA_CODES."""
    x = x.copy()
    for c, name in enumerate(names):
        values = x[~np.isnan(x[:, c]), c]
        if len(values) == 0 or not np.all(values == np.round(values)):
            continue
        digits = len(str(int(values.max())))
        calculated = name.startswith("_") or name.endswith("_") or name == "HTIN4"
        if calculated:
            codes = [9] if digits == 1 else []
        else:
            codes = [9] if name == "EMPLOY1" else [int("7" * digits), int("9" * digits)]
        codes += EXTRA_CODES.get(name, [])
        x[np.isin(x[:, c], codes), c] = np.nan
        if not calculated and digits >= 2:
            x[x[:, c] == int("8" * digits), c] = 0
    return x


def calculate_corr(v):
    """Pearson r between a column and the label, on rows where the column is present."""
    present = ~np.isnan(v)
    if present.sum() < 2 or v[present].std() == 0 or y[present].std() == 0:
        return 0.0
    return np.corrcoef(v[present], y[present])[0, 1]


X = replace_codes(x_train, names)
missing = np.isnan(X).mean(axis=0)
n_unique = np.array([len(np.unique(column[~np.isnan(column)])) for column in X.T])
nominal = np.isin(names, NOMINAL) & ~np.isin(names, ["DIABETE3", "BPHIGH4"]) 
corr = np.array([calculate_corr(column) for column in X.T])

partly = (missing > 0.01) & (missing < 0.99)
miss_corr = np.array([np.corrcoef(np.isnan(column), y)[0, 1] if p else 0 for column, p in zip(X.T, partly)])

print("train shape:", x_train.shape)
print(f"positives: {y.mean():.1%}  (always predicting 'no' gives {1 - y.mean():.1%} accuracy)")
print("features with survey codes:", int((missing > np.isnan(x_train).mean(axis=0)).sum()))
print("features missing > 50% / 80%:", int((missing > 0.5).sum()), "/", int((missing > 0.8).sum()))
print("constant:", int((n_unique <= 1).sum()), " binary:", int((n_unique == 2).sum()),
      " discrete (3-15):", int(((n_unique > 2) & (n_unique <= 15)).sum()), " continuous:", int((n_unique > 15).sum())) 

top = np.argsort(-np.abs(np.where((missing < 0.5) & ~nominal, corr, 0)))[:15]
print("\ntop 15 features (< 50% missing, not nominal) by correlation with the label:")
for c in top:
    print(f"  {names[c]:<10} r={corr[c]:+.3f}  missing {missing[c]:.0%}")

top_miss = np.argsort(-np.abs(miss_corr))[:10]
print("\ntop 10 columns whose missingness correlates with the label (to check for leaks):")
for c in top_miss:
    print(f"  {names[c]:<10} r={miss_corr[c]:+.3f}  missing {missing[c]:.0%}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(missing * 100, bins=20, range=(0, 100))
ax[0].set_xlabel("% missing")
ax[0].set_ylabel("number of features")
ax[0].set_title("Missing values per feature")

for a, values, idx, title in [(ax[1], corr, top, "Top features (red = higher CVD rate)"),
                              (ax[2], miss_corr, top_miss, "Missingness vs. label")]:  # fmt: skip
    order = idx[::-1]
    a.barh([names[c] for c in order], values[order], color=np.where(values[order] > 0, "tab:red", "tab:blue"))
    a.axvline(0, color="black", linewidth=0.8)
    a.set_xlabel("correlation with MICHD")
    a.set_title(title)
fig.tight_layout()
fig.savefig("eda_output/eda.png", dpi=150)


# CVD rate per category 
def rate_plot(a, values, title, labels=None):
    present = ~np.isnan(values)
    cats = np.unique(values[present])
    rates = [y[present][values[present] == cat].mean() * 100 for cat in cats]
    ticks = labels if labels is not None and len(labels) == len(cats) else [f"{cat:g}" for cat in cats]
    a.bar(range(len(cats)), rates, color="tab:red", alpha=0.8)
    a.set_xticks(range(len(cats)), ticks, rotation=45 if len(cats) > 6 else 0, ha="right" if len(cats) > 6 else "center")
    a.set_ylabel("CVD rate (%)")
    a.set_title(title)


# risk-factor count
factors = {"_RFHYPE5": 2,   # high blood pressure (2 = yes)
           "_RFCHOL": 2,    # high cholesterol
           "DIABETE3": 1,   # diabetes (1 = yes)
           "_RFSMOK3": 2,   # current smoker
           "_RFBMI5": 2,    # obese (BMI > 30)
           "_TOTINDA": 2}   # no physical activity in the past month (2 = inactive)
values = X[:, [col[n] for n in factors]]
known = ~np.isnan(values).any(axis=1)
risk = np.where(known, (values == list(factors.values())).sum(axis=1), np.nan)
print(f"risk-factor plot uses {known.mean():.0%} of people (all {len(factors)} answers known)")

ages = ["18-24"] + [f"{a}-{a + 4}" for a in range(25, 80, 5)] + ["80+"]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
rate_plot(ax[0, 0], X[:, col["_AGEG5YR"]], "CVD rate by age group", ages)
rate_plot(ax[0, 1], X[:, col["GENHLTH"]], "CVD rate by self-rated health",
          ["excellent", "very good", "good", "fair", "poor"])
rate_plot(ax[1, 0], risk, "CVD rate by number of risk factors\n(hypertension, cholesterol, diabetes, smoking, obesity, inactivity)")
rate_plot(ax[1, 1], X[:, col["DIABETE3"]], "CVD rate by DIABETE3 code (not ordinal)",
          ["yes", "pregnancy only", "no", "prediabetes"]) 
fig.tight_layout()
fig.savefig("eda_output/cvd_rates.png", dpi=150)
print("\nsaved eda_output/eda.png and eda_output/cvd_rates.png")