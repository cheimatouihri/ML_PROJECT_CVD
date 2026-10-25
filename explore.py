"""Quick sanity checks on the BRFSS CVD data.

Run all checks:        python explore.py
Inspect some columns:  python explore.py GENHLTH _AGEG5YR DIABETE3
"""
import sys
import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess 

DATA = "data"
KEY_VARS = ["_AGEG5YR", "SEX", "GENHLTH", "BPHIGH4", "TOLDHI2", "DIABETE3",
            "CVDSTRK3", "_SMOKER3", "_BMI5CAT", "PHYSHLTH"]
LEAK_SUSPECTS = ["HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF", "RLIVPAIN", "RDUCHART", "RDUCSTRK"]

x_train, x_test, y_train, _, _ = load_csv_data(DATA)
names = open(f"{DATA}/x_train.csv").readline().strip().split(",")[1:]
col = {name: i for i, name in enumerate(names)}
pos = y_train == 1  # works whether labels are {-1, 1} or {0, 1}


def section(title):
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def describe(name, top=12):
    """Value counts of one raw column, with the CVD rate for each value."""
    values = x_train[:, col[name]]
    present = ~np.isnan(values)
    v, n = np.unique(values[present], return_counts=True)
    extra = f" (top {top} shown)" if len(v) > top else ""
    print(f"\n{name}: {1 - present.mean():.1%} missing, {len(v)} distinct values{extra}")
    for value, count in sorted(zip(v, n), key=lambda t: -t[1])[:top]:
        rate = pos[present][values[present] == value].mean()
        print(f"  {value:>10g}   n={count:>7}   CVD rate={rate:6.1%}")
    if not present.all():
        print(f"  {'NaN':>10}   n={(~present).sum():>7}   CVD rate={pos[~present].mean():6.1%}")


def missing_vs_label(name):
    """Share missing, and CVD rate when the column is present vs missing."""
    missing = np.isnan(x_train[:, col[name]])
    present_rate = pos[~missing].mean() if (~missing).any() else np.nan
    missing_rate = pos[missing].mean() if missing.any() else np.nan
    return name, missing.mean(), present_rate, missing_rate


def print_missing_rows(rows):
    print(f"  {'column':<12} {'missing':>8} {'CVD|present':>12} {'CVD|missing':>12}")
    for name, share, present_rate, missing_rate in rows:
        print(f"  {name:<12} {share:>8.1%} {present_rate:>12.1%} {missing_rate:>12.1%}")


def correlation_with_label(x, feature_names, top=20):
    """Pearson correlation of each column with the 0/1 label (NaN rows skipped per column)."""
    target = pos.astype(float)
    scores = []
    for i, name in enumerate(feature_names):
        present = ~np.isnan(x[:, i])
        if present.sum() < 100:
            continue
        a, b = x[present, i], target[present]
        if a.std() == 0 or b.std() == 0:
            continue
        scores.append((name, np.corrcoef(a, b)[0, 1], present.mean()))
    scores.sort(key=lambda s: -abs(s[1]))
    for name, r, coverage in scores[:top]:
        print(f"  {name:<24} r={r:+.3f}   present={coverage:.0%}")


if len(sys.argv) > 1:
    for name in sys.argv[1:]:
        describe(name) if name in col else print(f"\n{name}: not a column")
    sys.exit()

section("Shapes and labels")
print(f"x_train {x_train.shape}   x_test {x_test.shape}   {len(names)} feature names")
assert x_train.shape[1] == len(names) == x_test.shape[1], "names don't line up with columns"
labels, counts = np.unique(y_train, return_counts=True)
print("labels:", dict(zip(labels.tolist(), counts.tolist())), f"-> positive rate {pos.mean():.2%}")
print(f"accuracy of always predicting 'no CVD': {1 - pos.mean():.2%}  (why accuracy is a bad metric here)")

# missing
section("Missing values per column")
miss_tr, miss_te = np.isnan(x_train).mean(axis=0), np.isnan(x_test).mean(axis=0)
buckets = [("none", miss_tr == 0), ("<=5%", (miss_tr > 0) & (miss_tr <= 0.05)),
           ("5-50%", (miss_tr > 0.05) & (miss_tr <= 0.5)), ("50-80%", (miss_tr > 0.5) & (miss_tr <= 0.8)),
           (">80%", miss_tr > 0.8)]
for label, mask in buckets:
    print(f"  {label:>7}: {mask.sum():>3} columns")
print("biggest train/test missingness gaps (should all be small):")
for i in np.argsort(-np.abs(miss_tr - miss_te))[:5]:
    print(f"  {names[i]:<12} train {miss_tr[i]:6.1%}   test {miss_te[i]:6.1%}")

# survey codes
section("Survey codes in the raw data")
for code in [77, 88, 99, 777, 888, 999, 7777, 9999]:
    hits = [names[i] for i in range(len(names)) if np.any(x_train[:, i] == code)]
    print(f"  {code:>5}: {len(hits):>3} columns, e.g. {', '.join(hits[:6])}")

# leakage
section("Leak check: is missingness itself predictive?")
rows = [missing_vs_label(n) for n in names if 0.001 < miss_tr[col[n]] < 0.999]
rows.sort(key=lambda r: -abs(r[2] - r[3]))
print("columns where 'missing or not' changes the CVD rate the most:")
print_missing_rows(rows[:15])
print("\nknown suspects (CVD rate near 100% when present = leak):")
print_missing_rows([missing_vs_label(n) for n in LEAK_SUSPECTS if n in col])

# key variables
section("Key variables: value counts and CVD rate per value")
for name in KEY_VARS:
    if name in col:
        describe(name)

section("Raw columns most correlated with the label (codes not cleaned, rough)")
correlation_with_label(x_train, names)

# preprocessing
tx_tr, tx_te, feats = preprocess(x_train, x_test, names)
print(f"train {tx_tr.shape}   test {tx_te.shape}   {len(feats)} feature names")
print("NaN:", int(np.isnan(tx_tr).sum() + np.isnan(tx_te).sum()),
      "  inf:", int(np.isinf(tx_tr).sum() + np.isinf(tx_te).sum()))
std = tx_tr.std(axis=0)
print("constant columns (besides bias):", [f for f, s in zip(feats, std) if s == 0 and f != "bias"])

leaky = [f for f in feats if f.split("=")[0].removesuffix("_missing") in LEAK_SUSPECTS]
print("leak suspects that survived:", leaky or "none")

varying = std > 0
drift = np.abs(tx_te.mean(axis=0) - tx_tr.mean(axis=0))[varying] / std[varying]
print("biggest train/test mean shifts (in train std units, > 0.1 worth a look):")
for i in np.argsort(-drift)[:5]:
    print(f"  {np.array(feats)[varying][i]:<24} {drift[i]:.3f}")

print("\nprocessed features most correlated with the label:")
correlation_with_label(tx_tr, feats)