import numpy as np

# columns that are irrelevant or duplicate
TO_REMOVE = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "SEQNO", "_PSU",
             "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALCOR", "_LLCPWT",
             "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN",
             "FVORANG", "VEGETAB1", "EXEROFT1", "EXERHMM1", "EXEROFT2", "EXERHMM2", "STRENGTH",
             "FLSHTMY2", "HIVTSTD3", "EXRACT11", "EXRACT21", "HAREHAB1", 
             "HAREHAB1"] # HAREHAB1 = leaky feature, rehab after heart attack or stroke.

# codes that are not 7/9 and represent unknown value
EXTRA_CODES = {"_AGEG5YR": [14], "_AGE65YR": [3], "DROCDY3_": [900], "_DRNKWEK": [99900],
               "MAXVO2_": [999], "FC60_": [999]}

# categories without order
NOMINAL = ["_STATE", "MARITAL", "EMPLOY1", "RENTHOM1", "_RACE", "_PRACE1", "_MRACE1",
           "DIABETE3", "BPHIGH4"]

def preprocess(x_train, x_test, names, max_missing=0.8, max_corr=0.9, indicators=True, clip=5):
    x_tr, x_te = x_train.copy(), x_test.copy()

    # deal with survey codes
    for col, name in enumerate(names):
        values = x_tr[~np.isnan(x_tr[:, col]), col]
        if len(values) == 0 or not np.all(values == np.round(values)):
            continue
        digits = len(str(int(values.max())))
        calculated = name.startswith("_") or name.endswith("_") or name == "HTIN4"
        if calculated:
            codes = [9] if digits == 1 else []
        else:
            codes = [9] if name == "EMPLOY1" else [int("7" * digits), int("9" * digits)]
        codes += EXTRA_CODES.get(name, [])
        for data in (x_tr, x_te):
            data[np.isin(data[:, col], codes), col] = np.nan
            if not calculated and digits >= 2:
                data[data[:, col] == int("8" * digits), col] = 0

    # drop columns that are mostly missing or constant features
    missing = np.isnan(x_tr).mean(axis=0)
    n_unique = np.array([len(np.unique(column[~np.isnan(column)])) for column in x_tr.T])
    keep = [col for col, name in enumerate(names)
            if name not in TO_REMOVE and missing[col] <= max_missing and n_unique[col] > 1]  # fmt: skip
    x_tr, x_te, n_unique, missing = x_tr[:, keep], x_te[:, keep], n_unique[keep], missing[keep]
    names = [names[col] for col in keep]

    # add a 0/1 column telling if the value was missing (for features more than 5% missing)
    if indicators:
        partly = np.where(missing > 0.05)[0]
        x_tr = np.hstack([x_tr, np.isnan(x_tr[:, partly]).astype(float)])
        x_te = np.hstack([x_te, np.isnan(x_te[:, partly]).astype(float)])
        n_unique = np.concatenate([n_unique, np.full(len(partly), 2)])
        names = names + [names[col] + "_missing" for col in partly]

    # fill missing with mode for discrete features, median for continuous
    for col in range(x_tr.shape[1]):
        values = x_tr[~np.isnan(x_tr[:, col]), col]
        if n_unique[col] <= 15:
            options, counts = np.unique(values, return_counts=True)
            fill = options[counts.argmax()]
        else:
            fill = np.median(values)
        x_tr[np.isnan(x_tr[:, col]), col] = fill
        x_te[np.isnan(x_te[:, col]), col] = fill

    # of two almost identical features, keep only the first one
    corr = np.abs(np.corrcoef(x_tr[::5].T))
    drop = set()
    for first in range(len(names)):
        if first not in drop:
            drop.update(second for second in range(first + 1, len(names)) if corr[first, second] > max_corr)
    keep = [col for col in range(len(names)) if col not in drop]
    x_tr, x_te = x_tr[:, keep], x_te[:, keep]
    names = [names[col] for col in keep]

    # standardize numeric features (clipping extreme values), one-hot encode nominal ones, add bias
    nominal = [col for col, name in enumerate(names) if name in NOMINAL]
    numeric = [col for col in range(len(names)) if col not in nominal]
    mean, std = x_tr[:, numeric].mean(axis=0), x_tr[:, numeric].std(axis=0)
    parts_tr = [np.ones((len(x_tr), 1)), np.clip((x_tr[:, numeric] - mean) / std, -clip, clip)]
    parts_te = [np.ones((len(x_te), 1)), np.clip((x_te[:, numeric] - mean) / std, -clip, clip)]
    out_names = ["bias"] + [names[col] for col in numeric]
    for col in nominal:
        for category in np.unique(x_tr[:, col]):
            parts_tr.append((x_tr[:, [col]] == category).astype(float))
            parts_te.append((x_te[:, [col]] == category).astype(float))
            out_names.append(f"{names[col]}={int(category)}")

    return np.hstack(parts_tr), np.hstack(parts_te), out_names

if __name__ == "__main__":
    from helpers import load_csv_data

    x_train, x_test, y_train, _, _ = load_csv_data("data")
    names = open("data/x_train.csv").readline().strip().split(",")[1:]
    tx_train, tx_test, features = preprocess(x_train, x_test, names)
    print("train:", tx_train.shape, " test:", tx_test.shape)
    print("NaN left:", np.isnan(tx_train).sum() + np.isnan(tx_test).sum())
