import numpy as np
from helpers import create_csv_submission
from implementations import ridge_regression, reg_logistic_regression
from utils import (
    load_preprocess,
    step_size,
    weighted_reg_logistic_regression,
    predict,
)

# parameters chosen by cross-validation
MODEL = "reg logistic"  
LAMBDA = 1e-4  
POS_WEIGHT = 1  # weight of positive samples
THRESHOLD = 0.0 
MAX_ITERS = 500  
OUTPUT = "submission.csv"

tx_train, tx_test, y_train, test_ids, _ = load_preprocess(use_cache=False)
y01 = (y_train == 1).astype(float) # convert labels to 0/1 for logistic regression
w0 = np.zeros(tx_train.shape[1])

if MODEL == "ridge":
    w, _ = ridge_regression(y_train, tx_train, LAMBDA)  # labels -1/1
else:
    gamma = step_size(tx_train, "logistic", LAMBDA, y01, POS_WEIGHT)
    if MODEL == "reg logistic":
        w, _ = reg_logistic_regression(y01, tx_train, LAMBDA, w0, MAX_ITERS, gamma)
    elif MODEL == "weighted reg logistic":
        w = weighted_reg_logistic_regression(y01, tx_train, LAMBDA, w0, MAX_ITERS, gamma, POS_WEIGHT)
    else:
        raise ValueError(f"unknown MODEL {MODEL!r}")

pred = predict(tx_test @ w, THRESHOLD)  # labels -1/1, as required for the submission
create_csv_submission(test_ids, pred, OUTPUT)
print(f"{MODEL} (lambda={LAMBDA}, pos_weight={POS_WEIGHT}, threshold={THRESHOLD})")
print(
    f"predicted positive: {np.mean(pred == 1):.1%} of the test set  ->  saved {OUTPUT}"
)