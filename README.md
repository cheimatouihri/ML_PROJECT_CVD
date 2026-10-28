# ML_PROJECT_CVD
CLASS PROJECT 1 - ML COURSE @EPFL

### implementations.py - ML methods (step 2)
| Function | Details |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression using gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression using stochastic gradient descent |
| `least_squares(y, tx)` | Least squares regression using normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression using normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression using gradient descent (y ∈ {0, 1}) |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ‖w‖²) |

### eda.py - Exploratory data analysis

Run `python eda.py`. It cleans survey codes with the same rules as `preprocess()` and then reports:
- dataset shape, class balance and the accuracy of always predicting "no"
- missing-value rates and feature types (constant / binary / discrete / continuous)
- the top 15 non-nominal features by correlation with MICHD
- the top 10 columns whose missingness correlates with the label. This allows to check for leak features (100% CVD).

Figures are saved to `eda_output/`:
- `eda.png`: missing values per feature, top correlations, missingness vs. label
- `cvd_rates.png`: CVD rate by age group, self-rated health, number of risk factors and diabetes status

### preprocessing.py

`preprocess(x_train, x_test, names, max_missing=0.8, max_corr=0.9, indicators=True, clip=5)` returns the processed train and test matrices and the feature names. All statistics are computed on the training set only. Steps:
1. **Survey codes**: 7s/9s ("don't know"/"refused") and the codes in `EXTRA_CODES` become NaN, and 88s ("none") become 0. Calculated variables only lose a 1-digit 9.
2. **Feature removal**: drops irrelevant, duplicate or leaky columns (`TO_REMOVE`), columns more than 80% missing and constant columns.
3. **Missing indicators**: adds a 0/1 column for each feature that is more than 5% missing.
4. **Imputation**: fills with the mode for discrete features (≤ 15 values) and the median for continuous ones.
5. **Redundancy**: of any pair of features with pearson correlation |r| > 0.9, keeps only the first feature.
6. **Encoding**: standardizes numeric features and clips them to ±5, one-hot encodes nominal features (`NOMINAL`) and adds a bias column.

Run `python preprocessing.py` to print the output shapes and confirm no NaNs are left.

