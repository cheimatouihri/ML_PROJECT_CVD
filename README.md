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

### utils.py - helper functions used for model training and inference

Helper functions shared by the notebook and `run.py`:
- `load_preprocess()`: loads the data and runs `preprocess()`. Returns `tx_train, tx_test, y_train` (labels −1/1), `test_ids` and the feature names.
- `build_k_indices()`: shuffled row indices for k-fold cross-validation (from lab 4).
- `f1_score()`, `predict()`, `threshold_curve()`:
  - `f1_score()`: F1 of the positive class
  - `predict()`: −1/1 labels from scores and a threshold
  - `threshold_curve()`: F1 over a range of thresholds (quantiles 50–99% of the scores)
- `step_size()`: safe gradient descent step, 1 / (largest curvature of the loss). The curvature is estimated by power iteration and accounts for class weights and λ.
- `weighted_reg_logistic_regression()`: regularized logistic regression where each positive sample counts `pos_weight` times (labels 0/1).
- `random_undersample()`: keeps all positives and `und_ratio` × as many randomly chosen negatives.
- `add_powers()`: polynomial expansion x², …, x^d of the non-binary columns, standardized with training statistics.
- `add_interactions()`: pairwise products of the main risk factors (`INTERACTION_FEATURES`, chosen from the EDA), standardized with training statistics.

### model_comparison.ipynb - train and validate models and techniques to deal with class imbalance

Compares all models with 5-fold cross-validation. For each setting, report validation F1 and accuracy (mean ± std over the folds) and the train F1, both at the default threshold (score 0) and at a threshold tuned for F1 on the out-of-fold scores.

Models tried:
- least squares and ridge regression for λ ∈ {1e-5, …, 1e-1}.
- linear regression with GD and SGD
- logistic and regularized logistic regression for each λ
- class imbalance:
  - weighted logistic regression (positive weight 1.5, 2, 3)
  - random undersampling of the training folds
- feature expansion:
  - polynomial features (degree 2 and 3)
  - pairwise interactions of risk factors

Outputs, saved to `results/`:
- `cv_results.csv`: all settings and their scores
- `model_comparison.png`: best setting of each model
- `f1_vs_lambda.png`: train vs. validation F1 against λ (bias–variance)
- `f1_vs_threshold.png`: F1 against the decision threshold for the best model
- `top_weights.png`: largest logistic regression weights
- `caught_vs_missed.png`: how the CVD cases the model detects differ from the ones it misses

The last cell prints the best settings to copy into `run.py`. Set `QUICK = True` to run on a 50k-row subsample for quick checks.

### run.py - inference and submissions

reproduce the submitted predictions
1. loads and preprocesses the data (`load_preprocess()`)
2. optionally adds polynomial features (`DEGREE > 1`) and/or interaction features (`INTERACTIONS = True`)
3. trains the chosen model on the full training set
4. predicts the test set with the tuned threshold and writes `submission.csv` (labels −1/1)

The settings at the top of the script come from cross-validation in `model_comparison.ipynb` (last cell output):

| Parameter | Value | Meaning |
|---|---|---|
| `MODEL` | `"weighted reg logistic"` | one of `"ridge"`, `"reg logistic"`, `"weighted reg logistic"` |
| `LAMBDA` | `1e-4` | L2 regularization strength |
| `POS_WEIGHT` | `2` | weight of positive samples in the loss |
| `DEGREE` | `1` | polynomial degree (1 = no powers) |
| `INTERACTIONS` | `True` | add pairwise risk-factor interactions |
| `THRESHOLD` | `-0.601730` | decision threshold on the score `tx @ w`|
| `MAX_ITERS` | `1000` | gradient descent iterations |
