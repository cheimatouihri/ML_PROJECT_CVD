# ML_PROJECT_CVD
CLASS PROJECT 1 - ML COURSE @EPFL

### implementations.py
| Function | Details |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression using gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression using stochastic gradient descent |
| `least_squares(y, tx)` | Least squares regression using normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression using normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression using gradient descent (y ∈ {0, 1}) |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ‖w‖²) |

### eda.py

### preprocessing.py

- Replace missing and refused with NaN, and "none" (88, 888) with 0.
- Remove the columns for dates, IDs, survey weights, duplicate raw answers and features more than 80% missing and constant features
- Add a 0/1 column for features more than 5% missing, saying if the value was missing.
- Fill missing values with the most common value (discrete) or the median (continuous).
- Remove one feature of each pair with correlation above 0.9.
- Standardize and one-hot encode categories without order, add bias column 

