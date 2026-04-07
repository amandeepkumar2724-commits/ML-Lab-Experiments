#  Aim:
#  Write a code to generate a regression line for multiple
#  input features and compute: MAE, MSE, RMSE, R², Adj R².
#
#  Dataset:
#  X  | 1  2  3
#  Y1 | 2  4  6
#  Y2 | 3  6  7
#
#  Method : Matrix Factorization
#  Formula: B = (X'X)^-1 * X'y
#  Line   : y = b0 + b1*x1 + b2*x2
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Input Data ───────────────────────────────────────────────
X_original = np.array([1, 2, 3])
y1         = np.array([2, 4, 6])
y2         = np.array([3, 6, 7])

# Design matrix — prepend a column of ones for the bias term
X_design = np.column_stack([np.ones(len(X_original)), X_original])

# ── Helper: Fit & Evaluate ───────────────────────────────────
def fit_regression(X, y, target_name):
    # Normal equation: B = (X'X)^-1 * X'y
    B      = np.linalg.inv(X.T @ X) @ X.T @ y
    y_pred = X @ B

    mae    = mean_absolute_error(y, y_pred)
    mse    = mean_squared_error(y, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y, y_pred)
    n, p   = len(y), X.shape[1] - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n{'=' * 45}")
    print(f"  {target_name}")
    print(f"{'=' * 45}")
    print(f"  Coefficients B : {np.round(B, 4)}")
    print(f"  Predicted      : {np.round(y_pred, 4)}")
    print(f"  MAE    : {mae:.4f}")
    print(f"  MSE    : {mse:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  R²     : {r2:.4f}")
    print(f"  Adj R² : {adj_r2:.4f}")
    return B, y_pred

# ── Run Regression ───────────────────────────────────────────
print("=" * 45)
print("    MULTIPLE LINEAR REGRESSION")
print("=" * 45)

B1, y1_pred = fit_regression(X_design, y1, "Target Y1")
B2, y2_pred = fit_regression(X_design, y2, "Target Y2")

# ── Plot ─────────────────────────────────────────────────────
plt.figure(figsize=(11, 4))
for i, (y, y_pred, B, label) in enumerate(
        [(y1, y1_pred, B1, 'Y1'), (y2, y2_pred, B2, 'Y2')], 1):
    plt.subplot(1, 2, i)
    plt.scatter(X_original, y, color='blue', s=100, zorder=5, label='Actual')
    plt.plot(X_original, y_pred, color='red', linewidth=2,
             label=f'y = {B[0]:.2f} + {B[1]:.2f}x')
    plt.title(f'Experiment 3 - Regression for {label}')
    plt.xlabel('X')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
