# ============================================================
#  Experiment 4 - Ridge Regression (L2 Regularization)
#  Name   : Amandeep Kumar
#  Roll   : 24UG010675
#  Date   : 09/02/26
# ============================================================
#
#  Aim:
#  Generate the regression coefficient using Ridge Regression
#  with regularization parameter lambda = 1.
#
#  Dataset:
#  Obs | x | y
#   1  | 2 | 1
#   2  | 3 | 2
#   3  | 4 | 3
#
#  Method : Ridge Regression
#  Line   : y = b0 + b1*x
#  Formula: B = (X'X + λI)^-1 * X'y
# ============================================================

import numpy as np

# ── Input Data ───────────────────────────────────────────────
# Design matrix: each row is [1 (bias), x]
X     = np.array([[1, 2],
                  [1, 3],
                  [1, 4]])
y     = np.array([[1], [2], [3]])
alpha = 1                           # Regularization parameter λ

# ── Ridge Formula ────────────────────────────────────────────
X_t_X          = np.dot(X.T, X)
lambda_I        = alpha * np.eye(X.shape[1])          # λI
X_t_X_plus_lI  = X_t_X + lambda_I
inv_matrix      = np.linalg.inv(X_t_X_plus_lI)
B               = np.dot(inv_matrix, np.dot(X.T, y))  # Ridge coefficients

# ── Predictions ──────────────────────────────────────────────
y_pred = np.dot(X, B)

print("=" * 45)
print("     RIDGE REGRESSION  (λ = 1)")
print("=" * 45)
print(f"\n  (X'X + λI)⁻¹ =\n{np.round(inv_matrix, 4)}")
print(f"\n  Coefficients:")
print(f"    b0 (intercept) : {B[0][0]:.4f}")
print(f"    b1 (slope)     : {B[1][0]:.4f}")
print(f"\n  Predicted y values : {y_pred.flatten().round(4)}")

# ── Performance Metrics ──────────────────────────────────────
y_flat    = y.flatten()
y_p_flat  = y_pred.flatten()
mse       = np.mean((y_flat - y_p_flat) ** 2)
mae       = np.mean(np.abs(y_flat - y_p_flat))
ss_res    = np.sum((y_flat - y_p_flat) ** 2)
ss_tot    = np.sum((y_flat - np.mean(y_flat)) ** 2)
r2        = 1 - ss_res / ss_tot

print("\n" + "=" * 45)
print("       PERFORMANCE METRICS")
print("=" * 45)
print(f"  MAE : {mae:.4f}")
print(f"  MSE : {mse:.4f}")
print(f"  R²  : {r2:.4f}")

print("\n" + "=" * 45)
print("  RESULT")
print("=" * 45)
print(f"  Regression Equation:")
print(f"  y = {B[0][0]:.2f} + {B[1][0]:.2f} * x")
print("=" * 45)
