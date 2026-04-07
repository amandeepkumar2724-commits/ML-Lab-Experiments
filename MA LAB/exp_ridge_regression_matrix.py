#  Aim:
#  Consider matrix B0 = [[1,2,1],[2,3,2],[3,4,3]].
#  Find B0, B1, B2 representing the regression model:
#    y = __ + x1*__ + x2*__
#  using Ridge Regression with lambda = 1.
#
#  Method : Ridge Regression
#  Line   : y = b0 + b1*x1 + b2*x2
#  Formula: B = (X'X + λI)^-1 * X'y
# ============================================================

import numpy as np

# ── Input Data ───────────────────────────────────────────────
# Design matrix with bias column (first column = 1)
X = np.array([[1, 1, 2],
              [1, 2, 3],
              [1, 3, 4]])

y = np.array([[1, 2, 3],
              [1, 2, 3],
              [1, 3, 4]])

alpha = 1   # λ

# ── Ridge Formula ────────────────────────────────────────────
X_t_X         = np.dot(X.T, X)
X_t_y         = np.dot(X.T, y)
lambda_I       = alpha * np.eye(X.shape[1])
inv_matrix     = np.linalg.inv(X_t_X + lambda_I)
B              = np.dot(inv_matrix, X_t_y)

print("=" * 50)
print("   RIDGE REGRESSION - MULTIPLE FEATURES (λ=1)")
print("=" * 50)
print(f"\n  X'X =\n{X_t_X}")
print(f"\n  (X'X + λI)⁻¹ =\n{np.round(inv_matrix, 4)}")
print(f"\n  Coefficient Matrix B =\n{np.round(B, 4)}")

# ── Predictions ──────────────────────────────────────────────
y_pred = np.dot(X, B)
print(f"\n  Predicted y =\n{np.round(y_pred, 4)}")

print("\n" + "=" * 50)
print("  RESULT")
print("=" * 50)
print(f"  b0 (intercept) : {B[0][0]:.4f}")
print(f"  b1             : {B[1][0]:.4f}")
print(f"  b2             : {B[2][0]:.4f}")
print(f"\n  Regression Equation:")
print(f"  y = {B[0][0]:.2f} + {B[1][0]:.2f}*x1 + {B[2][0]:.2f}*x2")
print("=" * 50)
