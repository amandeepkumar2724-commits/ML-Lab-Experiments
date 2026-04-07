#  Aim:
#  Write a code to generate a regression line on the following
#  datapoints and compute error metrics: MSE, MAE, RMSE, R²,
#  Adj R², SEE.
#
#  Dataset:
#  X | 1  2  3  4  5
#  Y | 2  4  5  4  5
#
#  Method: Least Square Method
#  Regression Line: y = b0 + b1*x
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Input Data ───────────────────────────────────────────────
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

X_reshaped = X.reshape(-1, 1)

# ── Model Fitting ────────────────────────────────────────────
model = LinearRegression()
model.fit(X_reshaped, y)

slope     = model.coef_[0]
intercept = model.intercept_

print("=" * 45)
print("       SIMPLE LINEAR REGRESSION")
print("=" * 45)
print(f"  Calculated Slope (b1)     : {slope:.4f}")
print(f"  Calculated Intercept (b0) : {intercept:.4f}")

# ── Predictions ──────────────────────────────────────────────
y_pred = model.predict(X_reshaped)
print(f"\n  Predicted y-values:")
print(f"  {y_pred}")

# ── Performance Metrics ──────────────────────────────────────
mae       = mean_absolute_error(y, y_pred)
mse       = mean_squared_error(y, y_pred)
rmse      = np.sqrt(mse)
r_squared = r2_score(y, y_pred)

n = len(y)
p = 1  # number of predictors

adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
see           = np.sqrt(np.sum((y - y_pred) ** 2) / (n - p - 1))

print("\n" + "=" * 45)
print("       PERFORMANCE METRICS")
print("=" * 45)
print(f"  MAE    : {mae:.4f}")
print(f"  MSE    : {mse:.4f}")
print(f"  RMSE   : {rmse:.4f}")
print(f"  R²     : {r_squared:.4f}")
print(f"  Adj R² : {adj_r_squared:.4f}")
print(f"  SEE    : {see:.4f}")
print("=" * 45)

# ── Plot ─────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color='blue', s=100, zorder=5, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Experiment 2 - Simple Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
