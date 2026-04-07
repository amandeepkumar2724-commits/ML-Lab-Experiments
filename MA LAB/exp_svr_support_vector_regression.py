#  Aim:
#  Apply SVR to determine a regression function for the
#  dataset such that data points lie within the epsilon-tube
#  (epsilon margin) while minimizing deviations outside.
#
#  Dataset:
#  Point | X | Y
#    A   | 1 | 2
#    B   | 2 | 3
#    C   | 3 | 2
#
#  X = [1, 2, 3],  Y = [2, 3, 2]
#
#  Objective:
#  - Construct a regression model using SVR
#  - Fit a function approximating the relationship between X and Y
#  - Observe how SVR places an epsilon-tube around regression line
#  - Identify support vectors influencing regression function
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Dataset ──────────────────────────────────────────────────
X = np.array([1, 2, 3]).reshape(-1, 1)
y = np.array([2, 3, 2])

# ── SVR Model ────────────────────────────────────────────────
epsilon = 0.1
svr = SVR(kernel='rbf', C=100, epsilon=epsilon)
svr.fit(X, y)

y_pred = svr.predict(X)

# ── Metrics ──────────────────────────────────────────────────
mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred)

print("=" * 45)
print("    SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 45)
print(f"  Kernel            : RBF")
print(f"  C (regularization): 100")
print(f"  Epsilon (ε-tube)  : {epsilon}")
print(f"\n  Support Vector Indices : {svr.support_}")
print(f"  Support Vectors        : {svr.support_vectors_.flatten()}")
print(f"\n  Predicted Values : {np.round(y_pred, 4)}")
print("\n" + "=" * 45)
print("       PERFORMANCE METRICS")
print("=" * 45)
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print("=" * 45)

# ── Plot ─────────────────────────────────────────────────────
X_plot = np.linspace(0.5, 3.5, 200).reshape(-1, 1)
y_plot = svr.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', s=150, zorder=5, label='Actual Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='SVR Prediction')
plt.fill_between(X_plot.flatten(),
                 y_plot - epsilon, y_plot + epsilon,
                 alpha=0.25, color='orange', label=f'ε-tube (±{epsilon})')
plt.scatter(svr.support_vectors_,
            svr.predict(svr.support_vectors_),
            s=250, facecolors='none', edgecolors='green',
            linewidths=2, zorder=6, label='Support Vectors')

point_labels = ['A', 'B', 'C']
for i, (xi, yi, name) in enumerate(zip(X.flatten(), y, point_labels)):
    plt.annotate(f' {name}', (xi, yi), fontsize=11, fontweight='bold')

plt.title('Experiment - Support Vector Regression (SVR)')
plt.xlabel('X'); plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
