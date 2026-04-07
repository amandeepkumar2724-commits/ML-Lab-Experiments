#  Aim:
#  Determine the optimal separating hyperplane using SVM
#  for the given dataset and classify a new data point.
#
#  Dataset:
#  Point | x1 | x2 | y
#    A   |  1 |  1 | -1
#    B   |  2 |  1 | +1
#    C   |  2 |  3 | -1
#    D   |  3 |  3 | -1
#  New point to classify: (2, 2)
#
#  Math:
#  Optimization : min 1/2 ||w||²
#  Constraint   : yi(w·xi + b) >= 1
#  Decision fn  : f(x) = w·x + b
#  Hyperplane   : w·x + b = 0
#  Margin       : 2 / ||w||
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ── Dataset ──────────────────────────────────────────────────
X = np.array([[1, 1],
              [2, 1],
              [2, 3],
              [3, 3]])
y = np.array([-1, 1, -1, -1])

# ── Train Hard-Margin SVM ────────────────────────────────────
model = SVC(kernel='linear', C=1e10)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]
margin = 2 / np.linalg.norm(w)

print("=" * 45)
print("    SUPPORT VECTOR MACHINE (SVM)")
print("=" * 45)
print(f"  Weight vector w    : {w}")
print(f"  Bias b             : {b:.4f}")
print(f"  Margin (2/||w||)   : {margin:.4f}")
print(f"\n  Support Vectors    :")
for sv in model.support_vectors_:
    print(f"    {sv}")

# ── Classify New Point ───────────────────────────────────────
new_point  = np.array([[2, 2]])
prediction = model.predict(new_point)
dec_val    = model.decision_function(new_point)

print("\n" + "=" * 45)
print("  RESULT")
print("=" * 45)
print(f"  New Point (2, 2)")
print(f"  Decision Value     : {dec_val[0]:.4f}")
print(f"  Predicted Class    : {prediction[0]}")
label = "Positive (+1)" if prediction[0] == 1 else "Negative (-1)"
print(f"  Class Label        : {label}")
print("=" * 45)

# ── Decision Boundary Visualization ─────────────────────────
xx, yy = np.meshgrid(np.linspace(0, 4, 300), np.linspace(0, 4, 300))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, levels=[-2, -1, 0, 1, 2],
             alpha=0.2, colors=['#ff9999', '#ffcccc', '#ffffff', '#ccccff', '#9999ff'])
plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            colors=['red', 'black', 'blue'], linewidths=[1.5, 2.5, 1.5],
            linestyles=['--', '-', '--'])

colors = ['red' if label == -1 else 'blue' for label in y]
point_labels = ['A', 'B', 'C', 'D']
for i, (pt, c, name) in enumerate(zip(X, colors, point_labels)):
    plt.scatter(*pt, color=c, s=150, zorder=5)
    plt.annotate(f' {name}', pt, fontsize=11, fontweight='bold')

plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=250, facecolors='none', edgecolors='black',
            linewidths=2, zorder=6, label='Support Vectors')
plt.scatter(*new_point[0], marker='*', color='green', s=300,
            zorder=7, label=f'New Point (2,2) → {prediction[0]}')

plt.title('Experiment 5B - SVM Decision Boundary')
plt.xlabel('x₁'); plt.ylabel('x₂')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
