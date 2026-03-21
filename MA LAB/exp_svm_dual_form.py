# ============================================================
#  Experiment - Linear SVM in Dual Form
#  Name   : Amandeep Kumar
#  Roll   : 24UG010675
#  Date   : Sheet No. 26
# ============================================================
#
#  Aim:
#  A linear SVM has the following alpha, support vectors and
#  output class y. Compute the predicted class label when
#  input feature vector is (0.2, 0.8, 0.7).
#
#  Dataset:
#  alpha | Support Vector | y
#    1   | (0, -1,  1)    | +1
#    1   | (0,  2, -1)    | -1
#    1   | (-1, 0,  2)    | -1
#
#  Decision Function (Dual Form):
#  f(x) = sum_i [ alpha_i * y_i * <v_i, x> ] + b
# ============================================================

import numpy as np

# ── Support Vectors & Labels ─────────────────────────────────
n_sv  = np.array([[ 0, -1,  1],
                  [ 0,  2, -1],
                  [-1,  0,  2]])

y_sv  = np.array([1, -1, -1])
alpha = np.array([1,  1,  1])

# ── Compute Weight Vector w = Σ αᵢ yᵢ xᵢ ───────────────────
w = np.sum(alpha[:, None] * y_sv[:, None] * n_sv, axis=0)

# ── Compute Bias b = y[0] - w · sv[0] ───────────────────────
b = y_sv[0] - np.dot(w, n_sv[0])

print("=" * 45)
print("    SVM - DUAL FORM")
print("=" * 45)
print(f"  Weight vector w : {w}")
print(f"  Bias b          : {b}")

# ── Decision Function for Test Point ─────────────────────────
x_test = np.array([0.2, 0.8, 0.7])
f_n    = np.dot(w, x_test) + b
y_pred = np.sign(f_n)

print(f"\n  Test Point x    : {x_test}")
print(f"  f(x_test)       : {f_n:.4f}")
print(f"  Predicted label : {int(y_pred)}")

print("\n" + "=" * 45)
print("  RESULT")
print("=" * 45)
print(f"  Weight vector w      : {w}")
print(f"  Bias b               : {b}")
print(f"  Decision f(x_test)   : {f_n:.2f}")
label = "Positive (+1)" if y_pred == 1 else "Negative (-1)"
print(f"  Predicted Class      : {label}")
print("=" * 45)
