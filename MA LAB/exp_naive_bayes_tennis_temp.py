#  Aim:
#  Implement Naive Bayes classifier to predict whether tennis
#  will be played based on temperature & weather conditions.
#
#  Dataset:
#  Day | Temp | Weather | Play Tennis
#   1  |  30  |   Hot   |     No
#   2  |  28  |   Hot   |     No
#   3  |  15  |  Cool   |    Yes
#   4  |  16  |  Cool   |    Yes
#   5  |  18  |  Cool   |    Yes
#   6  |  35  |   Hot   |     No
#
#  Predict: Will tennis be played when Temperature = 20°C?
#  Encoding: Hot=1, Cool=0 | Yes=1, No=0
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# ── Dataset ──────────────────────────────────────────────────
temp    = np.array([30, 28, 15, 16, 18, 35]).reshape(-1, 1)
weather = np.array([1,  1,  0,  0,  0,  1]).reshape(-1, 1)  # Hot=1, Cool=0

X = np.hstack([temp, weather])
y = np.array([0, 0, 1, 1, 1, 0])                            # No=0, Yes=1

# ── Train Model ──────────────────────────────────────────────
model = GaussianNB()
model.fit(X, y)

y_pred   = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print("=" * 50)
print("    NAIVE BAYES - TENNIS & TEMPERATURE")
print("=" * 50)
print(f"  Training Accuracy : {accuracy:.4f}")

# ── Predict at Temperature = 20°C ────────────────────────────
x_test     = np.array([[20, 0]])   # Temp=20°C, Cool weather
prediction = model.predict(x_test)
proba      = model.predict_proba(x_test)
result     = "Yes - Tennis WILL be played!" if prediction[0] == 1 \
             else "No  - Tennis WON'T be played."

print("\n" + "=" * 50)
print("  PREDICTION FOR TEMPERATURE = 20°C")
print("=" * 50)
print(f"  Result            : {result}")
print(f"  P(No)  = {proba[0][0]:.4f}")
print(f"  P(Yes) = {proba[0][1]:.4f}")

# ── Classification Report ────────────────────────────────────
print("\n" + "=" * 50)
print("  CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y, y_pred, target_names=['No', 'Yes']))

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1 – Temperature vs Play Tennis
colors = ['red' if label == 0 else 'green' for label in y]
axes[0].scatter(temp, y, c=colors, s=150, zorder=5, edgecolors='black')
axes[0].scatter(20, prediction[0], marker='*', color='blue',
                s=300, zorder=6, label='Predict (Temp=20°C)')
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Play Tennis (0=No, 1=Yes)')
axes[0].set_title('Exp 6B - Temperature vs Play Tennis')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2 – Probability bar chart
temps_range = np.arange(10, 40, 1).reshape(-1, 1)
probs_yes   = [model.predict_proba([[t, 0]])[0][1] for t in temps_range]
axes[1].plot(temps_range, probs_yes, color='blue', linewidth=2)
axes[1].axvline(20, color='orange', linestyle='--', label='Temp = 20°C')
axes[1].axhline(0.5, color='gray', linestyle=':', label='Threshold = 0.5')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('P(Play Tennis = Yes)')
axes[1].set_title('Probability Curve (Cool Weather)')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Experiment 6B - Naive Bayes Tennis Prediction', fontsize=13)
plt.tight_layout()
plt.show()
