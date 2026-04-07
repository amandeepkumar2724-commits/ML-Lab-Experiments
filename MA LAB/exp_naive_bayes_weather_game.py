#  Aim:
#  Implement Naive Bayes Classifier to predict whether a game
#  will be played based on weather conditions.
#
#  Dataset:
#  Instance | Weather | Play Game
#     1     |  Sunny  |   Yes
#     2     |  Sunny  |   Yes
#     3     |  Rainy  |   No
#     4     |  Rainy  |   No
#
#  Encoding: Sunny=1, Rainy=0 | Yes=1, No=0
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Dataset ──────────────────────────────────────────────────
# Encoding: Sunny=1, Rainy=0
weather   = np.array([[1], [1], [0], [0]])
play_game = np.array([1, 1, 0, 0])          # Yes=1, No=0

# ── Train Gaussian Naive Bayes ───────────────────────────────
model = GaussianNB()
model.fit(weather, play_game)

# ── Predictions on Training Data ─────────────────────────────
y_pred   = model.predict(weather)
accuracy = accuracy_score(play_game, y_pred)

print("=" * 45)
print("    NAIVE BAYES - WEATHER & GAME")
print("=" * 45)
print(f"  Training Accuracy : {accuracy:.4f}")
print(f"  Predictions       : {y_pred}")

# ── Predict New Weather Conditions ───────────────────────────
print("\n" + "=" * 45)
print("  PREDICTIONS FOR NEW INPUTS")
print("=" * 45)

test_cases = [('Sunny', [[1]]), ('Rainy', [[0]])]
for weather_name, val in test_cases:
    pred   = model.predict(val)
    prob   = model.predict_proba(val)
    result = "Yes ✓" if pred[0] == 1 else "No ✗"
    print(f"\n  Weather : {weather_name}")
    print(f"  Play Game?        : {result}")
    print(f"  P(No)  = {prob[0][0]:.4f}")
    print(f"  P(Yes) = {prob[0][1]:.4f}")

# ── Classification Report ────────────────────────────────────
print("\n" + "=" * 45)
print("  CLASSIFICATION REPORT")
print("=" * 45)
print(classification_report(play_game, y_pred, target_names=['No', 'Yes']))

# ── Plot ─────────────────────────────────────────────────────
weather_labels = ['Sunny', 'Sunny', 'Rainy', 'Rainy']
colors = ['green' if g == 1 else 'red' for g in play_game]

plt.figure(figsize=(6, 4))
plt.bar(range(len(play_game)),
        [model.predict_proba([[w]])[0][1] for w in weather.flatten()],
        color=colors, edgecolor='black', alpha=0.8)
plt.xticks(range(len(play_game)),
           [f'{w}\n(#{i+1})' for i, w in enumerate(weather_labels)])
plt.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold=0.5')
plt.ylabel('P(Play Game = Yes)')
plt.title('Experiment 6 - Naive Bayes: Game Prediction Probability')
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
