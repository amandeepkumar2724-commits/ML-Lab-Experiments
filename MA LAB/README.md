# 🤖 Machine Learning Lab Experiments
### Amandeep Kumar | Roll: 24UG010675 | 4th Semester | JNTU Gunupur

---

## 📋 Table of Contents

| # | Experiment | Topic |
|---|-----------|-------|
| 1 | [Experiment 2](#experiment-2) | Simple Linear Regression |
| 2 | [Experiment 3](#experiment-3) | Multiple Linear Regression |
| 3 | [Experiment 4](#experiment-4) | Ridge Regression |
| 4 | [Experiment 5](#experiment-5) | Ridge Regression (Matrix Input) |
| 5 | [Experiment 5B](#experiment-5b) | Support Vector Machine (SVM) |
| 6 | [Experiment 5 (SVR)](#experiment-svr) | Support Vector Regression (SVR) |
| 7 | [Experiment 5 (SVM Dual)](#experiment-svm-dual) | SVM in Dual Form |
| 8 | [Experiment 6](#experiment-6) | Naive Bayes – Weather (Game) |
| 9 | [Experiment 6B](#experiment-6b) | Naive Bayes – Tennis (Temperature) |
| 10 | [Experiment 7](#experiment-7) | K-Means Clustering |
| 11 | [Experiment 7B](#experiment-7b) | K-Medoid (PAM) Clustering |
| 12 | [Experiment 8](#experiment-8) | Agglomerative (Hierarchical) Clustering |

---

## Experiment 2
### Simple Linear Regression with Performance Metrics

**Date:** 12/01/26

#### 🎯 Aim
Write a code to generate a regression line on the following datapoints and compute error metrics: **MSE, MAE, RMSE, R², Adj R², SEE**.

| X | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Y | 2 | 4 | 5 | 4 | 5 |

#### 📌 Objective
To develop a **supervised learning model** using linear regression to find the best fit line for predicting outcomes from labelled data.

#### 🔧 Materials & Methods

- **Method:** Least Square Method
- **Regression Line:** `ŷ = b₀ + b₁x`

**Formulas:**

$$b_0 = \frac{n\sum x_i y_i - \sum x_i \cdot \sum y_i}{n\sum x_i^2 - \left(\sum x_i\right)^2}$$

$$b_1 = \frac{1}{n}\left[\sum y_i - b_0 \sum x_i\right]$$

- **Tools:** Google Colab, Python (NumPy, Matplotlib, Scikit-learn)

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

X_reshaped = X.reshape(-1, 1)

# Model fitting
model = LinearRegression()
model.fit(X_reshaped, y)

slope     = model.coef_[0]
intercept = model.intercept_

print(f"\n Calculated Slope (b1)     : {slope:.4f}")
print(f" Calculated Intercept (b0) : {intercept:.4f}")

# Predictions
y_pred = model.predict(X_reshaped)
print(f"\n Predicted y-values (y_pred) using model.predict:")
print(y_pred)

# Performance Metrics
mae       = mean_absolute_error(y, y_pred)
mse       = mean_squared_error(y, y_pred)
rmse      = np.sqrt(mse)
r_squared = r2_score(y, y_pred)

n = len(y)
p = 1  # number of predictors
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# SEE (Standard Error of Estimate)
see = np.sqrt(np.sum((y - y_pred) ** 2) / (n - p - 1))

print(f"\n MAE       : {mae:.4f}")
print(f" MSE       : {mse:.4f}")
print(f" RMSE      : {rmse:.4f}")
print(f" R²        : {r_squared:.4f}")
print(f" Adj R²    : {adj_r_squared:.4f}")
print(f" SEE       : {see:.4f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

#### 📊 Expected Output

```
 Calculated Slope (b1)     : 0.6000
 Calculated Intercept (b0) : 2.2000

 Predicted y-values (y_pred) using model.predict:
 [2.8  3.4  4.   4.6  5.2]

 MAE       : 0.4800
 MSE       : 0.3600
 RMSE      : 0.6000
 R²        : 0.6000
 Adj R²    : 0.4667
 SEE       : 0.7746
```

---

## Experiment 3
### Multiple Linear Regression (Matrix Factorization)

**Date:** 12/3/26

#### 🎯 Aim
Write a code to generate a regression line for multiple input features and compute performance metrics: **MAE, MSE, RMSE, R², Adj R²**.

| X | 1 | 2 | 3 |
|---|---|---|---|
| Y₁ | 2 | 4 | 6 |
| Y₂ | 3 | 6 | 7 |

#### 📌 Objective
To implement a **multiple regression model** using matrix factorization to generate the best regression equation for a dataset with multiple input features.

#### 🔧 Materials & Methods

- **Method:** Matrix Factorization Method
- **Regression Line:** `ŷ = b₀ + b₁x₁ + b₂x₂`
- **Formula:** `B = (XᵀX)⁻¹ Xᵀy`

Where:
- `B = [b₀, b₁, b₂]ᵀ` (coefficient vector)
- `X` = Design matrix
- `y` = Column vector of output values

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input data
X_original = np.array([1, 2, 3])
y1 = np.array([2, 4, 6])
y2 = np.array([3, 6, 7])

# Design matrix (add bias column of ones)
X_design = np.column_stack([np.ones(len(X_original)), X_original])

def fit_regression(X, y, target_name):
    # B = (XᵀX)⁻¹ Xᵀy
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    y_pred = X @ B

    mae  = mean_absolute_error(y, y_pred)
    mse  = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y, y_pred)
    n, p = len(y), X.shape[1] - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n=== {target_name} ===")
    print(f" Coefficients (B): {B}")
    print(f" Predicted Values : {y_pred}")
    print(f" MAE    : {mae:.4f}")
    print(f" MSE    : {mse:.4f}")
    print(f" RMSE   : {rmse:.4f}")
    print(f" R²     : {r2:.4f}")
    print(f" Adj R² : {adj_r2:.4f}")
    return B, y_pred

B1, y1_pred = fit_regression(X_design, y1, "Target Y1")
B2, y2_pred = fit_regression(X_design, y2, "Target Y2")

# Plot
plt.figure(figsize=(10, 4))
for i, (y, y_pred, label) in enumerate([(y1, y1_pred, 'Y1'), (y2, y2_pred, 'Y2')], 1):
    plt.subplot(1, 2, i)
    plt.scatter(X_original, y, color='blue', label='Actual')
    plt.plot(X_original, y_pred, color='red', label='Predicted')
    plt.title(f'Regression for {label}')
    plt.xlabel('X'); plt.ylabel(label)
    plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Experiment 4
### Ridge Regression (L2 Regularization)

**Date:** 9/02/26

#### 🎯 Aim
Generate the regression coefficient using **Ridge Regression** with regularization parameter λ = 1.

| Obs | x | y |
|-----|---|---|
| 1   | 2 | 1 |
| 2   | 3 | 2 |
| 3   | 4 | 3 |

#### 🔧 Materials & Methods

- **Regression Line:** `y = b₀ + b₁x`
- **Formula:** `B = (XᵀX + λI)⁻¹ Xᵀy`

#### 💻 Code

```python
import numpy as np

# Input data
X = np.array([[1, 2], [1, 3]])   # Design matrix [bias, x]
y = np.array([[1], [2], [3]])
alpha = 1                         # Regularization parameter λ

X_t_mul_X = np.dot(X.T, X)
lambda_mul_i = alpha * np.array([[1, 0], [0, 1]])   # λI

# B = (XᵀX + λI)⁻¹ Xᵀy
X_t_mul_X_plus_lambda_mul_i_inv = np.linalg.inv(X_t_mul_X + lambda_mul_i)
print("(XᵀX + λI)⁻¹ =\n", X_t_mul_X_plus_lambda_mul_i_inv)

y_pred = np.dot(X_t_mul_X_plus_lambda_mul_i_inv, np.dot(X.T, y))
print(f"\n The regression equation: y_pred = {y_pred[0][0]:.2f} + {y_pred[1][0]:.2f}*x")

# Result
print(f"\n Result: The regression equation is")
print(f" y_pred[{y_pred[0][0]:.2f}] + {y_pred[1][0]:.2f}*x")
```

#### 📊 Result
```
 The regression equation is:
 y = 2.44 + 0.44x + 0.42
```

---

## Experiment 5
### Ridge Regression (Multiple Features / Matrix Input)

**Date:** 13/3/26

#### 🎯 Aim
Consider a matrix where B₀ = `[[1,2,1],[2,3,2],[3,4,3]]`. Find B₀, B₁, B₂ which represent the regression model `y = __ + x₁__ + x₂__` using **Ridge Regression** with λ = 1.

#### 🔧 Method
- **Regression Line:** `ŷ = b₀ + b₁x₁ + b₂x₂`
- **Formula:** `B = (XᵀX + λI)⁻¹ Xᵀy`

#### 💻 Code

```python
import numpy as np

# Input data (Design matrix with bias column)
X = np.array([[1, 1, 2],
              [1, 2, 3],
              [1, 3, 4]])

y = np.array([[1, 2, 3],
              [1, 2, 3],
              [1, 3, 4]])

alpha = 1  # λ

X_t_mul_X = np.dot(X.T, X)
X_t_mul_y = np.dot(X.T, y)

lambda_mul_i = alpha * np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

# Ridge formula: B = (XᵀX + λI)⁻¹ Xᵀy
X_t_mul_X_plus_lambda_mul_i_inv = np.linalg.inv(X_t_mul_X + lambda_mul_i)

y_pred = np.dot(X_t_mul_X_plus_lambda_mul_i_inv, X_t_mul_y)
print(y_pred)

print(f"\n The regression equation:")
print(f" y_pred = {y_pred[0][0]:.2f} + {y_pred[1][0]:.2f}*x1 + {y_pred[2][0]:.2f}*x2")

# Result
print(f"\n Result: The regression equation is")
print(f" y = {y_pred[0][0]:.2f} + {y_pred[1][0]:.2f}*x₁ + {y_pred[2][0]:.2f}*x₂")
```

#### 📊 Result
```
 The regression equation is:
 y = 2.49 + 0.4*x₁ + 0.42*x₂
```

---

## Experiment 5B
### Support Vector Machine (SVM) Classification

**Date:** Sheet No. 22

#### 🎯 Aim
Determine the optimal separating hyperplane using **Support Vector Machine (SVM)** for the given dataset and classify a new data point.

#### 📊 Dataset

| Point | x₁ | x₂ | Target (y) |
|-------|----|----|-----------|
| A     | 1  | 1  | -1        |
| B     | 2  | 1  | +1        |
| C     | 2  | 3  | -1        |
| D     | 3  | 3  | -1        |

**New Point to classify:** (2, 2)

#### 📌 Objective
- Understand the concept of maximum-margin hyperplane
- Compute optimal values of **w** and **b**
- Classify new data points
- Visualize the decision boundary

#### 🔧 Mathematical Formulation

| Formula | Description |
|---------|-------------|
| `min ½‖w‖²` | Optimization function |
| `yᵢ(w·xᵢ + b) ≥ 1` | Constraint condition |
| `f(x) = w·x + b` | Decision function |
| `w·x + b = 0` | Hyperplane equation |
| `2/‖w‖` | Margin |

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Dataset
X = np.array([[1, 1],
              [2, 1],
              [2, 3],
              [3, 3]])
y = np.array([-1, 1, -1, -1])

# Train SVM with linear kernel
model = SVC(kernel='linear', C=1e10)  # Hard margin SVM
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

print(f" Weight vector w : {w}")
print(f" Bias b          : {b:.4f}")
print(f" Margin          : {2 / np.linalg.norm(w):.4f}")
print(f" Support Vectors : {model.support_vectors_}")

# Classify new point
new_point = np.array([[2, 2]])
prediction = model.predict(new_point)
print(f"\n New Point (2,2) classified as: {prediction[0]}")

# Decision boundary visualization
xx, yy = np.meshgrid(np.linspace(0, 4, 200), np.linspace(0, 4, 200))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'white', 'blue'])
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linewidths=[1, 2, 1])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=100, zorder=5)
plt.scatter(*new_point[0], marker='*', color='green', s=200, label='New Point (2,2)', zorder=6)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')
plt.title('SVM Decision Boundary')
plt.xlabel('x₁'); plt.ylabel('x₂')
plt.legend(); plt.grid(True)
plt.show()
```

---

## Experiment SVR
### Support Vector Regression (SVR)

**Date:** Sheet No. 24

#### 🎯 Aim
Apply **Support Vector Regression (SVR)** to determine a regression function such that data points lie within the ε-tube (epsilon margin) while minimizing deviations outside the margin.

#### 📊 Dataset

| Point | X | Y |
|-------|---|---|
| A     | 1 | 2 |
| B     | 2 | 3 |
| C     | 3 | 2 |

`X = [1, 2, 3]`, `Y = [2, 3, 2]`

#### 📌 Objective
- Construct a regression model using SVR
- Fit a function approximating the relationship between X and Y
- Observe how SVR places an ε-tube around the regression line
- Identify support vectors that influence the regression function

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Dataset
X = np.array([1, 2, 3]).reshape(-1, 1)
y = np.array([2, 3, 2])

# SVR with RBF kernel
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X, y)

y_pred = svr.predict(X)

print(f" Support Vectors    : {svr.support_vectors_.flatten()}")
print(f" Predicted Values   : {y_pred}")
print(f" Support Vector Idx : {svr.support_}")

# Smooth prediction curve
X_plot = np.linspace(0.5, 3.5, 100).reshape(-1, 1)
y_plot = svr.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', zorder=5, label='Actual Data', s=100)
plt.plot(X_plot, y_plot, color='red', label='SVR Prediction')
plt.fill_between(X_plot.flatten(),
                 y_plot - 0.1, y_plot + 0.1,
                 alpha=0.2, color='orange', label='ε-tube')
plt.scatter(svr.support_vectors_, svr.predict(svr.support_vectors_),
            s=200, facecolors='none', edgecolors='green', linewidths=2, label='Support Vectors')
plt.title('Support Vector Regression (SVR)')
plt.xlabel('X'); plt.ylabel('Y')
plt.legend(); plt.grid(True)
plt.show()
```

---

## Experiment SVM Dual
### Linear SVM in Dual Form

**Date:** Sheet No. 26

#### 🎯 Aim
A linear SVM has the following values of **α (alpha), support vectors, and output class y**. Compute the predicted class label of this SVM when input feature vector is **(0.2, 0.8, 0.7)**.

#### 📊 Dataset

| α | Support Vector | y  |
|---|---------------|----|
| 1 | (0, -1, 1)    | +1 |
| 1 | (0, 2, -1)    | -1 |
| 1 | (-1, 0, 2)    | -1 |

#### 🔧 Decision Function (Dual Form)

$$f(x) = \sum_{i=1}^{n} \alpha_i y_i \langle v_i, x \rangle + b$$

#### 💻 Code

```python
import numpy as np

# Support vectors
n_sv = np.array([[0, -1, 1],
                 [0,  2,-1],
                 [-1, 0, 2]])

y_sv   = np.array([1, -1, -1])
alpha  = np.array([1,  1,  1])
n_test = np.array([1,  1,  1])  # Test point multiplier

# Compute w = Σ αᵢ yᵢ xᵢ
w = np.sum(alpha[:, None] * y_sv[:, None] * n_sv, axis=0)
print(f" Weight vector w : {w}")

# Compute bias b = y[0] - w · sv[0]
b = y_sv[0] - np.dot(w, n_sv[0])
print(f" Bias b          : {b}")

# Decision function f(x) = w · x_test + b
x_test = np.array([0.2, 0.8, 0.7])
f_n    = np.dot(w, x_test) + b
print(f"\n Decision Function f(x_test) : {f_n}")

# Predicted class
y_pred = np.sign(f_n)
print(f" Predicted class label for x_test : {int(y_pred)}")

# Results
print("\n ===== RESULT =====")
print(f" Weight vector w  : {w}")
print(f" Bias b           : {b}")
print(f" f(x_test)        : {f_n:.2f}")
print(f" Predicted label  : {int(y_pred)}")
```

#### 📊 Result
```
 Weight vector w  : [1, -3, 0]
 Bias b           : -2
 f(x_test)        : 4.2
 Predicted label  : -1
```

---

## Experiment 6
### Naive Bayes Classifier – Weather & Game Prediction

**Date:** Sheet No. 27

#### 🎯 Aim
Implement **Naive Bayes Classifier** using Python to predict whether a game will be played based on weather conditions.

#### 📊 Dataset

| Instance | Weather | Play Game |
|----------|---------|-----------|
| 1        | Sunny   | Yes       |
| 2        | Sunny   | Yes       |
| 3        | Rainy   | No        |
| 4        | Rainy   | No        |

#### 📌 Objective
- Understand the working of the Naive Bayes algorithm
- Encode categorical data numerically
- Predict the outcome for given weather conditions

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Encode: Sunny=1, Rainy=0 | Yes=1, No=0
weather   = np.array([[1], [1], [0], [0]])
play_game = np.array([1, 1, 0, 0])

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(weather, play_game)

# Predictions
y_pred = model.predict(weather)
print(" Predictions:", y_pred)
print(" Accuracy   :", accuracy_score(play_game, y_pred))

# Predict for new weather
test_cases = {'Sunny (1)': [[1]], 'Rainy (0)': [[0]]}
for label, val in test_cases.items():
    pred = model.predict(val)
    prob = model.predict_proba(val)
    result = "Yes" if pred[0] == 1 else "No"
    print(f"\n Weather: {label} → Play Game: {result}")
    print(f" Probabilities [No, Yes]: {prob[0]}")

# Classification report
print("\n", classification_report(play_game, y_pred, target_names=['No', 'Yes']))
```

---

## Experiment 6B
### Naive Bayes Classifier – Tennis & Temperature Prediction

**Date:** Sheet No. 29

#### 🎯 Aim
Implement a **Naive Bayes classifier** using Scikit-learn to predict whether tennis will be played based on temperature and weather conditions.

#### 📊 Dataset

| Day | Temperature | Weather | Play Tennis |
|-----|-------------|---------|-------------|
| 1   | 30          | Hot     | No          |
| 2   | 28          | Hot     | No          |
| 3   | 15          | Cool    | Yes         |
| 4   | 16          | Cool    | Yes         |
| 5   | 18          | Cool    | Yes         |
| 6   | 35          | Hot     | No          |

**Predict:** Will tennis be played when **Temperature = 20°C**?

#### 📌 Objective
- Understand Naive Bayes with 2 features
- Train the model on a 6-day dataset
- Predict outcome at Temperature = 20°C

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Temperature and Weather (Hot=1, Cool=0)
temp    = np.array([30, 28, 15, 16, 18, 35]).reshape(-1, 1)
weather = np.array([1, 1, 0, 0, 0, 1]).reshape(-1, 1)

# Combine features
X = np.hstack([temp, weather])
y = np.array([0, 0, 1, 1, 1, 0])  # 0=No, 1=Yes

# Train model
model = GaussianNB()
model.fit(X, y)

# Predictions on training data
y_pred = model.predict(X)
print(" Training Accuracy:", accuracy_score(y, y_pred))

# Predict for Temperature = 20°C (Cool weather assumed)
x_test = np.array([[20, 0]])  # Temp=20, Cool weather
prediction = model.predict(x_test)
proba      = model.predict_proba(x_test)

result = "Yes, Tennis will be played!" if prediction[0] == 1 else "No, Tennis won't be played."
print(f"\n At Temperature=20°C : {result}")
print(f" Probabilities [No, Yes]: {proba[0]}")

print("\n", classification_report(y, y_pred, target_names=['No', 'Yes']))

# Visualization
plt.figure(figsize=(8, 5))
colors = ['red' if label == 0 else 'green' for label in y]
plt.scatter(temp, weather, c=colors, s=200, zorder=5)
plt.scatter(20, 0, marker='*', color='blue', s=300, label='Predict (Temp=20°C)')
plt.xlabel('Temperature'); plt.ylabel('Weather (0=Cool, 1=Hot)')
plt.title('Naive Bayes - Tennis Prediction')
plt.legend(); plt.grid(True)
plt.show()
```

---

## Experiment 7
### K-Means Clustering

**Date:** Sheet No. 31

#### 🎯 Aim
Implement **K-Means Clustering** algorithm in Python using Scikit-learn on the given dataset with **K=2 clusters** and initial centroids **A(2,3)** and **C(6,6)**.

#### 📊 Dataset

| Datapoint | X | Y |
|-----------|---|---|
| A         | 2 | 3 |
| B         | 3 | 4 |
| C         | 6 | 6 |
| D         | 7 | 7 |

- **K = 2**
- **Initial Centroids:** A(2,3) and C(6,6)

#### 📌 Objective
- Understand the working of K-Means Clustering algorithm
- Divide the dataset into two clusters
- Visualize the clustered datapoints and final centroids

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
X = np.array([[2, 3], [3, 4], [6, 6], [7, 7]])
labels = ['A', 'B', 'C', 'D']

# Initial centroids
initial_centroids = np.array([[2, 3], [6, 6]])

# K-Means model
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, max_iter=300, random_state=42)
kmeans.fit(X)

cluster_labels  = kmeans.labels_
final_centroids = kmeans.cluster_centers_

print(" Cluster Assignments:")
for i, (point, label) in enumerate(zip(labels, cluster_labels)):
    print(f"  Point {point} {X[i]} → Cluster {label + 1}")

print(f"\n Final Centroids:")
for i, c in enumerate(final_centroids):
    print(f"  Centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})")

print(f"\n Inertia (Within-cluster sum of squares): {kmeans.inertia_:.4f}")

# Visualization
colors = ['red' if l == 0 else 'blue' for l in cluster_labels]
plt.figure(figsize=(7, 6))
for i, (point, color, name) in enumerate(zip(X, colors, labels)):
    plt.scatter(*point, color=color, s=200, zorder=5)
    plt.annotate(name, (point[0] + 0.1, point[1] + 0.1), fontsize=12, fontweight='bold')

plt.scatter(final_centroids[:, 0], final_centroids[:, 1],
            marker='X', s=300, c='black', zorder=6, label='Final Centroids')
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1],
            marker='o', s=300, facecolors='none', edgecolors='green',
            linewidths=2, zorder=6, label='Initial Centroids')

plt.title('K-Means Clustering (K=2)')
plt.xlabel('X'); plt.ylabel('Y')
plt.legend(); plt.grid(True)
plt.show()
```

#### 📊 Result
```
 Cluster Assignments:
  Point A [2, 3] → Cluster 1
  Point B [3, 4] → Cluster 1
  Point C [6, 6] → Cluster 2
  Point D [7, 7] → Cluster 2

 Final Centroids:
  Centroid 1: (2.50, 3.50)
  Centroid 2: (6.50, 6.50)
```

---

## Experiment 7B
### K-Medoid (PAM) Clustering

**Date:** 30/03/26 | Sheet No. 35–36

#### 🎯 Aim
Implement the **K-Medoid (PAM)** clustering on a dataset of 5 points with **K=2** and initial medoids **m1 = Point A (2)** and **m2 = Point D (10)**.

#### 📊 Dataset

| Point | Value |
|-------|-------|
| A     | 2     |
| B     | 4     |
| C     | 5     |
| D     | 10    |
| E     | 12    |

- **K = 2**
- **Initial Medoids:** m1 = Point A (2), m2 = Point D (10)

#### 📌 Objective
- Apply the PAM algorithm with given initial medoids
- Divide the dataset into 2 final clusters

#### 🔧 Materials & Methods

- **Tools:** Python 3.x, NumPy, Matplotlib, k-medoid, Jupyter Notebook
- **Algorithm:** PAM (Partitioning Around Medoids) — assigns each point to the nearest medoid, then updates medoids by minimizing total absolute distances within clusters

#### 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt

data   = np.array([2, 4, 5, 10, 12])
names  = ['A', 'B', 'C', 'D', 'E']
k      = 2
medoid = [0, 3]   # Initial medoid indices: A=0, D=3

for _ in range(10):
    clusters = {i: [] for i in range(k)}
    for i, p in enumerate(data):
        clusters[np.argmin([abs(p - data[m]) for m in medoid])].append(i)

    new_medoids = [
        min(V, key=lambda m: sum(abs(data[m] - data[j]) for j in V))
        for V in clusters.values()
    ]
    if new_medoids == medoid:
        break
    medoid = new_medoids

colors = ['steelblue', 'tomato']
fig, ax = plt.subplots(figsize=(9, 3))
for i, (v, name) in enumerate(zip(data, names)):
    c = colors[0] if i in clusters[0] else colors[1]
    ax.scatter(v, 0, color=c, s=200, zorder=3)
for k, m in enumerate(medoid):
    ax.scatter(data[m], 0, color=colors[k], marker='*',
               s=500, edgecolors='black', zorder=5)
ax.axvline(0, color='black')
ax.set_title("K-medoid Clustering (K=2)")
plt.tight_layout()
plt.show()
```

#### 📊 Result

```
Final Clusters:
  Cluster 1: {A=2, B=4, C=5}  →  Medoid = 4 (Point B)
  Cluster 2: {D=10, E=12}     →  Medoid = 10 (Point D)
```

#### 🔍 Conclusion

- K-Medoid is preferred when the dataset contains noise or extreme values that would distort a mean-based centroid.

---

## Experiment 8
### Agglomerative (Hierarchical) Clustering

**Date:** 08/03/26 | Sheet No. 37–38

#### 🎯 Aim
Apply an **agglomerative clustering** approach to create a dendrogram for the following points using **single linkage**.

#### 📊 Dataset

| Point | X | Y |
|-------|---|---|
| A     | 1 | 1 |
| B     | 2 | 1 |
| C     | 4 | 3 |
| D     | 5 | 4 |

#### 📌 Objective
- Group data points into clusters by iteratively merging the closest cluster pair
- Visualize the process through a dendrogram using the **single linkage** method

#### 🔧 Materials & Methods

- **Tools:** Python 3.x, NumPy, Scikit-learn, SciPy, Matplotlib
- **Linkage:** Single linkage (minimum pairwise distance between clusters)

#### 💻 Code

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

points_label = ['a', 'b', 'c', 'd']
points_value = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])

print("Data points and value:")
for i, label in enumerate(points_label):
    print(f"{label}: {points_value[i]}")

agglomerative_model = AgglomerativeClustering(n_clusters=2, linkage='single')
agglomerative_model.fit(points_value)

print("\n--- Agglomerative Clustering Results ---")
print(f"Cluster labels: {agglomerative_model.labels_}")

cluster_labels = agglomerative_model.labels_
num_cluster    = len(np.unique(cluster_labels))

for i in range(num_cluster):
    cluster_points = [points_label[j] for j, label in enumerate(cluster_labels) if label == i]
    cluster_values = [points_value[j] for j, label in enumerate(cluster_labels) if label == i]
    print(f"{i+1}th Cluster {i+1}: points {cluster_points}, values {cluster_values}")

# Dendrogram
Z = linkage(points_value, method='single')
plt.figure(figsize=(7, 4))
dendrogram(Z, labels=points_label)
plt.title("Agglomerative Clustering Dendrogram (Single Linkage)")
plt.xlabel("Points"); plt.ylabel("Distance")
plt.tight_layout()
plt.show()
```

#### 📊 Result

```
Data point and value:
  a: [1, 1]    b: [2, 1]    c: [4, 3]    d: [5, 4]

--- Agglomerative Clustering Results ---
Cluster labels: [1 1 0 0]
Cluster 1: Points ['c', 'd'], Values [array([4,3]), array([5,4])]
Cluster 2: Points ['a', 'b'], Values [array([1,1]), array([2,1])]
```

#### 🔍 Conclusion

The code successfully demonstrated hierarchical agglomerative clustering using single linkage. The dendrogram accurately reflected the merge distances, matching the single linkage calculation.

---

## 📦 Requirements


```txt
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

Install all dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

---

## 👤 Author

| Field | Details |
|-------|---------|
| **Name** | Amandeep Kumar |
| **Univ. Roll** | 24UG010675 |
| **College Roll** | 24CSEAIML179 |
| **Semester** | 4th |
| **University** | JNTU Gunupur |

---

*All experiments implemented in Python using NumPy, Matplotlib, and Scikit-learn.*
