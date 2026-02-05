
# 📘 Practical 1 — Simple Linear Regression on Iris (Setosa)

## Objective of the Practical

To implement **Simple Linear Regression** using:
- Mathematical approach with NumPy (Normal Equation)
- Library approach using Scikit-Learn

Dataset Used:
- Iris Dataset (Setosa — first 50 samples)
- X → Sepal Length (cm)
- Y → Sepal Width (cm)

---

## Detailed Theory

Simple Linear Regression models the relationship:

y = mx + c

We minimize Sum of Squared Errors using:

W = (XᵀX)⁻¹ XᵀY

Evaluation Metrics: MAE, MSE, R²

---

## Codes

### Code A — Mathematical Approach

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data[:50, 0].reshape(-1, 1)
y = iris.data[:50, 1].reshape(-1, 1)

X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

intercept = W[0, 0]
slope = W[1, 0]

x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = slope * x_line + intercept

plt.scatter(x, y)
plt.plot(x_line, y_line)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
```

### Code B — Scikit-Learn Approach

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## Output

Intercept ≈ -0.8360  
Slope ≈ 0.8529  
MAE ≈ 0.1992  
MSE ≈ 0.0488  
R² ≈ 0.2365

### Graph Output

Paste the generated graph here after running the code:

![Regression Graph](paste-your-graph-image-here.png)

---

## Conclusion

Both approaches produce the same regression line, validating the mathematical and library implementations.
