
# 📘 Practical 2 — Multiple Linear Regression on California Housing Dataset

## Objective of the Practical

To implement **Multiple Linear Regression** to predict **Median House Value** using 8 independent features from the California Housing dataset.

---

## Theory

### What is Multiple Linear Regression?

Multiple Linear Regression models the relationship between multiple independent variables and one dependent variable:

y = b0 + b1x1 + b2x2 + ... + bnxn

Where:
- b0 = intercept
- bi = coefficient of each feature
- xi = independent variables

### Matrix Representation

Y = XW

W = (XᵀX)⁻¹ XᵀY

### Evaluation Metrics

MAE = (1/n) Σ|yi − ŷi|  
MSE = (1/n) Σ(yi − ŷi)²  
R² = 1 − (SSres / SStot)

If R² ≈ 0.60 → 60% variation explained.

---

## Codes

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("MSE Train:", mean_squared_error(y_train, y_train_pred))
print("MSE Test:", mean_squared_error(y_test, y_test_pred))
print("R2 Train:", r2_score(y_train, y_train_pred))
print("R2 Test:", r2_score(y_test, y_test_pred))

plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)], color='red')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Perfect Fit Line)")
plt.show()
```

---

## Output

MSE Train ≈ 0.524  
MSE Test ≈ 0.531  
R² Train ≈ 0.606  
R² Test ≈ 0.595

---

## Conclusion

The model explains about 60% of the variance in housing prices and generalizes well on test data.
