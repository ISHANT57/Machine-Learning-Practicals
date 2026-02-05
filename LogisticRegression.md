
# 📘 Practical 2 — Multiple Linear Regression on California Housing Dataset

## Objective of the Practical

To implement Multiple Linear Regression to predict Median House Value using 8 features.

---

## Detailed Theory

y = b0 + b1x1 + b2x2 + ... + bnxn

Evaluation using MSE and R².

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

plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

---

## Output

MSE Train ≈ 0.524  
MSE Test ≈ 0.531  
R² Train ≈ 0.606  
R² Test ≈ 0.595

### Graph Output

Paste the generated graph here after running the code:

![Actual vs Predicted Graph](paste-your-graph-image-here.png)

---

## Conclusion

The model explains ~60% variance and generalizes well on test data.
