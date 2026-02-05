
# 📘 Practical 2 — Multiple Linear Regression on California Housing Dataset

## Objective of the Practical

To implement Multiple Linear Regression to predict Median House Value using 8 features.

---

## Theory

🔹 What is Multiple Linear Regression?

When multiple independent variables affect the dependent variable:

𝑦
=
𝑏
0
+
𝑏
1
𝑥
1
+
𝑏
2
𝑥
2
+
⋯
+
𝑏
𝑛
𝑥
𝑛
y=b
0
	​

+b
1
	​

x
1
	​

+b
2
	​

x
2
	​

+⋯+b
n
	​

x
n
	​


Where:

𝑏
0
b
0
	​

 = intercept

𝑏
𝑖
b
i
	​

 = coefficient of each feature

🔹 Matrix Representation
𝑌
=
𝑋
𝑊
Y=XW

Solution using Normal Equation:

𝑊
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑌
W=(X
T
X)
−1
X
T
Y
🔹 Why Multiple Features?

Real-world problems depend on many factors.
Example: House price depends on income, rooms, age, location, etc.

🔹 Evaluation Metrics

Same metrics used:

MSE — error measurement

R² — how well model explains variance

If:

R² ≈ 0.60 → 60% variation explained

Close Train/Test MSE → model is well generalized

🔹 Perfect Fit Line (Graph Concept)

In Actual vs Predicted graph, ideal points lie on:

𝑦
=
𝑥
y=x

This is called Perfect Fit Line.

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

<img width="729" height="591" alt="image" src="https://github.com/user-attachments/assets/9f1e0429-9358-440b-ad87-d40addac4eb8" />


---

## Conclusion

The model explains ~60% variance and generalizes well on test data.
