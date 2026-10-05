import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# X and y
X = df[["Study_Hours"]]
y = df["Pass"]

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Prediction
hours = [[4]]
prediction = model.predict(hours)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")

# Graph
plt.scatter(df["Study_Hours"], df["Pass"])
plt.xlabel("Study Hours")
plt.ylabel("Pass (1) / Fail (0)")
plt.title("Study Hours vs Pass/Fail")
plt.show()