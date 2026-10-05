import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Performance": [5, 5, 6, 6, 7, 8, 8, 9],
    "Training_Hours": [10, 15, 20, 25, 30, 35, 40, 45],
    "Projects": [1, 1, 2, 2, 3, 4, 5, 6],
    "Promotion": [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Input features
X = df[["Experience", "Performance", "Training_Hours", "Projects"]]

# Target
y = df["Promotion"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# New employee prediction
new_employee = [[5, 8, 35, 4]]

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("Employee will get PROMOTION")
else:
    print("Employee will NOT get PROMOTION")

# Pie Chart - Promotion Status

promotion_count = df["Promotion"].value_counts()

plt.pie(
    promotion_count,
    labels=["No Promotion", "Promotion"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Employee Promotion Prediction")
plt.show()