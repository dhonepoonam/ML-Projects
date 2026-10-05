import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Temperature": [60, 65, 70, 75, 80, 85, 90, 95],
    "Pressure": [30, 32, 35, 38, 40, 43, 45, 48],
    "Vibration": [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
    "Failure": [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Input features
X = df[["Temperature", "Pressure", "Vibration"]]

# Target
y = df["Failure"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict new machine condition
new_machine = [[82, 41, 4.2]]

prediction = model.predict(new_machine)

if prediction[0] == 1:
    print("Machine will FAIL")
else:
    print("Machine will WORK NORMALLY")
# Temperature vs Machine Failure

plt.bar(df["Temperature"], df["Failure"])

plt.xlabel("Temperature (°C)")
plt.ylabel("Failure (0 = Normal, 1 = Failure)")
plt.title("Machine Failure Based on Temperature")

plt.show()