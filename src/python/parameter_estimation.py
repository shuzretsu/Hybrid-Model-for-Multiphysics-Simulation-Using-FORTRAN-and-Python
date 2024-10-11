from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Simulated data for parameter estimation
X = np.array([[1.0, 0.01], [1.5, 0.015], [2.0, 0.02]])  # Features: [density, viscosity]
y = np.array([0.01, 0.015, 0.02])  # Target: thermal diffusivity

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict based on user input
pred_density = float(input("Enter fluid density: "))
pred_viscosity = float(input("Enter fluid viscosity: "))
pred_features = np.array([[pred_density, pred_viscosity]])
predicted_diffusivity = model.predict(pred_features)

print(f"Predicted thermal diffusivity: {predicted_diffusivity[0]}")
