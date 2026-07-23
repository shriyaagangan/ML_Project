# import libraries

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

print("=" * 60)
print("PriceSense Fuel Price Prediction - Testing Module")
print("=" * 60)

# Load the Trained Model

model = tf.keras.models.load_model("fuel_model.keras")
print("Model Loaded Successfully.")

# Load Feature Scaler

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
print("Scaler Loaded Successfully")

# Load Feature List

with open("features.pkl", "rb") as file:
    features = pickle.load(file)
print("Feature List Loaded Successfully")
print("Features Used:")
print(features)

# Load testing Data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Testing Data Loaded Successfully.")
print("X_test Shape :", X_test.shape)
print("y_test Shape :", y_test.shape)

# Scale Testing Data (Feature Scaling)

X_test = scaler.transform(X_test)
print("Feature Scaling Applied Successfully")

# Predict Fuel Prices

predictions = model.predict(X_test)
print("Prediction Completed Successfully")

# Split Actual and Predicted Values

petrol_actual = y_test[:,0]
diesel_actual = y_test[:,1]
lpg_actual = y_test[:,2]

petrol_pred = predictions[:,0]
diesel_pred = predictions[:,1]
lpg_pred = predictions[:,2]
    
# Evaluation Function

def evaluate(actual, predicted, fuel_name):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\n==============================")
    print(f"{fuel_name} Performance")
    print("==============================")

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

# Evaluate Each Fuel

evaluate(
    petrol_actual,
    petrol_pred,
    "Petrol"
)

evaluate(
    diesel_actual,
    diesel_pred,
    "Diesel"
)

evaluate(
    lpg_actual,
    lpg_pred,
    "LPG"
)

# Scatter Plot: Actual vs Predicted

os.makedirs("static", exist_ok=True)

fuel_names = ["Petrol", "Diesel", "LPG"]
for i, fuel in enumerate(fuel_names):
    plt.figure(figsize=(5,5))
    plt.scatter(y_test[:, i], predictions[:, i], alpha=0.6,label="Predicted Values")
    plt.plot(
        [y_test[:, i].min(), y_test[:, i].max()],
        [y_test[:, i].min(), y_test[:, i].max()],
        "r--",
        label="Ideal Prediction"
    )
    plt.title(f"{fuel}: Actual vs Predicted")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"static/{fuel.lower()}_scatter.png", dpi=300)
    plt.show()
    plt.close()
    print(f"{fuel} Scatter Plot Saved Successfully")

print("\n" + "=" * 60)
print("PriceSense Testing Completed")
print("=" * 60)
