# Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import warnings
warnings.filterwarnings("ignore")

# Project Banner
print("=" * 60)
print("PriceSense Fuel Price Prediction - Testing Module")
print("=" * 60)

# Create Graph Folder
os.makedirs("static", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# Load the Trained Model
print("\nLoading Trained Model...")
model = tf.keras.models.load_model("fuel_model.keras")
print("Model Loaded Successfully.")

# Display Model Architecture
model.summary()

# Load Scaler
print("\nLoading Scaler...")
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
print("Scaler Loaded Successfully")

# Load Feature List
print("\nLoading Feature List...")
with open("features.pkl", "rb") as file:
    features = pickle.load(file)
print("Feature List Loaded Successfully")

# Load Testing Data
print("\nLoading Testing Data...")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
print("Testing Data Loaded Successfully.")
print("X_test Shape :", X_test.shape)
print("y_test Shape :", y_test.shape)
print("Number of Test Samples :", len(X_test))
print("Number of Features :", X_test.shape[1])
print("Number of Output Variables :", y_test.shape[1])

# Predict Fuel Prices
print("\nPredicting Fuel Prices...")
predictions = model.predict(X_test)
print("Prediction Shape :", predictions.shape)
print("Prediction Completed Successfully")

# Model Performance Evaluation
print("\nCalculating Model Performance...")
# Petrol Metrics
petrol_mae = mean_absolute_error(y_test[:, 0], predictions[:, 0])
petrol_mse = mean_squared_error(y_test[:, 0], predictions[:, 0])
petrol_rmse = np.sqrt(petrol_mse)
petrol_r2 = r2_score(y_test[:, 0], predictions[:, 0])
# Diesel Metrics
diesel_mae = mean_absolute_error(y_test[:, 1], predictions[:, 1])
diesel_mse = mean_squared_error(y_test[:, 1], predictions[:, 1])
diesel_rmse = np.sqrt(diesel_mse)
diesel_r2 = r2_score(y_test[:, 1], predictions[:, 1])

# Display Results
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print("\nPetrol Price Prediction")
print("-" * 30)
print("MAE  :", round(petrol_mae, 4))
print("MSE  :", round(petrol_mse, 4))
print("RMSE :", round(petrol_rmse, 4))
print("R² Score :", round(petrol_r2, 4))
print("\nDiesel Price Prediction")
print("-" * 30)
print("MAE  :", round(diesel_mae, 4))
print("MSE  :", round(diesel_mse, 4))
print("RMSE :", round(diesel_rmse, 4))
print("R² Score :", round(diesel_r2, 4))

# Create Prediction DataFrame
results = {
    "Actual Petrol": y_test[:, 0],
    "Predicted Petrol": predictions[:, 0],
    "Actual Diesel": y_test[:, 1],
    "Predicted Diesel": predictions[:, 1]
}
result_df = pd.DataFrame(results)
print("\nSample Prediction Results")
print(result_df.head())

# Save Prediction Results
result_df.to_csv("prediction_results.csv", index=False)
print("\nPrediction Results Saved Successfully.")

# Actual vs Predicted Petrol Price
plt.figure(figsize=(10,5))
plt.plot(
    y_test[:,0],
    label="Actual Petrol",
    color="blue"
)
plt.plot(
    predictions[:,0],
    label="Predicted Petrol",
    color="red"
)
plt.title("Actual vs Predicted Petrol Price")
plt.xlabel("Test Samples")
plt.ylabel("Price (USD/Litre)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/petrol_prediction.png")
plt.show()
plt.close()
print("Petrol Prediction Graph Saved Successfully.")

# Actual vs Predicted Diesel Price
plt.figure(figsize=(10,5))
plt.plot(
    y_test[:,1],
    label="Actual Diesel",
    color="green"
)
plt.plot(
    predictions[:,1],
    label="Predicted Diesel",
    color="orange"
)
plt.title("Actual vs Predicted Diesel Price")
plt.xlabel("Test Samples")
plt.ylabel("Price (USD/Litre)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/diesel_prediction.png")
plt.show()
plt.close()
print("Diesel Prediction Graph Saved Successfully.")

# Petrol Prediction Error
petrol_error = y_test[:,0] - predictions[:,0]
plt.figure(figsize=(8,5))
plt.hist(
    petrol_error,
    bins=20,
    color="skyblue"
)
plt.title("Petrol Prediction Error")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/graphs/petrol_error.png")
plt.show()
plt.close()
print("Petrol Error Graph Saved Successfully.")

# Diesel Prediction Error
diesel_error = y_test[:,1] - predictions[:,1]
plt.figure(figsize=(8,5))
plt.hist(
    diesel_error,
    bins=20,
    color="lightgreen"
)
plt.title("Diesel Prediction Error")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/graphs/diesel_error.png")
plt.show()
plt.close()
print("Diesel Error Graph Saved Successfully.")

# Testing Summary
print("\n" + "=" * 60)
print("TESTING SUMMARY")
print("=" * 60)
print("Testing Samples :", len(X_test))
print("Number of Features :", len(features))
print("\nPetrol R² Score :", round(petrol_r2,4))
print("Diesel R² Score :", round(diesel_r2,4))
print("\nGraphs Generated")
print("----------------------------")
print("petrol_prediction.png")
print("diesel_prediction.png")
print("petrol_error.png")
print("diesel_error.png")
print("\nPrediction File")
print("----------------------------")
print("prediction_results.csv")
print("\nTesting Completed Successfully.")
print("=" * 60)
print("PriceSense Testing Module Finished")
print("=" * 60)



