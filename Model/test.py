import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("PriceSense Fuel Price Prediction - Testing Module")
print("=" * 60)

try:
    # Load test data
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    print("Test data loaded successfully.")
    print("X_test Shape :", X_test.shape)
    print("y_test Shape :", y_test.shape)
    
    # Load the scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("Scaler loaded successfully.")

    # Load the trained model
    model = tf.keras.models.load_model("fuel_model.keras")

    print("Model loaded successfully.")
    
    # Scale the test data
    X_test = scaler.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test).flatten()

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance : ")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"R² Score                 : {r2:.4f}")
    print("-" * 40)

    print("\nTesting completed successfully.")
    
except FileNotFoundError:
    print("Error: One or more required files are missing.")
    print("Please ensure that 'X_test.npy', 'y_test.npy', 'scaler.pkl', and 'fuel_model.keras' are present in the directory.")