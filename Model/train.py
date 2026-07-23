#  import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Read the dataset

try:
    df = pd.read_csv("global_fuel_prices_2020_2026.csv")
    print("Dataset Loaded Successfully")
    print("Dataset Shape :", df.shape)

except FileNotFoundError:
    print("Dataset not found.")
    print("Creating Mock Dataset...")

    np.random.seed(42)

    date_range = pd.date_range(
        start="2020-01-01",
        end="2026-04-01",
        freq="W"
    )

    countries = [
        "India",
        "United States",
        "Canada",
        "Germany",
        "United Kingdom",
        "Australia",
        "Japan"
    ]

    regions = [
        "Asia",
        "North America",
        "Europe",
        "Oceania"
    ]

    income_levels = [
        "High",
        "Upper Middle",
        "Lower Middle"
    ]

    subsidy_levels = [
        "High",
        "Medium",
        "Low"
    ]

    mock_data = []

    for country in countries:
        for dt in date_range:
            region = np.random.choice(regions)
            income = np.random.choice(income_levels)
            subsidy = np.random.choice(subsidy_levels)
            brent = np.random.uniform(50,110)
            tax = np.random.uniform(5,40)
            petrol = (
                0.018 * brent
                + 0.020 * tax
                + np.random.normal(0,0.08)
            )

            diesel = (
                0.016 * brent
                + 0.017 * tax
                + np.random.normal(0,0.07)
            )

            lpg = (
                0.012 * brent
                + 0.010 * tax
                + np.random.normal(0,0.05)
            )

            mock_data.append([
                dt,
                country,
                region,
                income,
                subsidy,
                round(petrol,3),
                round(diesel,3),
                round(lpg,3),
                round(brent,2),
                round(tax,2)
            ])

    df = pd.DataFrame(
        mock_data,
        columns=[
            "date",
            "country",
            "region",
            "income_level",
            "subsidy_level",
            "petrol_usd_liter",
            "diesel_usd_liter",
            "lpg_usd_liter",
            "brent_crude_usd",
            "tax_percentage"
        ]
    )

    df.to_csv(
        "global_fuel_prices_2020_2026.csv",
        index=False
    )

    print("Mock Dataset Created Successfully")
    print(f"Dataset Shape : {df.shape}")
 

#  Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

#  Extract time-based Features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day"] = df["date"].dt.day

# Convert categorical (text) columns into numerical values
# because the neural network accepts only numerical input.
df["country"] = df["country"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes
df["income_level"] = df["income_level"].astype("category").cat.codes
df["subsidy_level"] = df["subsidy_level"].astype("category").cat.codes

# Remove rows with missing values in important columns
df = df.dropna(subset=[
    "petrol_usd_liter",
    "diesel_usd_liter",
    "lpg_usd_liter",
    "brent_crude_usd",
    "tax_percentage"
])
print(f"Data Preprocessing Completed. Remaining Samples: {df.shape[0]}")

# Visualization

os.makedirs("static", exist_ok=True)
plt.figure(figsize=(12,6))

sns.lineplot(data=df, x="date", y="petrol_usd_liter", label="Petrol")
sns.lineplot(data=df, x="date", y="diesel_usd_liter", label="Diesel")
sns.lineplot(data=df, x="date", y="lpg_usd_liter", label="LPG")
# Brent Crude is not a direct type of fuel, but rather a high-quality, unrefined crude oil extracted from the North Sea.
# sns.lineplot(data=df, x="date", y="brent_crude_usd", label="Brent Crude")
plt.title("PriceSense Analysis: Global Fuel Price Trends (2020–2026)")
plt.xlabel("Date")
plt.ylabel("Price (USD/Litre)")
plt.grid(True)
plt.legend()
plt.savefig("static/price_trend.png", dpi=300)
plt.show()
plt.close()
print("Price Trend Graph Saved Successfully")

# Feature Selection

features = [
    "year",
    "month",
    "week_of_year",
    "day",
    "country",
    "region",
    "income_level",
    "subsidy_level",
    "brent_crude_usd",
    "tax_percentage"
]
X = df[features].values
# Multi Output Target
y = df[
    [
        "petrol_usd_liter",
        "diesel_usd_liter",
        "lpg_usd_liter"
    ]
].values

# Train Test Split(Split data)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Samples :", len(X_train))
print("Validation Samples :", len(X_test))


# Save the testing arrays out to disk so 'test.py' can pick them up blindly later
# ( .npyis NumPy's binary file format.It stores arrays efficiently.)
# Save test data

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
# print("Training/Testing vectors partitioned. Test arrays cached to disk array profiles.")
print("Test Data Saved Successfully")


# Feature Scaling using StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature Scaling Completed Successfully")

# Build Deep Learning Model

model = Sequential()

model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))

# Output Layer (Predict Petrol, Diesel and LPG Prices)
model.add(Dense(3, activation="linear"))
print("Model Created Successfully")

# Compile Model

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)
print("Model Compiled Successfully")

# Train Model

print("Training Started ...")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1
)
print("Training Completed Successfully")

# Training and Validation Loss Graph

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("PriceSense: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("static/training_validation_loss.png", dpi=300)
plt.show()
plt.close()
print("Training and Validation Loss Graph Saved Successfully")

# Training and Validation MAE Graph

plt.figure(figsize=(10,5))
plt.plot(history.history["mean_absolute_error"], label="Training MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
plt.title("PriceSense: Training vs Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(True)
plt.savefig("static/training_validation_mae.png", dpi=300)
plt.show()
plt.close()
print("Training and Validation MAE Graph Saved Successfully")

# Save the trained model

model.save("fuel_model.keras")
print("Model Saved Successfully")

# Save Scaler

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
print("Scaler Saved Successfully")

# Save Feature List

with open("features.pkl", "wb") as file:
    pickle.dump(features, file)
print("Feature List Saved Successfully")

print("\n----- PriceSense Training Completed -----")
print("Generated Files:")

print("static/price_trend.png")
print("static/training_validation_loss.png")
print("static/training_validation_mae.png")
print("X_test.npy")
print("y_test.npy")
print("fuel_model.keras")
print("scaler.pkl")
print("features.pkl")

