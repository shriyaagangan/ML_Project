# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# import pickle
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, r2_score

# import os

# # Read the dataset
# try:
#     df = pd.read_csv("global_fuel_prices_2020_2026.csv")
#     print("Provided Data Imported Successfully")
#     print("Data Shape: " , df.shape)
    
# except FileNotFoundError:
#     print("Error: File not found")
#     date_range = pd.date_range(start="2020-01-01", end="2026-04-01", freq="W")
#     countries = ["United States", "Canada", "India", "Germany", "United Kingdom"]

#     mock_data = []

#     for country in countries:
#         for dt in date_range:
#             brent = 60 + 30*np.sin(dt.dayofyear/365) + np.random.normal(0, 5)
#             tax = np.random.uniform(10, 45)

#             petrol = (brent * 0.015) + (tax * 0.02) + np.random.normal(0, 0.1)
#             diesel = (brent * 0.013) + (tax * 0.015) + np.random.normal(0, 0.08)

#             mock_data.append([
#                 dt,
#                 country,
#                 "RegionX",
#                 "High",
#                 "Low",
#                 petrol,
#                 diesel,
#                 0.80,
#                 brent,
#                 tax
#             ])

#     df = pd.DataFrame(mock_data, columns=[
#         "date",
#         "country",
#         "region",
#         "income_level",
#         "subsidy_level",
#         "petrol_usd_liter",
#         "diesel_usd_liter",
#         "lpg_usd_liter",
#         "brent_crude_usd",
#         "tax_percentage"
#     ])
#     # print("Mock Data Created Successfully")
#     # check the print state,=ment delete it maybe afterwards.

# # Convert date column to datetime
# df['date'] = pd.to_datetime(df['date'])

# # Extract time-based features
# df['year'] = df['date'].dt.year
# df['month'] = df['date'].dt.month
# df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

# # Remove rows with missing values in important columns
# df = df.dropna(subset=[
#     'petrol_usd_liter',
#     'diesel_usd_liter',
#     'brent_crude_usd',
#     'tax_percentage'
# ])

# # print("Data droped is Completed.")


# os.makedirs("static", exist_ok=True)
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=df, x='date', y='petrol_usd_liter', label='Petrol Price (USD/L)')
# sns.lineplot(data=df, x='date', y='brent_crude_usd', color='orange', label='Brent Crude (USD/Barrel)')
# plt.title('PriceSense Analysis: Fuel Pricing Over Time vs Brent Benchmark')
# plt.xlabel("Date")
# plt.ylabel("Price (USD)")
# plt.savefig('static/price_trend.png', dpi=300)
# plt.show()
# plt.close()

# # Feature Selection
# features = ['year', 'month', 'week_of_year', 'brent_crude_usd', 'tax_percentage']
# X = df[features].values
# # Target
# y = df['petrol_usd_liter'].values

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, 
#     y, 
#     test_size=0.2, 
#     random_state=42
# )

# # Save the testing arrays out to disk so 'test.py' can pick them up blindly later( .npyis NumPy's binary file format.It stores arrays efficiently.)
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)
# print("Training/Testing vectors partitioned. Test arrays cached to disk array profiles.")

# # scaler (ROBUST FEATURE SCALING)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Deep Learning Model
# model = Sequential()

# model.add(tf.keras.Input(shape=(X_train.shape[1],)))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# print("Model architecture defined.")

# # Compile
# model.compile(
#     optimizer='adam',
#     loss='mean_squared_error',
#     metrics=['mean_absolute_error']
# )

# model.fit(
#     X_train,
#     y_train,
#     epochs=100,
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     verbose=1
# )
# # Save the trained model

# # here instead of saving the model with '.h5' extension, we will use the '.keras' extension which is the recommended format for saving Keras models in TensorFlow 2.x
# # This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` 
# # or `keras.saving.save_model(model, 'my_model.keras')` 
# # ==== by vscode snice it recommends to use the native Keras format instead of the legacy HDF5 format. 
# # The native Keras format is more efficient and supports additional features, such as saving the model architecture, weights, and training configuration in a single file. 
# # It also allows for better compatibility with future versions of Keras and TensorFlow.
# # Therefore, it is recommended to use the native Keras format for saving models in TensorFlow 2.x.

# model.save("fuel_model.keras")
# print("Core neural network structures compiled and saved to 'fuel_model.keras'")

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# print("\nProject training completed successfully.")
# print("All files generated successfully.")




# =========================or=======================================


# 1. import libraries
from datetime import date

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
    print("Dataset Shape :", df.shape)
 

#  Convert date column to datetime

df["date"] = pd.to_datetime(df["date"])

#  Extract time-based Features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day"] = df["date"].dt.day

# Encode Categorical Columns (see if needed or useful otherwise remove)
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

print("Data droped is Completed.")


# Visualization

os.makedirs("static", exist_ok=True)
plt.figure(figsize=(12,6))

sns.lineplot(data=df, x="date", y="petrol_usd_liter", label="Petrol")
sns.lineplot(data=df, x="date", y="diesel_usd_liter", label="Diesel")
sns.lineplot(data=df, x="date", y="lpg_usd_liter", label="LPG")
# Brent Crude is not a direct type of fuel, but rather a high-quality, unrefined crude oil extracted from the North Sea.
sns.lineplot(data=df, x="date", y="brent_crude_usd", label="Brent Crude")

plt.title("PriceSense Analysis: Fuel Price Trends")
plt.xlabel("Date")
plt.ylabel("Price (USD/Litre)")
plt.grid(True)
plt.legend()
plt.savefig("static/price_trend.png", dpi=300)
plt.show()
plt.close()
print("Graph Saved Successfully")


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
print("Test data is saved successfully")


# Feature Scaling ((ROBUST FEATURE SCALING))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature Scaling Completed")

# Neural Network
# Deep Learning Model

model = Sequential()

model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))

# Three Outputs=3
model.add(Dense(3, activation="linear"))

print("Model Created Successfully")



# Compile Model

model.compile(

    optimizer="adam",

    loss="mean_squared_error",

    metrics=["mean_absolute_error"]

)

print("Model Compiled Successfully")

# ==========================================
# Train Model
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=100,

    batch_size=32,

    verbose=1

)

print("Training Completed Successfully")

# ==========================================
# Save Model
# ==========================================

model.save("fuel_model.keras")

print("Model Saved")

# Save Scaler

with open("scaler.pkl", "wb") as file:

    pickle.dump(scaler, file)

print("Scaler Saved")

# Save Feature List

with open("features.pkl", "wb") as file:

    pickle.dump(features, file)

print("Feature List Saved")

print("\n===================================")
print("PriceSense Training Completed")
print("===================================")
print("Generated Files:")
print("✔ fuel_model.keras")
print("✔ scaler.pkl")
print("✔ features.pkl")
print("✔ static/price_trend.png")

