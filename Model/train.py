# Import Libraries
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
warnings.filterwarnings("ignore")

# Project Banner
print("=" * 60)
print(" PriceSense - Dynamic Fuel Price Forecast Platform ")
print(" Model Training Module ")
print("=" * 60)

# Create Required Folders
print("\nCreating Project Folders...")
os.makedirs("static", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)
print("Folders Created Successfully.")

# Load Dataset
print("\nLoading Dataset...")
try:
    df = pd.read_csv("global_fuel_prices_2020_2026.csv")
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Dataset File Not Found!")
    exit()

# Display Dataset Information
print("\nDataset Information")
df.info()
print("\nDataset Shape")
print(df.shape)
print("\nFirst Five Records")
print(df.head())
print("\nLast Five Records")
print(df.tail())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Records
print("\nChecking Duplicate Records...")
duplicate = df.duplicated().sum()
print("\nDuplicate Rows :", duplicate)
if duplicate > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate Rows Removed.")
else:
    print("No Duplicate Rows Found.")
    
# Convert Date Column
print("\nConverting Date Column...")
df["date"] = pd.to_datetime(df["date"])
print("Date Converted Successfully.")

# Feature 
print("\nCreating Date Features...")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day"] = df["date"].dt.day
print("Date Features Created Successfully.")

# Remove Missing Values
important_columns = [
    "petrol_usd_liter",
    "diesel_usd_liter",
    "brent_crude_usd",
    "tax_percentage"
]
df = df.dropna(subset=important_columns)
print("\nRemaining Records :", len(df))

# Dataset Summary
print("\nDataset Summary")
print(df.describe())
print("=" * 60)

# Exploratory Data Analysis (EDA)

print("\nGenerating Graphs...")

# 1. Fuel Price Trend
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["petrol_usd_liter"], label="Petrol", color="blue")
plt.plot(df["date"], df["diesel_usd_liter"], label="Diesel", color="green")
plt.title("Fuel Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD/Litre)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/price_trend.png")
plt.show()
plt.close()
print("Price Trend Graph Saved Successfully.")

# 2. Petrol Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["petrol_usd_liter"], bins=25, kde=True, color="royalblue")
plt.title("Petrol Price Distribution")
plt.xlabel("Petrol Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/graphs/petrol_distribution.png")
plt.show()
plt.close()
print("Petrol Distribution Graph Saved Successfully.")

# 3. Diesel Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["diesel_usd_liter"], bins=25, kde=True, color="darkgreen")
plt.title("Diesel Price Distribution")
plt.xlabel("Diesel Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/graphs/diesel_distribution.png")
plt.show()  
plt.close()
print("Diesel Distribution Graph Saved Successfully.")

# 4. Brent Crude Price Trend
plt.figure(figsize=(12,5))
plt.plot(df["date"], df["brent_crude_usd"], color="red")
plt.title("Brent Crude Oil Price")
plt.xlabel("Date")
plt.ylabel("USD")
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/brent_price.png")
plt.show()
plt.close()
print("Brent Price Graph Saved Successfully.")

# 5. Monthly Average Fuel Price
monthly_avg = df.groupby("month")[["petrol_usd_liter","diesel_usd_liter"]].mean()
plt.figure(figsize=(10,5))
plt.plot(monthly_avg.index, monthly_avg["petrol_usd_liter"], marker="o", label="Petrol")
plt.plot(monthly_avg.index, monthly_avg["diesel_usd_liter"], marker="s", label="Diesel")
plt.title("Monthly Average Fuel Price")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/monthly_average.png")
plt.show()
plt.close()
print("Monthly Average Graph Saved Successfully.")

# 6. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/graphs/correlation_heatmap.png")
plt.show()
plt.close()
print("Heatmap Graph Saved Successfully.")

# Feature Selection
print("\nSelecting Features...")
features = [
    "year",
    "month",
    "week_of_year",
    "brent_crude_usd",
    "tax_percentage"
]
target = [
    "petrol_usd_liter",
    "diesel_usd_liter"
]
X = df[features]
y = df[target]
print("Features Selected Successfully.")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# Save Testing Data
print("\nSaving Testing Dataset...")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
print("Testing Data Saved Successfully.")

# Feature Scaling
print("\nScaling Features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature Scaling Completed.")

# Save Scaler
print("\nSaving Scaler...")
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
print("Scaler Saved Successfully.")

# Save Feature List
print("\nSaving Feature Names...")
with open("features.pkl", "wb") as file:
    pickle.dump(features, file)
print("Feature List Saved.")
print("\nNumber of Features :", len(features))
print("=" * 60)
print("Data Preparation Completed Successfully")
print("=" * 60)

# Build Deep Learning Model
print("\nBuilding Deep Learning Model...")
model = Sequential()
# Input Layer
model.add(Dense(
    64,
    activation="relu",
    input_shape=(X_train.shape[1],)
))
# Hidden Layer 1
model.add(Dropout(0.20))
model.add(Dense(
    32,
    activation="relu"
))
# Hidden Layer 2
model.add(Dense(
    16,
    activation="relu"
))
# Output Layer
model.add(Dense(
    2,
    activation="linear"
))
print("\nModel Architecture")
print("Model Created Successfully.")

# Compile Model
print("\nCompiling Model...")
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)
print("Model Compiled Successfully.")

# Train Model
print("\nTraining Model...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
print("Training History Saved.")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Loss :", round(loss,4))
print("Test MAE  :", round(mae,4))
print("\nModel Training Completed.")

# Plot Training Loss
plt.figure(figsize=(8,5))
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/loss_curve.png")
plt.show()
plt.close()
print("Loss Graph Saved.")

# Plot Mean Absolute Error
plt.figure(figsize=(8,5))
plt.plot(
    history.history["mae"],
    label="Training MAE"
)
plt.plot(
    history.history["val_mae"],
    label="Validation MAE"
)
plt.title("Model MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/graphs/mae_curve.png")
plt.show()
plt.close()
print("MAE Graph Saved.")

# Save Trained Model
print("\nSaving Model...")
model.save("fuel_model.keras")
print("Model Saved Successfully.")

# Training Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print("Dataset Records :", len(df))
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))
print("Number of Features :", X_train.shape[1])
print("Output Variables : Petrol Price, Diesel Price")
print("\nGenerated Files:-")
print("fuel_model.keras")
print("scaler.pkl")
print("features.pkl")
print("X_test.npy")
print("y_test.npy")
print("\nGenerated Graphs:-")
print("price_trend.png")
print("petrol_distribution.png")
print("diesel_distribution.png")
print("brent_price.png")
print("monthly_average.png")
print("correlation_heatmap.png")
print("loss_curve.png")
print("mae_curve.png")
print("\nModel Training Completed Successfully.")
print("="*60)
print("PriceSense Training Module Finished")
print("="*60)


