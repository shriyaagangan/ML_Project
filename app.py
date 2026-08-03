# Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

# Create Flask Application(app)
app = Flask(__name__)
app.secret_key =  "pricesense123"  

# Create Required Folder
os.makedirs("static", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# Load the Trained Model
print("=" * 60)
print("Loading PriceSense Model...")
print("=" * 60)
model = None
try:
    model = tf.keras.models.load_model("fuel_model.keras")
    print("Model Loaded Successfully.")
except Exception as error:
    print("Model could not be loaded.")
    print(error)

# Load Scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
print("Scaler Loaded Successfully")

# Load Feature List
with open("features.pkl", "rb") as file:
    features = pickle.load(file)
print("Features Loaded Successfully")
print("Number of Features:", len(features))

# Create Prediction History File
history_file = "prediction_history.csv"
if not os.path.exists(history_file):
    history = pd.DataFrame(columns=[
        "Year",
        "Month",
        "Week",
        "Brent Crude",
        "Tax Percentage",
        "Predicted Petrol",
        "Predicted Diesel"
    ])
    history.to_csv(history_file, index=False)
    print("Prediction History File Created.")
else:
    print("Prediction History File Found.")
    
# Dummy Login Credentials
USERNAME = "admin"
PASSWORD = "admin123"
print("=" * 60)
print("PriceSense Web Application Ready")
print("=" * 60)

# Login Page
@app.route("/", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["username"] = username
            flash("Login Successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid Username or Password", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")

# Dashboard
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    total_features = len(features)
    model_name = "Fuel Price Prediction Model"
    output_variables = 2
    return render_template(
        "dashboard.html",
        username=session["username"],
        total_features=total_features,
        model_name=model_name,
        output_variables=output_variables
    )
    
# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged Out Successfully.", "info")
    return redirect(url_for("login"))
    
# Prediction Page
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "username" not in session:
        return redirect(url_for("login"))
    if model is None:
        flash("Model is not available.", "danger")
        return redirect(url_for("dashboard"))
    petrol_price = None
    diesel_price = None
    if request.method == "POST":
        try:
            # Read User Input
            year = int(request.form["year"])
            month = int(request.form["month"])
            week = int(request.form["week"])
            brent = float(request.form["brent"])
            tax = float(request.form["tax"])
            # Create Input Array
            user_input = np.array([[
                year,
                month,
                week,
                brent,
                tax
            ]])
            # Scale Input
            user_input = scaler.transform(user_input)
            # Predict Prices
            prediction_result = model.predict(user_input, verbose=0)
            petrol_price = round(float(prediction_result[0][0]), 2)
            diesel_price = round(float(prediction_result[0][1]), 2)
            # Save Prediction History
            new_record = pd.DataFrame({
                "Year": [year],
                "Month": [month],
                "Week": [week],
                "Brent Crude": [brent],
                "Tax Percentage": [tax],
                "Predicted Petrol": [petrol_price],
                "Predicted Diesel": [diesel_price]
            })
            history_data = pd.read_csv(history_file)
            history_data = pd.concat(
                [history_data, new_record],
                ignore_index=True
            )
            history_data.to_csv(history_file, index=False)
            flash(
                "Prediction Generated Successfully.",
                "success"
            )
        except Exception as error:
            print(error)
            flash(
                "Invalid Input. Please Try Again.",
                "danger"
            )
    return render_template(
        "prediction.html",
        username=session["username"],
        petrol_price=petrol_price,
        diesel_price=diesel_price
    )

# Prediction History
@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    history_data = pd.read_csv(history_file)
    return render_template(
        "history.html",
        username=session["username"],
        tables=history_data.to_html(
            classes="table table-bordered table-striped",
            index=False
        )
    )

# Analysis Page
@app.route("/analysis")
def analysis():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "analysis.html",
        username=session["username"],
        trend_graph="price_trend.png",
        petrol_distribution="petrol_distribution.png",
        diesel_distribution="diesel_distribution.png",
        brent_graph="brent_price.png",
        monthly_graph="monthly_average.png",
        heatmap="correlation_heatmap.png",
        loss_graph="loss_curve.png",
        mae_graph="mae_curve.png",
        petrol_prediction="petrol_prediction.png",
        diesel_prediction="diesel_prediction.png",
        petrol_error="petrol_error.png",
        diesel_error="diesel_error.png"
    )

# Report Page
@app.route("/report")
def report():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "report.html",
        username=session["username"]
    )
    
# About Page
@app.route("/about")
def about():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "about.html",
        username=session["username"]
    )
    
# Run Application
if __name__ == "__main__":
    app.run(
        debug=True
    )

