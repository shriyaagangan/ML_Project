# Import Libraries

import numpy as np
import pickle
import tensorflow as tf
import os
from flask import Flask, render_template, request, redirect, url_for, session 

# Create Flask Application

app = Flask(__name__)
app.secret_key = "pricesense"  # Change this to a random secret key

# Load the Trained Model

model = tf.keras.models.load_model("fuel_model.keras")
print("Model Loaded Successfully")

# Load Scaler

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
print("Scaler Loaded Successfully")

# Load Feature List

with open("features.pkl", "rb") as file:
    features = pickle.load(file)

print("Features Loaded Successfully")
print("Features:", features)

# Visitor Counter

VISITOR_FILE = "visitors.txt"
if not os.path.exists(VISITOR_FILE):
    with open(VISITOR_FILE, "w") as file:
        file.write("0")

def get_visitors():
    with open(VISITOR_FILE, "r") as file:
        return int(file.read())

def increase_visitors():
    count = get_visitors() + 1

    with open(VISITOR_FILE, "w") as file:
        file.write(str(count))

    return count

# Login Page

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        session["username"] = username
        increase_visitors()
        return redirect(url_for("dashboard"))
    
    return render_template("login.html")

# Dashboard

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session["username"],
        visitors=get_visitors()
    )

# Prediction Result

@app.route("/result", methods=["POST"])
def result():
    try:
        input_data = np.array([[
            float(request.form["year"]),
            float(request.form["month"]),
            float(request.form["week"]),
            float(request.form["day"]),
            float(request.form["country"]),
            float(request.form["region"]),
            float(request.form["income_level"]),
            float(request.form["subsidy_level"]),
            float(request.form["brent_crude_usd"]),
            float(request.form["tax_percentage"])
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        petrol = round(float(prediction[0][0]), 3)
        diesel = round(float(prediction[0][1]), 3)
        lpg = round(float(prediction[0][2]), 3)

        return render_template(
            "result.html",
            petrol=petrol,
            diesel=diesel,
            lpg=lpg,

            trend_graph="price_trend.png",
            loss_graph="training_validation_loss.png",
            mae_graph="training_validation_mae.png",
            petrol_graph="petrol_scatter.png",
            diesel_graph="diesel_scatter.png",
            lpg_graph="lpg_scatter.png"
        )

    except Exception as e:
        return f"Prediction Error: {e}"

# About Page

@app.route("/about")
def about():
    return render_template("about.html")

# Contact Page

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Logout

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# Run Application

if __name__ == "__main__":
    app.run(debug=True)
    
