import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, request

# Create Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("fuel_model.keras")

# Load the saved scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from HTML form
        year = float(request.form["year"])
        month = float(request.form["month"])
        week = float(request.form["week"])
        brent = float(request.form["brent_crude_usd"])
        tax = float(request.form["tax_percentage"])

        # Store values in NumPy array
        input_data = np.array([[year, month, week, brent, tax]])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict fuel price
        prediction = model.predict(input_scaled)

        # Get predicted value
        predicted_price = round(prediction[0][0], 2)

        # Display result on webpage
        return render_template(
            "index.html",
            prediction=f"Predicted Fuel Price: $ {predicted_price} USD/Liter"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {e}"
        )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
