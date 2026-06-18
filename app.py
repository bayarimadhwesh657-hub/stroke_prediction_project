from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import joblib

app = Flask(__name__)


# LOAD MODEL + PREPROCESSORS

model = joblib.load("stroke_model_compressed.joblib")
imputer = pickle.load(open("imputer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# DATABASE SETUP


def create_table():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            hypertension INTEGER,
            heart_disease INTEGER,
            glucose REAL,
            bmi REAL,
            prediction TEXT,
            percentage REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

create_table()


# HOME PAGE


@app.route("/")
def home():
    return render_template("index.html")


# PREDICTION ROUTE

@app.route("/predict", methods=["POST"])
def predict():


    # GET USER INPUT

    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    glucose = float(request.form["glucose"])
    bmi = float(request.form["bmi"])

    # PREPARE INPUT
    features = np.array([[age, hypertension, heart_disease, glucose, bmi]])

    # Apply preprocessing
    features = imputer.transform(features)
    features = scaler.transform(features)

    # MODEL PREDICTION
    probability = model.predict_proba(features)[0][1] * 100

    print("ML Probability:", round(probability, 2))

    # MEDICAL RULES LOGIC

    high_risk = (
        (hypertension == 1 and heart_disease == 1)
        or (glucose >= 180 and bmi >= 35)
        or (hypertension == 1 and glucose >= 180)
        or (heart_disease == 1 and glucose >= 180)
        or (age >= 60 and hypertension == 1)
    )

    moderate_risk = (
        age >= 45
        or hypertension == 1
        or heart_disease == 1
        or glucose >= 120
        or bmi >= 27
    )

    # FINAL RESULT
    if high_risk:
        result = "Unhealthy (High Stroke Risk)"
        final_score = 85

    elif moderate_risk:
        result = "Moderate Risk"
        final_score = 55

    else:
        result = "Healthy"
        final_score = 20

    print("Final Result:", result)
    print("Final Score:", final_score)

    # SAVE TO DATABASE
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_history(
            age,
            hypertension,
            heart_disease,
            glucose,
            bmi,
            prediction,
            percentage,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        age,
        hypertension,
        heart_disease,
        glucose,
        bmi,
        result,
        final_score,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    # RETURN OUTPUT
    return render_template(
        "index.html",
        prediction_text=result,
        percentage=final_score
    )

# HISTORY PAGE

@app.route("/history")
def history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)

# RUN APP

if __name__ == "__main__":
    app.run(debug=True)