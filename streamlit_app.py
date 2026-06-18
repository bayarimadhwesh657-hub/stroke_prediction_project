import streamlit as st
import numpy as np
import joblib
import pickle

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Stroke Risk Detection System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Stroke Risk Detection System")
st.write("Enter patient details to predict stroke risk.")

# -----------------------------------
# LOAD MODEL FILES
# -----------------------------------
@st.cache_resource
def load_models():

    model = joblib.load("stroke_model_compressed.joblib")

    with open("imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, imputer, scaler


model, imputer, scaler = load_models()

# -----------------------------------
# USER INPUTS
# -----------------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1],
    help="0 = No, 1 = Yes"
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1],
    help="0 = No, 1 = Yes"
)

glucose = st.number_input(
    "Average Glucose Level",
    min_value=1.0,
    max_value=500.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=1.0,
    max_value=100.0,
    value=25.0
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Predict Stroke Risk"):

    try:

        # Additional validation
        if age <= 0:
            st.error("❌ Age must be greater than 0")
            st.stop()

        if glucose <= 0:
            st.error("❌ Glucose level must be greater than 0")
            st.stop()

        if bmi <= 0:
            st.error("❌ BMI must be greater than 0")
            st.stop()

        # Create feature array
        features = np.array([
            age,
            hypertension,
            heart_disease,
            glucose,
            bmi
        ]).reshape(1, -1)

        # Preprocessing
        features = imputer.transform(features)
        features = scaler.transform(features)

        # Prediction
        prediction = model.predict(features)[0]

        # Probability
        probability = model.predict_proba(features)[0][1]

        st.divider()

        # Stroke Prediction Result
        if prediction == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

        st.info(
            f"Stroke Risk Probability: {probability:.2%}"
        )

        # -----------------------------------
        # HEALTH STATUS CLASSIFICATION
        # -----------------------------------

        if (
            glucose < 100
            and bmi < 25
            and hypertension == 0
            and heart_disease == 0
        ):
            st.success(
                "💚 Health Status: HEALTHY"
            )

        elif (
            100 <= glucose < 170
            and 25 <= bmi < 30
            and hypertension == 0
            and heart_disease == 0
        ):
            st.warning(
                "🟡 Health Status: MODERATE"
            )

        else:
            st.error(
                "🔴 Health Status: UNHEALTHY"
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")