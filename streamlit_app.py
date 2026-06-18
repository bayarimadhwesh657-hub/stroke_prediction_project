import streamlit as st
import numpy as np
import joblib
import pickle

st.set_page_config(
    page_title="Stroke Risk Detection System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Stroke Risk Detection System")
st.write("Enter patient details to predict stroke risk.")

@st.cache_resource
def load_models():
    model = joblib.load("stroke_model_compressed.joblib")

    with open("imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, imputer, scaler


model, imputer, scaler = load_models()

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=120.0,
    value=45.0
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

glucose = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

if st.button("Predict Stroke Risk"):
    try:
        features = np.array([
            age,
            hypertension,
            heart_disease,
            glucose,
            bmi
        ]).reshape(1, -1)

        features = imputer.transform(features)
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

        st.info(
            f"Stroke Risk Probability: {probability:.2%}"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")