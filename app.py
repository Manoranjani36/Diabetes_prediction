import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = "models/diabetes_model.pkl"

st.title("🩺 Diabetes Prediction")
st.write("Enter the patient's details to predict the diabetes outcome.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Inputs based on the standard diabetes prediction dataset
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=40.0)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "No Info", "current", "former", "ever", "not current"]
)
bmi = st.number_input("BMI", min_value=5.0, max_value=100.0, value=27.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5)
blood_glucose = st.number_input(
    "Blood Glucose Level", min_value=50.0, max_value=400.0, value=120.0
)

input_df = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": blood_glucose
}])

if st.button("🔮 Predict Diabetes", use_container_width=True):
    prediction = model.predict(input_df)[0]

    if int(prediction) == 1:
        st.error("Prediction: Diabetes")
        st.warning("This is a machine-learning prediction and is not a medical diagnosis.")
    else:
        st.success("Prediction: No Diabetes")
        st.info("This is a machine-learning prediction and is not a medical diagnosis.")

st.markdown("---")
st.caption("Diabetes Prediction • Machine Learning Classification Project")
