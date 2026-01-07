import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/termination_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Employee Termination Prediction")
st.subheader("Predict Employee Termination based on Satisfaction Level")

emp_satisfaction = st.slider(
    "Employee Satisfaction",
    min_value=1,
    value=3,
    max_value=5,
    step=1
)

if st.button("Predict Termination"):
    input_df = pd.DataFrame({
        "EmpSatisfaction": [emp_satisfaction]
    })

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error("Employee is likely to be Terminated.")
    else:
        st.success("Employee is likely to remain Active.")