import streamlit as st
import requests
import pandas as pd

API_URL = 'https://new-repo-42yn.onrender.com/predict'

st.title('Employee Termination')
st.subheader('Using Logistic Regression')


emp_satisfaction = st.slider(
    'Employee Satisfaction',
    max_value=5,
    min_value=1,
    value=3,
    step=1
)

if st.button('Predict Termination or Active'):
    payload = {
        "EmpSatisfaction": emp_satisfaction
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json

            if result["Predicted Termination"] == 1:
                st.error("Employee is likely to be terminated.")
            else:
                st.success("Employee is likely to remain active..")
        else:
            st.warning("API returned an error..")
    except requests.exceptions.RequestException:
        st.error("Could not connect to API.")