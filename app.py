import streamlit as st
import pickle
import numpy as np

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")
st.write("Enter the patient's details below to get a prediction:")

glucose = st.number_input("Glucose", value=120, min_value=0)
blood_pressure = st.number_input("Blood Pressure", value=70, min_value=0)
bmi = st.number_input("BMI", value=30.0, min_value=0.0)
insulin = st.number_input("Insulin", value=80, min_value=0)
age = st.number_input("Age", value=35, min_value=0)

if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, insulin, bmi, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("Prediction: The patient is likely diabetic.")
    else:
        st.success("Prediction: The patient is not diabetic.")