import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("Diabetes Prediction System")

st.write("Enter the patient details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            diabetes_pedigree,
                            age]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Not Diabetic")