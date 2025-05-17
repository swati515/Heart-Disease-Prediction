import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("🫀 Heart Disease Risk Prediction")
st.write("Provide your symptoms and risk factors to check the chances of heart disease.")

def user_input():
    Chest_Pain = st.selectbox("Chest Pain", ["Yes", "No"])
    Chest_Pain = 1 if Chest_Pain == "Yes" else 0
    Shortness_of_Breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    Shortness_of_Breath = 1 if Shortness_of_Breath == "Yes" else 0
    Fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    Fatigue = 1 if Fatigue == "Yes" else 0
    Palpitations = st.selectbox("Palpitations", ["Yes", "No"])
    Palpitations = 1 if Palpitations == "Yes" else 0
    Dizziness = st.selectbox("Dizziness", ["Yes", "No"])
    Dizziness = 1 if Dizziness == "Yes" else 0
    Swelling = st.selectbox("Swelling", ["Yes", "No"])
    Swelling = 1 if Swelling == "Yes" else 0
    Pain_Arms_Jaw_Back = st.selectbox("Pain in Arms/Jaw/Back", ["Yes", "No"])
    Pain_Arms_Jaw_Back = 1 if Pain_Arms_Jaw_Back == "Yes" else 0
    Cold_Sweats_Nausea = st.selectbox("Cold Sweats/Nausea", ["Yes", "No"])
    Cold_Sweats_Nausea = 1 if Cold_Sweats_Nausea == "Yes" else 0
    High_BP = st.selectbox("High Blood Pressure", ["Yes", "No"])
    High_BP = 1 if High_BP == "Yes" else 0
    High_Cholesterol = st.selectbox("High Cholesterol", ["Yes", "No"])
    High_Cholesterol = 1 if High_Cholesterol == "Yes" else 0
    Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    Diabetes = 1 if Diabetes == "Yes" else 0
    Smoking = st.selectbox("Smoking", ["Yes", "No"])
    Smoking = 1 if Smoking == "Yes" else 0
    Obesity = st.selectbox("Obesity", ["Yes", "No"])
    Obesity = 1 if Obesity == "Yes" else 0
    Sedentary_Lifestyle = st.selectbox("Sedentary Lifestyle", ["Yes", "No"])
    Sedentary_Lifestyle = 1 if Sedentary_Lifestyle == "Yes" else 0
    Family_History = st.selectbox("Family History", ["Yes", "No"])
    Family_History = 1 if Family_History == "Yes" else 0
    Chronic_Stress = st.selectbox("Chronic Stress", ["Yes", "No"])
    Chronic_Stress = 1 if Chronic_Stress == "Yes" else 0
    Gender = st.radio("Gender", ["Male", "Female"])
    Gender = 1 if Gender == "Male" else 0

    data = {
        'Chest_Pain': Chest_Pain,
        'Shortness_of_Breath': Shortness_of_Breath,
        'Fatigue': Fatigue,
        'Palpitations': Palpitations,
        'Dizziness': Dizziness,
        'Swelling': Swelling,
        'Pain_Arms_Jaw_Back': Pain_Arms_Jaw_Back,
        'Cold_Sweats_Nausea': Cold_Sweats_Nausea,
        'High_BP': High_BP,
        'High_Cholesterol': High_Cholesterol,
        'Diabetes': Diabetes,
        'Smoking': Smoking,
        'Obesity': Obesity,
        'Sedentary_Lifestyle': Sedentary_Lifestyle,
        'Family_History': Family_History,
        'Chronic_Stress': Chronic_Stress,
        'Gender': Gender
    }

    return pd.DataFrame([data])

# Get input
input_df = user_input()

# Prediction button
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"🔴 High Risk of Heart Disease ({probability*100:.2f}% probability)")
    else:
        st.success(f"🟢 Low Risk of Heart Disease ({probability*100:.2f}% probability)")
