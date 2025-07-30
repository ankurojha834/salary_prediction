import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("Employee Income Prediction")

# User input
age = st.slider("Age", 17, 75, 30)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.number_input("FNLWGT", value=100000)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
race = st.selectbox("Race", encoders['race'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.slider("Hours per week", 1, 99, 40)
native_country = st.selectbox("Native Country", encoders['native-country'].classes_)
educational_num = st.slider("Educational Number", 5, 16, 10)

# Encode inputs
input_data = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0],
    'educational-num': educational_num
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Income: {prediction}")
