import pandas as pd
import joblib
import numpy as np
import streamlit as st

# App title
st.title("Diabetes Prediction App")
st.write("Enter the patient details in the sidebar to check the diabetes risk:")

# Data collection
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
X = pd.DataFrame(np.reshape(data, (1,8)), columns = features, index = range(1))

# scalling the input
with open("scaler.pkl", 'rb') as sc:
    scaler = joblib.load(sc)
X_scaled = scaler.transform(X)

# predicting the output
with open("Diabetes pedictor model.pkl", 'rb') as mdl:
    model = joblib.load(mdl)

if st.button("Predict"):
    Y = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)[0][1]
    if (Y[0] == 1):
        st.warning(f"⚠️ Diabetes Detected (Risk: {prob:.2%})")
    else:
       st.success(f"✅ No Diabetes (Risk: {prob:.2%})")

# Developer
st.markdown("---")
st.markdown("Developed by [Saifullah Umar](https://github.com/SaifUllahUmar0317)")