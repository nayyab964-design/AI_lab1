import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("best_churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("Model loaded successfully!")

# ---------------- LAYOUT (Inputs) ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=70.0)

# ---------------- PREDICTION LOGIC ----------------
if st.button("Predict Churn", type="primary"):

    # 1. Map Inputs directly to the Names expected by the model
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        # Default binary features based on your logs
        "PhoneService_Yes": 1, 
        "PaperlessBilling_Yes": 1
    }

    input_df = pd.DataFrame([input_data])

    # 2. FULL FEATURE ALIGNMENT (Based on your Error Log)
    # Ye wo exact list hai jo model expect kar raha hai
    expected_features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
        'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
        'MultipleLines_No phone service', 'MultipleLines_Yes', 
        'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No internet service', 'TechSupport_Yes', 
        'StreamingTV_No internet service', 'StreamingTV_Yes', 
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check'
    ]

    # Jo columns input form mein nahi hain, unhein 0 set karein
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Order alignment
    input_df = input_df[expected_features]

    # 3. Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob = probability[1] * 100

    # ---------------- RESULTS ----------------
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Customer likely to churn ({churn_prob:.1f}%)")
    else:
        st.success(f"✅ LOW RISK: Customer likely to stay ({100 - churn_prob:.1f}% Retention)")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob,
        title={"text": "Churn Risk (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if churn_prob > 50 else "green"}}
    ))
    st.plotly_chart(fig)
