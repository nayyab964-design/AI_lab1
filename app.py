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
    # File name must match your repo exactly
    with open("best_churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("Model loaded successfully!")

# ---------------- LAYOUT ----------------
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
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

# ---------------- PREDICTION LOGIC ----------------
if st.button("Predict Churn", type="primary"):

    # 1. Replicating your Week 3 Binary Encoding
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges
    }

    input_df = pd.DataFrame([input_data])

    # 2. FEATURE ALIGNMENT (Fixes the ValueError)
    # This list must include EVERY numeric column from your Week 3 X_train
    expected_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'Partner', 'Dependents'] 
    
    # Ensure all expected columns exist (even those not in the form)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match the training set exactly
    input_df = input_df[expected_features]

    # 3. PREDICTION
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob = probability[1] * 100

    # ---------------- RESULTS ----------------
    if prediction == 1:
        st.error("⚠️ HIGH RISK: Customer likely to churn")
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
    else:
        st.success("✅ LOW RISK: Customer likely to stay")
        st.metric("Retention Probability", f"{100 - churn_prob:.1f}%")

    # ---------------- VISUALIZATION ----------------
    # Gauge chart as required by deliverables
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob,
        title={"text": "Churn Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if churn_prob > 50 else "green"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "salmon"}
            ]
        }
    ))
    st.plotly_chart(fig)

    # Business Recommendation
    if churn_prob > 70:
        st.
