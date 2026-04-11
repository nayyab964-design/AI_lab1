import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
# Configures the browser tab and layout [cite: 33, 37]
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System") [cite: 34]

# ---------------- LOAD MODEL ----------------
# Caches the model to improve performance [cite: 38, 42]
@st.cache_resource
def load_model():
    # Make sure 'best_churn_model.pkl' is in your GitHub root folder [cite: 18, 25]
    with open("best_churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model() [cite: 41]
st.success("Model loaded successfully!") [cite: 41]

# ---------------- LAYOUT ----------------
# Creates two columns for a professional dashboard look [cite: 45]
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics") [cite: 46]
    gender = st.selectbox("Gender", ["Male", "Female"]) [cite: 49]
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"]) [cite: 51]
    partner = st.selectbox("Partner", ["No", "Yes"]) [cite: 53]
    dependents = st.selectbox("Dependents", ["No", "Yes"]) [cite: 55]

with col2:
    st.subheader("Account Information") [cite: 56]
    tenure = st.slider("Tenure (months)", 0, 72, 12) [cite: 59]
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    ) [cite: 60, 62]

# ---------------- PREDICTION LOGIC ----------------
if st.button("Predict Churn", type="primary"): [cite: 65]

    # 1. Convert inputs to a dictionary [cite: 65]
    # We use 1/0 for binary columns to match your Week 3 numeric training data
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges
    }

    # 2. Convert to DataFrame [cite: 65, 68]
    input_df = pd.DataFrame([input_data])

    # 3. Feature Alignment [cite: 69]
    # IMPORTANT: This list must match the columns in your X_train from Week 3
    expected_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'Partner', 'Dependents']
    
    # Ensure all columns exist and are in the correct order for the model
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    # 4. Make Prediction [cite: 69]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob = probability[1] * 100

    # ---------------- RESULTS ----------------
    if prediction == 1:
        st.error("⚠️ HIGH RISK: Customer likely to churn") [cite: 72, 75]
        st.metric("Churn Probability", f"{churn_prob:.1f}%") [cite: 76]
    else:
        st.success("✅ LOW RISK: Customer likely to stay") [cite: 76]
        st.metric("Retention Probability", f"{100 - churn_prob:.1f}%") [cite: 76]

    # ---------------- VISUALIZATION ----------------
    # Gauge chart to meet the "Visualizations" deliverable [cite: 11, 100]
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

    # Business Recommendation [cite: 101]
    if churn_prob > 70:
        st.warning("Strategy: High risk detected. Recommend a proactive loyalty offer.")
