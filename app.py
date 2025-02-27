import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("churn_prediction_model.pkl")

# Title of the app
st.title("Customer Churn Prediction App")

# User Inputs
tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=10, max_value=150, value=50)
total_charges = tenure * monthly_charges  # Auto-calculate
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Credit card (automatic)"])

# Convert categorical inputs into numerical format (match model training)
contract_map = {"Month-to-month": [1, 0], "One year": [0, 1], "Two year": [0, 0]}
payment_map = {"Electronic check": [1, 0, 0], "Mailed check": [0, 1, 0], "Credit card (automatic)": [0, 0, 1]}

# Create the input feature array
input_data = np.array([
    tenure, monthly_charges, total_charges,
    *contract_map[contract_type], *payment_map[payment_method]
]).reshape(1, -1)

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    churn_prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 High Risk: Customer is likely to churn. (Probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ Low Risk: Customer is unlikely to churn. (Probability: {churn_prob:.2f})")
