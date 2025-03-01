import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("Xpinnocio_churn_prediction_model.pkl")

# Set page title
st.title("📊 Customer Churn Prediction App")

# Sidebar for user inputs
st.sidebar.header("📌 Customer Details")

# User Inputs
tenure = st.sidebar.slider("📅 Customer Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("💵 Monthly Charges", min_value=10, max_value=150, value=50)
total_charges = tenure * monthly_charges  # Auto-calculate

contract_type = st.sidebar.selectbox("📜 Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Credit card (automatic)"])

# Convert categorical inputs into numerical format (matching model training)
contract_map = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
payment_map = {"Electronic check": [1, 0, 0], "Mailed check": [0, 1, 0], "Credit card (automatic)": [0, 0, 1]}

# ✅ Ensure input values match the training format
columns = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check", "PaymentMethod_Credit card (automatic)"
]

# Create DataFrame with proper structure
input_values = [
    tenure, monthly_charges, total_charges,
    *contract_map[contract_type], *payment_map[payment_method]
]

# Debugging: Check if input data matches expected feature count
st.write(f"Expected features: {len(columns)}, Provided features: {len(input_values)}")

# Convert input into DataFrame
input_data_df = pd.DataFrame([input_values], columns=columns)

# Debugging: Print expected vs. provided features
st.write("🔹 Model Expected Features:", model.feature_names_in_)
st.write("🔹 Input Data Columns:", input_data_df.columns.tolist())

# Ensure feature order matches training set
input_data_df = input_data_df[model.feature_names_in_]

# Convert to NumPy array
input_data = input_data_df.to_numpy()

# Debugging: Check final input shape
st.write("Final Input Data Shape:", input_data.shape)

# Prediction Button
if st.sidebar.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    churn_prob = model.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 **High Risk:** Customer is likely to churn. (Probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ **Low Risk:** Customer is unlikely to churn. (Probability: {churn_prob:.2f})")

    # Show probability bar
    st.progress(churn_prob)
