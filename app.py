import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("Xpinnocio_churn_prediction_model.pkl")

# Set page title and layout
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# App Title and Description
st.title("📊 Customer Churn Prediction App")
st.markdown("""
🚀 Predict whether a customer is likely to **churn** (leave the service) based on their account details.

🔹 **How to use?**
- Adjust the **customer details** in the **sidebar**.
- Click **"Predict Churn"** to see results.
""")

# Sidebar for user inputs
st.sidebar.header("📌 Customer Details")

# Collect user inputs
gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("💍 Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("👶 Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("📞 Phone Service", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("📄 Paperless Billing", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("📶 Multiple Lines", ["Yes", "No"])

# Numerical inputs
tenure = st.sidebar.slider("📅 Customer Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("💵 Monthly Charges", min_value=10, max_value=150, value=50)
total_charges = tenure * monthly_charges  # Auto-calculate

# Encode categorical variables
gender_map = {"Male": 0, "Female": 1}
binary_map = {"No": 0, "Yes": 1}

# Define all features expected by the model
expected_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_Yes"
]

# Create input feature array
input_values = [
    gender_map[gender], 0,  # SeniorCitizen (default 0)
    binary_map[partner], binary_map[dependents], tenure,
    binary_map[phone_service], binary_map[paperless_billing],
    monthly_charges, total_charges, binary_map[multiple_lines]
]

# Convert input into DataFrame with correct feature names
input_data_df = pd.DataFrame([input_values], columns=expected_features)

# Ensure correct feature order matches the trained model
for col in model.feature_names_in_:
    if col not in input_data_df.columns:
        input_data_df[col] = 0  # Add missing columns with default values

# Align feature order
input_data_df = input_data_df[model.feature_names_in_]

# Convert to NumPy array
input_data = input_data_df.to_numpy()

# Debugging: Show expected vs. provided features
st.write("🔹 Model Expected Features:", model.feature_names_in_)
st.write("🔹 Input Data Columns:", input_data_df.columns.tolist())

# Prediction Button
if st.sidebar.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    churn_prob = float(model.predict_proba(input_data)[0][1])  # Ensure it's a float

    # Display results
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 **High Risk:** Customer is likely to churn. (Probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ **Low Risk:** Customer is unlikely to churn. (Probability: {churn_prob:.2f})")

    # Fix progress bar error by ensuring a valid range
    st.progress(min(max(churn_prob, 0.01), 1.0))

# Footer
st.markdown("""
---
🎯 Developed by **Your Name** | 🚀 Powered by **Machine Learning**
""")
