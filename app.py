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

# Numerical inputs (scaled to match training)
tenure = (st.sidebar.slider("📅 Customer Tenure (Months)", 0, 72, 12) - 0) / (72 - 0)  # Scale to 0-1
monthly_charges = (st.sidebar.number_input("💵 Monthly Charges", min_value=10, max_value=150, value=50) - 10) / (150 - 10)  # Scale to 0-1
total_charges = (tenure * monthly_charges - 0) / ((72 * 150) - 0)  # Scale to 0-1

# Additional categorical features
contract = st.sidebar.selectbox("📜 Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Credit card (automatic)"])
internet_service = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("🔒 Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("📂 Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("🛡️ Device Protection", ["Yes", "No"])
tech_support = st.sidebar.selectbox("🛠 Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("📺 Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("🎥 Streaming Movies", ["Yes", "No"])

# Encode categorical variables
gender_map = {"Male": 0, "Female": 1}
binary_map = {"No": 0, "Yes": 1}
contract_map = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
payment_map = {"Electronic check": [1, 0, 0], "Mailed check": [0, 1, 0], "Credit card (automatic)": [0, 0, 1]}
internet_map = {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No": [0, 0, 1]}

# Define expected model features (must match training data exactly)
expected_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
    "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

# Create input feature array (Flatten categorical lists)
input_values = [
    gender_map[gender], 0,  # SeniorCitizen (default 0)
    binary_map[partner], binary_map[dependents], tenure,
    binary_map[phone_service], binary_map[paperless_billing],
    monthly_charges, total_charges, binary_map[multiple_lines]
]
input_values += internet_map[internet_service]
input_values += [binary_map[online_security], binary_map[online_backup],
                 binary_map[device_protection], binary_map[tech_support],
                 binary_map[streaming_tv], binary_map[streaming_movies]]
input_values += contract_map[contract]
input_values += payment_map[payment_method]

# Debugging: Ensure input length matches expected features
st.write(f"✅ Expected Features: {len(expected_features)} | Provided Features: {len(input_values)}")

# Convert input into DataFrame
input_data_df = pd.DataFrame([input_values], columns=expected_features)

# Make prediction
prediction = model.predict(input_data_df)[0]
churn_prob = float(model.predict_proba(input_data_df)[0][1])

# Display results
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.error(f"🚨 **High Risk:** Customer is likely to churn. (Probability: {churn_prob:.2f})")
else:
    st.success(f"✅ **Low Risk:** Customer is unlikely to churn. (Probability: {churn_prob:.2f})")

# Fix progress bar error
st.progress(min(max(churn_prob, 0.01), 1.0))

# Footer
st.markdown("""
---
🎯 Developed by **Xpinnocio2** | 🚀 Powered by **x102_ML**
""")
