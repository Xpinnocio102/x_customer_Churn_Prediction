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
contract_map = {"Month-to-month": [0, 0], "One year": [1, 0], "Two year": [0, 1]}
payment_map = {"Electronic check": [0, 1, 0], "Mailed check": [0, 0, 1], "Credit card (automatic)": [1, 0, 0]}
internet_map = {"DSL": [0, 0], "Fiber optic": [1, 0], "No": [0, 1]}

# Define expected model features (must match training data exactly)
expected_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
    "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
]

# Create input feature array
input_values = [
    gender_map[gender], 0,  # SeniorCitizen (default 0)
    binary_map[partner], binary_map[dependents], tenure,
    binary_map[phone_service], binary_map[paperless_billing],
    monthly_charges, total_charges, binary_map[multiple_lines],
    *internet_map[internet_service],  # Internet service encoding
    binary_map[online_security], binary_map[online_backup],
    binary_map[device_protection], binary_map[tech_support],
    binary_map[streaming_tv], binary_map[streaming_movies],
    *contract_map[contract],  # Contract type encoding
    *payment_map[payment_method]  # Payment method encoding
]

# Convert input into DataFrame with correct feature names
input_data_df = pd.DataFrame([input_values], columns=expected_features)

# Ensure correct feature order matches the trained model
for col in model.feature_names_in_:
    if col not in input_data_df.columns:
        input_data_df[col] = 0  # Add missing columns

input_data_df = input_data_df[model.feature_names_in_]

# Convert to NumPy array
input_data = input_data_df.to_numpy()

# Debugging: Show expected vs. provided features
st.write("🔹 Model Expected Features:", model.feature_names_in_)
st.write("🔹 Input Data Columns:", input_data_df.columns.tolist())
st.write("🔹 Input Data Values:", input_data_df.values)

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
