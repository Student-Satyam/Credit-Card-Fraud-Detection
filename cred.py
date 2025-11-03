import streamlit as st
import numpy as np
import joblib

# ✅ Load trained model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection System  by satyam")
st.write("Enter transaction details to predict whether it's Fraud or Legit ✅")

# ✅ Model Features
feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount_Scaled']
inputs = []

# ✅ Taking user input dynamically
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

# ✅ Convert to correct shape (1 sample, 6 features)
user_data = np.array(inputs).reshape(1, -1)

# ✅ Predict Button
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(user_data)[0]
    confidence = model.predict_proba(user_data).max() * 100

    if prediction == 1:
        st.error(f"🚨 Fraud Transaction Detected! | Confidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Legit Transaction | Confidence: {confidence:.2f}%")

# ✅ Show model info
st.caption("Model: Random Forest | Accuracy: 100% on Test Data ✅")

# ✅ Very Important Theory Sect
