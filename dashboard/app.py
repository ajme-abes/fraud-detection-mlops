import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")

st.title("🚨 Fraud Detection Monitoring Dashboard")

# =============================
# 🔹 Fetch Data
# =============================
try:
    logs_response = requests.get(f"{API_URL}/logs/raw").json()
    acc_response = requests.get(f"{API_URL}/monitoring/accuracy").json()
    
    # Ensure logs_data is always a LIST
    if isinstance(logs_response, list):
        logs_data = logs_response
    elif isinstance(logs_response, dict):
        # If it's a single dict, wrap it in a list
        logs_data = [logs_response] 
    else:
        logs_data = []
        
except Exception as e:
    st.error(f"API Connection Error: {e}")
    st.stop()

# =============================
# 🔹 Create DataFrame
# =============================
if not logs_data or logs_data == [{}]:
    df = pd.DataFrame()
else:
    # This now handles [dict, dict] or [dict]
    df = pd.DataFrame(logs_data) 

# Check if 'prediction' column exists before doing math
if not df.empty and "prediction" in df.columns:
    fraud_rate = df["prediction"].mean()
else:
    fraud_rate = 0

# =============================
# 🔹 Metrics
# =============================
col1, col2, col3 = st.columns(3)

total_preds = len(df)

# Check if the 'prediction' column actually exists in the data
if not df.empty and "prediction" in df.columns:
    # Ensure values are numeric before calculating mean
    fraud_rate = pd.to_numeric(df["prediction"]).mean()
else:
    fraud_rate = 0.0

accuracy = acc_response.get("accuracy", "N/A")

col1.metric("Total Predictions", total_preds)
col2.metric("Fraud Rate", f"{fraud_rate:.2f}")
col3.metric("Accuracy", accuracy)

# =============================
# 🔹 Table
# =============================
st.subheader("Recent Predictions")

if not df.empty:
    st.dataframe(df.tail(20))
else:
    st.write("No data yet")

# =============================
# 🔹 Drift Report
# =============================
st.subheader("Drift Monitoring")

if st.button("Run Drift Check"):
    res = requests.post(f"{API_URL}/monitoring/run-drift").json()
    st.write(res)

st.markdown(f"👉 [View Full Drift Report]({API_URL}/reports/drift_report.html)")

if st.button("🚀 Trigger Retraining"):
    res = requests.post(f"{API_URL}/monitoring/retrain")
    if res.status_code == 200:
        st.success(res.json().get("message"))
    else:
        st.error(f"Error: {res.text}")