import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🚨 Insurance Claim Fraud Detection (Logistic vs Random Forest)")

# --------------------------
# Load trained pipelines
# --------------------------
@st.cache_resource
def load_models():
    log_pipe = joblib.load(Path("model/logistic_pipeline.pkl"))
    rf_pipe  = joblib.load(Path("model/rf_pipeline.pkl"))
    return log_pipe, rf_pipe

try:
    log_pipeline, rf_pipeline = load_models()
except Exception as e:
    st.error("Models not found. Please train models first by running `python src/train.py`.")
    st.stop()

st.markdown("Upload a CSV containing the **same feature columns** used in training.")

uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    # If target exists in uploaded CSV, drop it for prediction
    if "fraud_reported" in df.columns:
        st.info("Detected 'fraud_reported' in uploaded CSV. It will be ignored for prediction.")
        X_pred = df.drop(columns=["fraud_reported"])
    else:
        X_pred = df.copy()

    # Predict with both models
    with st.spinner("Running predictions..."):
        log_pred = log_pipeline.predict(X_pred)
        log_prob = log_pipeline.predict_proba(X_pred)[:, 1]

        rf_pred  = rf_pipeline.predict(X_pred)
        rf_prob  = rf_pipeline.predict_proba(X_pred)[:, 1]

    # Assemble results
    results = df.copy()
    results["Logistic_Pred"] = (log_pred == 1).map({True: "Fraud", False: "Legitimate"})
    results["Logistic_Prob"] = log_prob
    results["RF_Pred"] = (rf_pred == 1).map({True: "Fraud", False: "Legitimate"})
    results["RF_Prob"] = rf_prob

    st.subheader("Predictions")
    st.dataframe(results.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

    # Optional: simple threshold tuner for logistic
    st.markdown("---")
    st.subheader("Threshold Inspection (Logistic)")
    thr = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    thr_pred = (log_prob >= thr).astype(int)
    st.write(f"At threshold **{thr:.2f}**, predicted fraud count: **{thr_pred.sum()}** / {len(thr_pred)}")

else:
    st.info("Upload a CSV to get predictions.")

with st.sidebar:
    st.header("About")
    st.write("""
This app loads two **trained pipelines**:
- **Logistic Regression** (with scaling)  
- **Random Forest** (tree-based, no scaling)

Preprocessing is embedded in the pipelines:
- Date feature engineering: `days_since_policy`, `policy_bind_month`, `incident_month`, `incident_weekday`
- Drops identifiers: `policy_number`, `insured_zip`, `incident_location`
- One-Hot encodes categoricals
- Scales numeric features (Logistic only)
- Class imbalance handled via `class_weight='balanced'`
""")
