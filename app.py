"""
FinShield AI — Real-Time Fraud Detection App
Deployed via Streamlit Community Cloud
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="FinShield AI | Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    base         = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(base, "models", "xgb_fraud_model.pkl")
    scaler_path  = os.path.join(base, "models", "scaler.pkl")
    feature_path = os.path.join(base, "models", "feature_columns.json")

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    with open(feature_path) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ FinShield AI — Fraud Detection System")
st.markdown("**NovaPay International · Global Transaction Intelligence Platform**")
st.markdown(
    "Enter transaction details below and click **Analyse Transaction** "
    "to get an instant fraud risk assessment."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Transaction Details")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Transaction Info**")
    amount_usd        = st.number_input("Transaction Amount (USD)", 1.0, 100000.0, 250.0, step=50.0)
    channel           = st.selectbox("Channel",
                            ["mobile_app","web","POS","ATM","USSD","bank_transfer"])
    merchant_category = st.selectbox("Merchant Category", [
                            "e-commerce","grocery","utilities","travel","entertainment",
                            "restaurants","healthcare","financial_services",
                            "crypto_exchange","gaming","remittance","fuel",
                            "education","subscription"])
    payment_method    = st.selectbox("Payment Method",
                            ["debit_card","credit_card","wallet",
                             "bank_transfer","crypto","BNPL"])

with c2:
    st.markdown("**Account Info**")
    device_type        = st.selectbox("Device Type",
                             ["Android","iOS","Windows","macOS","Linux","Unknown"])
    kyc_tier           = st.selectbox("KYC Tier",
                             ["unverified","basic","enhanced","premium"])
    region             = st.selectbox("Region", [
                             "North America","Europe","Asia","Africa",
                             "South America","Middle East","Oceania"])
    account_age_months = st.slider("Account Age (months)", 1, 120, 12)

with c3:
    st.markdown("**Behavioural History**")
    tx_count_30d       = st.slider("Transactions last 30 days", 0, 100, 15)
    avg_tx_amount_30d  = st.number_input("Avg Transaction Amount 30d (USD)",
                             1.0, 50000.0, 200.0)
    failed_tx_7d       = st.slider("Failed Transactions (last 7 days)", 0, 20, 0)
    login_attempts_24h = st.slider("Login Attempts (last 24h)", 0, 20, 1)

st.divider()
c4, c5 = st.columns(2)

with c4:
    st.markdown("**Time & Location**")
    distinct_countries_30d = st.slider("Distinct Countries Transacted (30 days)", 1, 10, 1)
    hour        = st.slider("Transaction Hour (0-23)", 0, 23, 14)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    month       = st.slider("Month", 1, 12, 6)

with c5:
    st.markdown("**Risk Flags**")
    is_new_device       = st.checkbox("New / Unrecognised Device")
    is_night_tx         = st.checkbox("Night Transaction (00:00-05:59)")
    is_weekend          = st.checkbox("Weekend Transaction")
    ip_country_mismatch = st.checkbox("IP Address / Country Mismatch")
    velocity_flag       = st.checkbox("High Velocity Flag (>40 txns / 30 days)")

# ── Feature builder ───────────────────────────────────────────────────────────
def build_input_row():
    kyc_map   = {"unverified": 0, "basic": 1, "enhanced": 2, "premium": 3}
    log_ratio = np.log1p(amount_usd / (avg_tx_amount_30d + 1e-6))

    risk_score = (
        int(ip_country_mismatch)         * 2.0 +
        int(is_new_device)               * 1.5 +
        int(is_night_tx)                 * 1.0 +
        int(velocity_flag)               * 1.5 +
        int(failed_tx_7d >= 3)           * 2.0 +
        int(login_attempts_24h >= 5)     * 1.5 +
        int(distinct_countries_30d >= 4) * 2.0 +
        int(account_age_months < 3)      * 1.0
    )

    base = {
        "account_age_months":     account_age_months,
        "tx_count_30d":           tx_count_30d,
        "failed_tx_7d":           failed_tx_7d,
        "login_attempts_24h":     login_attempts_24h,
        "distinct_countries_30d": distinct_countries_30d,
        "is_new_device":          int(is_new_device),
        "is_night_tx":            int(is_night_tx),
        "is_weekend":             int(is_weekend),
        "velocity_flag":          int(velocity_flag),
        "ip_country_mismatch":    int(ip_country_mismatch),
        "hour":                   hour,
        "day_of_week":            day_of_week,
        "month":                  month,
        "quarter":                (month - 1) // 3 + 1,
        "log_amount":             np.log1p(amount_usd),
        "log_avg_tx_amt":         np.log1p(avg_tx_amount_30d),
        "log_amount_ratio":       log_ratio,
        "risk_score":             risk_score,
        "kyc_encoded":            kyc_map[kyc_tier],
    }

    row = pd.DataFrame([base])

    cat_inputs = {
        "channel": channel, "merchant_category": merchant_category,
        "payment_method": payment_method, "device_type": device_type,
        "region": region,
    }
    for cat, val in cat_inputs.items():
        col_name = f"{cat}_{val}"
        for fc in feature_cols:
            if fc.startswith(cat + "_") and fc not in row.columns:
                row[fc] = 0
        if col_name in feature_cols:
            row[col_name] = 1

    for fc in feature_cols:
        if fc not in row.columns:
            row[fc] = 0

    return row[feature_cols]

# ── Prediction button ─────────────────────────────────────────────────────────
if st.button("🔍  Analyse Transaction", type="primary", use_container_width=True):
    row  = build_input_row()
    prob = model.predict_proba(row)[0][1]
    pred = int(prob >= 0.50)

    st.divider()
    st.subheader("🔎 Fraud Risk Assessment")
    res_col, gauge_col = st.columns([1, 2])

    with res_col:
        if pred == 1:
            st.error("🚨 HIGH RISK — FRAUD LIKELY")
        else:
            st.success("✅ LOW RISK — TRANSACTION CLEAN")
        st.metric("Fraud Probability", f"{prob:.1%}")
        st.metric("Decision Threshold", "50%")
        st.metric("Model", "XGBoost")

    with gauge_col:
        fig, ax = plt.subplots(figsize=(6, 1.8))
        bar_color = "#E05252" if pred == 1 else "#2ECC71"
        ax.barh(["Risk Level"], [prob],          color=bar_color,  height=0.45)
        ax.barh(["Risk Level"], [1 - prob], left=prob,
                color="#EEEEEE", height=0.45)
        ax.axvline(0.5, color="orange", lw=2.5,
                   linestyle="--", label="Threshold (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraud Probability")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Fraud Score: {prob:.1%}", fontweight="bold")
        st.pyplot(fig)

    st.divider()
    st.markdown("**What the score means:**")
    if prob < 0.30:
        st.info("🟢 Very low risk. Transaction profile is consistent with normal behaviour.")
    elif prob < 0.50:
        st.warning("🟡 Moderate risk. Some unusual signals present but below fraud threshold.")
    elif prob < 0.75:
        st.error("🔴 High risk. Multiple fraud indicators detected. Review recommended.")
    else:
        st.error("🚨 Very high risk. Strong fraud signals across multiple dimensions. Block or escalate.")

    st.caption("Model: XGBoost · FinShield Synthetic v1.0 · Threshold: 0.50")
