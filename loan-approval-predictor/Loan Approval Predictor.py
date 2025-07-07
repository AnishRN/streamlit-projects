import streamlit as st
import joblib
import numpy as np

RanFor = joblib.load("RanForModel.pkl")
XB = joblib.load("XBModel.pkl")

st.title("🏦 Loan Approval Prediction App")
st.markdown("🔍 Fill out the form below to check **loan approval prediction** using two models: Random Forest (RanFor) and XGBoost (XB).")

with st.form("loan_form"):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    income_annum = st.number_input("Annual Income", min_value=0, step=10000)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=10000)
    loan_term = st.number_input("Loan Term (in months)", min_value=0, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)

    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=10000)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=10000)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=10000)

    submit = st.form_submit_button("Predict")

if submit:
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    input_data = np.array([[
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    rf_pred = RanFor.predict(input_data)[0]
    xgb_pred = XB.predict(input_data)[0]

    pred_map = {1: "✅ Approved", 0: "❌ Rejected"}

    st.subheader("📊 Model Predictions:")
    st.info(f"**Random Forest (RanFor):** {pred_map[rf_pred]}")
    st.info(f"**XGBoost (XB):** {pred_map[xgb_pred]}")
