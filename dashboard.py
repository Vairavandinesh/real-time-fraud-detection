
import csv
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
CASES_FILE = Path("fraud_cases.csv")
PREDICTIONS_FILE = Path("fraud_predictions.csv")

st.set_page_config(page_title="Fraud Detection Platform", layout="wide")
st.title("Real-Time Financial Fraud Detection")
st.caption("PaySim | XGBoost | FastAPI | Kafka | Evidently AI")

# --------------------------------------------------
# 1. Kafka prediction history
# --------------------------------------------------
st.header("Kafka Transaction Predictions")

if PREDICTIONS_FILE.exists() and PREDICTIONS_FILE.stat().st_size > 0:
    predictions = pd.read_csv(PREDICTIONS_FILE)

    if not predictions.empty:
        total = len(predictions)
        fraud_count = (predictions["prediction"] == "Fraud").sum()
        normal_count = (predictions["prediction"] == "Normal").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Processed", total)
        c2.metric("Flagged as Fraud", int(fraud_count))
        c3.metric("Classified as Normal", int(normal_count))

        st.dataframe(predictions, use_container_width=True)

        st.download_button(
            "Download prediction history",
            data=PREDICTIONS_FILE.read_bytes(),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("The prediction history is empty.")
else:
    st.info(
        "No Kafka predictions found yet. Start the Kafka consumer "
        "and producer to populate this section."
    )

st.divider()

# --------------------------------------------------
# 2. Manual transaction scoring
# --------------------------------------------------
st.header("Score a Transaction")

with st.form("transaction_form"):
    c1, c2, c3 = st.columns(3)

    amount = c1.number_input("Amount", min_value=0.0, value=9000.0)
    old_org = c2.number_input(
        "Origin balance before", min_value=0.0, value=10000.0
    )
    new_org = c3.number_input(
        "Origin balance after", min_value=0.0, value=1000.0
    )

    old_dest = c1.number_input(
        "Destination balance before", min_value=0.0, value=5000.0
    )
    new_dest = c2.number_input(
        "Destination balance after", min_value=0.0, value=14000.0
    )
    txn_type = c3.selectbox(
        "Transaction type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    )

    submitted = st.form_submit_button("Score transaction")

if submitted:
    payload = {
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "type": txn_type,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.session_state["last_result"] = {
            **payload,
            **result,
        }
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")

# --------------------------------------------------
# 3. Prediction result and analyst review
# --------------------------------------------------
result = st.session_state.get("last_result")

if result:
    st.subheader("Prediction Result")
    m1, m2 = st.columns(2)

    m1.metric("Classification", result["prediction"])
    m2.metric("Fraud Probability", f'{result["fraud_probability"]:.2%}')

    case_status = st.selectbox(
        "Case status",
        ["New", "Under Review", "Confirmed Fraud", "False Positive", "Closed"],
        key="case_status",
    )

    analyst_feedback = st.text_input(
        "Analyst notes",
        key="analyst_notes",
    )

    if st.button("Save analyst review"):
        fields = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "type",
            "prediction", "fraud_probability", "case_status",
            "analyst_feedback",
        ]

        write_header = (
            not CASES_FILE.exists() or CASES_FILE.stat().st_size == 0
        )

        with CASES_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if write_header:
                writer.writeheader()

            writer.writerow({
                **result,
                "case_status": case_status,
                "analyst_feedback": analyst_feedback,
            })

        st.success("Analyst review saved.")

# --------------------------------------------------
# 4. Saved analyst cases
# --------------------------------------------------
st.divider()
st.header("Saved Analyst Cases")

if CASES_FILE.exists() and CASES_FILE.stat().st_size > 0:
    cases = pd.read_csv(CASES_FILE)
    st.dataframe(cases, use_container_width=True)

    st.download_button(
        "Download case records",
        data=CASES_FILE.read_bytes(),
        file_name="fraud_cases.csv",
        mime="text/csv",
    )
else:
    st.info("No analyst reviews saved yet.")
