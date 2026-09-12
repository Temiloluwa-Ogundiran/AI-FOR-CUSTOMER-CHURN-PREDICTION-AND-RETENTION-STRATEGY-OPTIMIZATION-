"""Churn Prediction + Retention Decision Dashboard (Ch.4 reference implementation).
SYNTHETIC DEMONSTRATION ONLY. Not Nigerian bank performance. No causal churn-reduction claim.
Run: streamlit run app.py
"""
import os, json
import joblib
import pandas as pd
import streamlit as st
from churn_core import (
    ALL_FEATURES, NUMERIC_FEATURES, ALLOWED_CHANNELS, ALLOWED_REGIONS,
    score_customers, reason_codes, risk_band, expected_value,
    SAVE_PROB, TREATMENT_COST, CONTACT_COST, FALSE_POSITIVE_HARM,
)

BASE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(BASE, "selected_pipeline.joblib")

st.set_page_config(page_title="AI Churn & Retention Dashboard (Synthetic Demo)", layout="wide")
st.title("AI Churn Prediction & Retention Decision Support — Synthetic Demo")

@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE):
        return None
    return joblib.load(BUNDLE)

bundle = load_bundle()
if bundle is None:
    st.error("No trained pipeline found. Run `python train.py` first to generate selected_pipeline.joblib.")
    st.stop()

model = bundle["model"]
threshold = float(bundle["threshold"])
report = bundle.get("report", {})

# Sidebar: model audit
st.sidebar.header("Model audit (locked test, synthetic)")
tm = report.get("test_metrics_selected", {})
for k in ["roc_auc", "pr_auc", "brier", "recall", "precision", "lift@10%"]:
    st.sidebar.metric(k, f"{tm.get(k, 0):.4f}" if isinstance(tm.get(k), float) else tm.get(k))
st.sidebar.write(f"Selected: **{report.get('selected_model')}** @ thr={threshold:.2f}")
st.sidebar.write(f"Prevalence (synthetic): {report.get('prevalence', 0):.1%}")
st.sidebar.write(f"Seed: {report.get('seed')}, n={report.get('n_records')}")
st.sidebar.caption(f"Economics illustrative: save_p={SAVE_PROB}, cost={TREATMENT_COST}, contact={CONTACT_COST}, fp_harm={FALSE_POSITIVE_HARM}")

tab1, tab2, tab3 = st.tabs(["Single customer scoring", "Batch CSV scoring", "Audit: Top-50 + importance"])

with tab1:
    st.subheader("Input customer snapshot (one row at prediction timestamp)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 18, 100, 35)
        tenure_months = st.number_input("Tenure (months)", 0, 360, 24)
        num_products = st.selectbox("Num products", [1, 2, 3, 4], index=0)
        is_active_member = st.selectbox("Active member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        balance = st.number_input("Balance (NGN)", 0.0, 10000000.0, 150000.0)
        estimated_clv = st.number_input("Estimated CLV (NGN)", 0.0, 10000000.0, 120000.0)
    with c2:
        txn_avg_90d = st.number_input("Txn avg / month (90d)", 0.0, 200.0, 8.0)
        txn_count_30d = st.number_input("Txn count (30d)", 0, 200, 6)
        app_sessions_30d = st.number_input("App sessions (30d)", 0, 200, 8)
        days_since_last_txn = st.number_input("Days since last txn", 0, 365, 5)
    with c3:
        failed_txn_rate = st.slider("Failed txn rate", 0.0, 1.0, 0.03, 0.01)
        num_complaints = st.number_input("Num complaints", 0, 20, 0)
        num_unresolved = st.number_input("Num unresolved", 0, 20, 0)
        downtime_exposure = st.number_input("Downtime exposure (days)", 0, 30, 0)
    with c4:
        fraud_flag = st.selectbox("Fraud flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        monthly_fee = st.selectbox("Monthly fee (NGN)", [0, 50, 100, 250, 500], index=1)
        primary_channel = st.selectbox("Primary channel", ALLOWED_CHANNELS)
        region_group = st.selectbox("Region group", ALLOWED_REGIONS)

    human_ok = st.checkbox("Human reviewer approves contact if flagged (required for action)")
    if st.button("Score customer", type="primary"):
        row = pd.DataFrame([{
            "age": age, "tenure_months": tenure_months, "num_products": num_products,
            "is_active_member": is_active_member, "balance": balance, "estimated_clv": estimated_clv,
            "txn_avg_90d": txn_avg_90d, "txn_count_30d": txn_count_30d,
            "app_sessions_30d": app_sessions_30d, "days_since_last_txn": days_since_last_txn,
            "failed_txn_rate": failed_txn_rate, "num_complaints": num_complaints,
            "num_unresolved": num_unresolved, "downtime_exposure": downtime_exposure,
            "fraud_flag": fraud_flag, "monthly_fee": monthly_fee,
            "primary_channel": primary_channel, "region_group": region_group,
        }])
        scored = score_customers(model, row, threshold).iloc[0]
        p = float(scored["churn_probability"])
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn probability", f"{p:.1%}")
        m2.metric("Risk band", scored["risk_band"])
        m3.metric("Expected retention value (illustrative)", f"₦{scored['expected_retention_value']:,.0f}")
        m4.metric("Decision", scored["decision"])
        st.info(f"**Recommended action:** {scored['recommended_action']}")
        st.write(f"**Reason codes (associative, not causal):** {scored['reason_codes']}")
        st.caption(f"Evidence: failed_rate={failed_txn_rate:.0%}, unresolved={num_unresolved}, "
                   f"inactivity={days_since_last_txn}d, txns30={txn_count_30d}, sessions30={app_sessions_30d} | "
                   f"Assumes save_p={SAVE_PROB}, treatment=₦{TREATMENT_COST:,.0f}. {scored['suppression_note']}.")
        if "FLAG" in scored["decision"] and not human_ok:
            st.error("Flagged — contact BLOCKED until human reviewer approves (tick the checkbox).")
        elif "FLAG" in scored["decision"]:
            st.success("Flagged — approved for next step per contact policy + complaint escalation path.")

with tab2:
    st.subheader("Batch scoring (CSV with schema columns)")
    st.caption(f"Required columns: {', '.join(ALL_FEATURES)}. Unknown categories ignored safely. Target excluded from features.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            bdf = pd.read_csv(up)
            missing = [c for c in ALL_FEATURES if c not in bdf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                out = score_customers(model, bdf, threshold)
                st.dataframe(out.head(50))
                st.download_button("Download scored CSV", out.to_csv(index=False), "scored_customers.csv")
        except Exception as e:
            st.error(f"Scoring failed: {e}")
    st.markdown("No file? Use the synthetic demo set:")
    if os.path.exists(os.path.join(BASE, "all_scored_customers.csv")):
        demo = pd.read_csv(os.path.join(BASE, "all_scored_customers.csv"), nrows=200)
        st.dataframe(demo.head(20))

with tab3:
    st.subheader("Top-50 highest-risk (synthetic)")
    p50 = os.path.join(BASE, "Top_50_scored_customers.csv")
    if os.path.exists(p50):
        st.dataframe(pd.read_csv(p50))
    st.subheader("Global diagnostics: permutation importance (test, synthetic)")
    pi = os.path.join(BASE, "permutation_importance.csv")
    if os.path.exists(pi):
        pidf = pd.read_csv(pi)
        st.bar_chart(pidf.set_index("feature")["importance_mean"].head(15))
        st.dataframe(pidf)
    st.subheader("Full audit report")
    st.json(report)

st.divider()
st.caption(
    "Limitations (Ch.4.14): synthetic records; no external validation on Nigerian bank data; no live CRM/core-banking "
    "integration; no randomized campaign — churn reduction not causally demonstrated; expected-value uses illustrative "
    "assumptions; explanations are associative. Production gates: DPIA/lawful basis, reconciled tokenized extraction, "
    "temporal back-test, shadow mode, randomized pilot, independent validation, monitoring + rollback."
)
