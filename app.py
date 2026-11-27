
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================
# LOAD TRAINED MODELS
# =====================================

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

churn_model = joblib.load(
    os.path.join(PROJECT_FOLDER, "churn_model.pkl")
)

clv_model = joblib.load(
    os.path.join(PROJECT_FOLDER, "clv_model.pkl")
)

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Customer Churn & CLV Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Customer Intelligence Dashboard")

st.write(
    "Predict customer churn risk and estimate customer "
    "lifetime value over the next 90 days."
)

st.divider()

# =====================================
# CUSTOMER INPUTS
# =====================================

st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "Days Since Last Purchase",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Number of Previous Orders",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Total Historical Spending (£)",
        min_value=0.0,
        value=1000.0
    )

with col2:
    average_order_value = st.number_input(
        "Average Order Value (£)",
        min_value=0.0,
        value=200.0
    )

    tenure = st.number_input(
        "Customer Tenure (Days)",
        min_value=0,
        value=180
    )

    purchase_interval = st.number_input(
        "Average Days Between Orders (Enter 0 if unknown)",
        min_value=0.0,
        value=45.0
    )

st.divider()

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Customer Behaviour", type="primary"):

    if purchase_interval == 0:
        purchase_interval = np.nan

    customer = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Average_Order_Value": average_order_value,
        "Tenure": tenure,
        "Purchase_Interval": purchase_interval
    }])

    # Churn prediction
    churn_probability = churn_model.predict_proba(customer)[0]

    churn_classes = list(churn_model.classes_)
    churn_index = churn_classes.index(1)

    probability = churn_probability[churn_index] * 100

    # CLV prediction
    predicted_clv = clv_model.predict(customer)[0]
    predicted_clv = max(0, predicted_clv)

    # =================================
    # DISPLAY RESULTS
    # =================================

    st.subheader("Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Churn Probability",
            f"{probability:.1f}%"
        )

        if probability >= 70:
            st.error("High Churn Risk")
        elif probability >= 40:
            st.warning("Medium Churn Risk")
        else:
            st.success("Low Churn Risk")

    with result2:
        st.metric(
            "Predicted 90-Day CLV",
            f"£{predicted_clv:,.2f}"
        )

    # =================================
    # BUSINESS RECOMMENDATION
    # =================================

    st.subheader("Business Recommendation")

    if probability >= 70:
        st.write(
            "This customer has a high predicted churn risk. "
            "Consider targeted retention campaigns, personalised "
            "offers, or customer re-engagement."
        )

    elif probability >= 40:
        st.write(
            "This customer has a moderate churn risk. "
            "Consider monitoring their purchase activity and "
            "sending relevant promotions."
        )

    else:
        st.write(
            "This customer has a relatively low predicted churn "
            "risk. Consider loyalty rewards and strategies to "
            "maintain engagement."
        )

    st.caption(
        "Predictions are estimates based on historical purchasing "
        "patterns, not guarantees of future behaviour."
    )

st.divider()

st.caption(
    "Model outputs: churn probability and predicted revenue "
    "over the next 90 days. Currency: GBP (£)."
)
