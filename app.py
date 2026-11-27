import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Customer Churn & CLV Predictor",
    page_icon="📊",
    layout="wide"
)

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# =====================================
# LOAD MODELS AND SUPPORTING FILES
# =====================================

@st.cache_resource
def load_models():
    churn = joblib.load(os.path.join(PROJECT_FOLDER, "churn_model.pkl"))
    clv = joblib.load(os.path.join(PROJECT_FOLDER, "clv_model.pkl"))
    return churn, clv

churn_model, clv_model = load_models()

threshold_path = os.path.join(PROJECT_FOLDER, "rfm_thresholds.json")
with open(threshold_path, "r") as file:
    rfm_thresholds = json.load(file)

summary_path = os.path.join(PROJECT_FOLDER, "segment_summary.csv")
segment_summary = None

if os.path.exists(summary_path):
    segment_summary = pd.read_csv(summary_path, index_col=0)

# =====================================
# CUSTOMER SEGMENT CLASSIFICATION
# =====================================

def classify_customer(recency, frequency, monetary):
    r_score = 5 - np.searchsorted(
        rfm_thresholds["Recency"], recency, side="right"
    )

    f_score = 1 + np.searchsorted(
        rfm_thresholds["Frequency"], frequency, side="left"
    )

    m_score = 1 + np.searchsorted(
        rfm_thresholds["Monetary"], monetary, side="right"
    )

    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    elif f_score >= 4 and r_score >= 3:
        return "Loyal Customers"
    elif r_score >= 4 and f_score >= 2:
        return "Potential Loyalists"
    elif r_score <= 2 and f_score >= 3:
        return "At Risk"
    elif r_score <= 2 and f_score <= 2:
        return "Hibernating"
    else:
        return "Regular Customers"


def segment_recommendation(segment):
    recommendations = {
        "Champions": (
            "Reward this customer's loyalty with VIP benefits, early access "
            "to new products, referral rewards, and personalised experiences."
        ),
        "Loyal Customers": (
            "Strengthen the relationship with loyalty rewards, exclusive "
            "offers, and product recommendations based on purchase history."
        ),
        "Potential Loyalists": (
            "Encourage repeat purchases with a second-purchase incentive, "
            "product recommendations, and a simple loyalty programme."
        ),
        "At Risk": (
            "Launch a win-back campaign with a personalised message, relevant "
            "discount, or reminder of products the customer may need."
        ),
        "Hibernating": (
            "Try a low-cost reactivation campaign. Test a tailored offer, but "
            "avoid spending heavily if the customer does not respond."
        ),
        "Regular Customers": (
            "Build purchase frequency through useful recommendations, "
            "replenishment reminders, and targeted promotions."
        )
    }
    return recommendations.get(segment, "Monitor this customer's activity.")

# =====================================
# DASHBOARD HEADER
# =====================================

st.title("📊 E-Commerce Customer Intelligence Dashboard")
st.write(
    "Predict customer churn risk, estimate revenue over the next 90 days, "
    "and identify customer segments from historical purchasing behaviour."
)
st.caption("Currency: GBP (£) | Prediction horizon: 90 days")
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

predict_clicked = st.button(
    "Predict Customer Behaviour",
    type="primary"
)

# =====================================
# PREDICTIONS AND CUSTOMER INSIGHTS
# =====================================

if predict_clicked:
    model_purchase_interval = (
        np.nan if purchase_interval == 0 else purchase_interval
    )

    customer = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Average_Order_Value": average_order_value,
        "Tenure": tenure,
        "Purchase_Interval": model_purchase_interval
    }])

    # Churn prediction
    churn_probabilities = churn_model.predict_proba(customer)[0]
    churn_classes = list(churn_model.classes_)
    churn_index = churn_classes.index(1)
    probability = churn_probabilities[churn_index] * 100

    # CLV prediction
    predicted_clv = max(0, float(clv_model.predict(customer)[0]))

    # RFM customer segment
    customer_segment = classify_customer(
        recency, frequency, monetary
    )

    st.divider()
    st.subheader("Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric("Churn Probability", f"{probability:.1f}%")
        if probability >= 70:
            st.error("High Churn Risk")
        elif probability >= 40:
            st.warning("Medium Churn Risk")
        else:
            st.success("Low Churn Risk")

    with result2:
        st.metric("Predicted 90-Day CLV", f"£{predicted_clv:,.2f}")

    with result3:
        st.metric("Customer Segment", customer_segment)

    # Business recommendations
    st.subheader("Customer Strategy")
    st.write(segment_recommendation(customer_segment))

    if probability >= 70:
        st.warning(
            "The churn model also indicates high churn risk. Prioritise "
            "a retention or re-engagement action."
        )
    elif probability >= 40:
        st.info(
            "The churn model indicates moderate churn risk. Monitor activity "
            "and consider a targeted retention message."
        )
    else:
        st.success(
            "The churn model indicates relatively low churn risk. Focus on "
            "maintaining engagement and encouraging repeat purchases."
        )

    st.caption(
        "The customer segment is based on RFM thresholds from the reference "
        "customer cohort. Churn and CLV are model estimates, not guarantees."
    )

# =====================================
# HISTORICAL SEGMENT BUSINESS INSIGHTS
# =====================================

st.divider()
st.header("Customer Segmentation Insights")
st.write(
    "These charts summarise the historical customer segments from the "
    "project dataset. They are descriptive results, not live predictions."
)

if segment_summary is not None and not segment_summary.empty:
    required_columns = {
        "Customer_Count",
        "Total_Future_Revenue",
        "Actual_Churn_Rate"
    }

    if required_columns.issubset(segment_summary.columns):
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Customers by Segment")
            st.bar_chart(
                segment_summary[["Customer_Count"]].sort_values(
                    "Customer_Count", ascending=False
                )
            )

        with chart_col2:
            st.subheader("Future Revenue by Segment")
            st.bar_chart(
                segment_summary[["Total_Future_Revenue"]].sort_values(
                    "Total_Future_Revenue", ascending=False
                )
            )

        st.subheader("Segment Performance Summary")
        display_summary = segment_summary.copy()

        if "Average_Spending" in display_summary.columns:
            display_summary["Average_Spending"] = display_summary[
                "Average_Spending"
            ].map(lambda value: f"£{value:,.2f}")

        if "Average_Future_CLV" in display_summary.columns:
            display_summary["Average_Future_CLV"] = display_summary[
                "Average_Future_CLV"
            ].map(lambda value: f"£{value:,.2f}")

        if "Total_Future_Revenue" in display_summary.columns:
            display_summary["Total_Future_Revenue"] = display_summary[
                "Total_Future_Revenue"
            ].map(lambda value: f"£{value:,.2f}")

        if "Actual_Churn_Rate" in display_summary.columns:
            display_summary["Actual_Churn_Rate"] = display_summary[
                "Actual_Churn_Rate"
            ].map(lambda value: f"{value:.1f}%")

        st.dataframe(display_summary, use_container_width=True)
    else:
        st.info(
            "The segment summary file is present but does not contain all "
            "required columns for the charts."
        )
else:
    st.info(
        "To display historical segment charts, add segment_summary.csv "
        "to the same project folder as app.py."
    )

st.divider()
st.caption(
    "Churn probability estimates the likelihood of no purchase in the "
    "following 90 days. CLV estimates revenue over that same horizon. "
    "Customer segments are rule-based RFM classifications. Currency: GBP (£)."
)
