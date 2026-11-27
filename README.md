# E-Commerce Customer Churn & Lifetime Value Predictor

An end-to-end Data Science and Machine Learning project that predicts customer churn, estimates future customer revenue, and segments customers based on purchasing behaviour.

**Live Dashboard:** https://ecommerce-churn-clv-predictor-abb.streamlit.app/

## 1. Project Overview

Customer retention is important for the long-term growth of e-commerce businesses. Losing customers can reduce revenue and increase the cost of acquiring new customers.

This project uses historical retail transaction data to develop machine learning models that help businesses:

- Identify customers at risk of becoming inactive.
- Estimate customer revenue over the next 90 days.
- Segment customers based on purchasing behaviour.
- Develop targeted customer retention and marketing strategies.

## 2. Project Objectives

1. Develop a machine learning model to predict customer churn.
2. Build a regression model to estimate 90-day Customer Lifetime Value (CLV).
3. Perform customer segmentation using Recency, Frequency and Monetary (RFM) analysis.
4. Identify important factors influencing customer behaviour.
5. Develop an interactive Streamlit dashboard for business decision-making.

## 3. Dataset

**Source:** UCI Machine Learning Repository – Online Retail Dataset.

The dataset contains transactions from a UK-based online retail business between December 2010 and December 2011.

Original dataset:
- 541,909 transactions
- 8 columns
- Customer transaction and purchasing information

Important features include:
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

## 4. Data Cleaning and Preparation

The data preparation process included:

- Removing duplicate records.
- Handling missing CustomerID values.
- Removing cancelled transactions and negative quantities.
- Removing non-product transaction descriptions.
- Investigating and removing two extreme transaction anomalies.
- Creating a Revenue feature using Quantity × UnitPrice.
- Preparing customer-level features for modelling.

Final cleaned transaction dataset: **392,433 records**.

## 5. Exploratory Data Analysis

The analysis explored:

- Customer purchasing frequency.
- Customer spending patterns.
- Purchase recency.
- Customer tenure.
- Revenue distribution.
- Correlations between purchasing behaviour and spending.

Key finding: Customers with more than 10 invoices represented approximately 7.7% of customers but generated approximately 50.66% of revenue.

## 6. Feature Engineering

The following customer-level features were developed:

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of unique orders |
| Monetary | Total historical spending |
| Average Order Value | Average revenue per order |
| Tenure | Days since first purchase |
| Purchase Interval | Average time between purchases |

## 7. Machine Learning Models

### Churn Prediction

**Algorithm:** HistGradientBoostingClassifier

Target definition: A customer is classified as churned if they make no purchase during the following 90 days.

Chronological test results:

- Accuracy: 64%
- Churn precision: 56%
- Churn recall: 82%
- Churn F1-score: 0.66
- ROC-AUC: 0.712

### Customer Lifetime Value Prediction

**Algorithm:** RandomForestRegressor with log-transformed target values.

Target: Actual customer revenue over the following 90 days.

Chronological test results:

- Mean Absolute Error (MAE): £603.06
- Root Mean Squared Error (RMSE): £3,207.23
- R² Score: 0.383

The model outperformed the median prediction baseline, which had an MAE of £790.23.

## 8. Customer Segmentation

Customers were grouped into six RFM-based segments:

1. Champions
2. Loyal Customers
3. Potential Loyalists
4. At Risk
5. Hibernating
6. Regular Customers

The segmentation supports targeted marketing and customer retention strategies.

Key findings:

- Champions represented approximately 22.48% of customers.
- Champions generated approximately 64.81% of observed future revenue.
- Hibernating customers had the highest observed churn rate at approximately 66.04%.

These are historical test-period observations, not model predictions.

## 9. Interactive Dashboard

The project includes a Streamlit dashboard that provides:

- Customer churn probability.
- Predicted 90-day CLV.
- Automatic customer segmentation.
- Segment-specific marketing recommendations.
- Customer distribution charts.
- Future revenue contribution charts.
- Historical segment performance summaries.

**Try the application:** https://ecommerce-churn-clv-predictor-abb.streamlit.app/

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib
- Google Colab
- GitHub

## 11. Project Limitations

- The dataset covers approximately one year of transactions.
- Churn represents 90-day inactivity, not confirmed permanent customer loss.
- CLV estimates are limited to a 90-day prediction horizon.
- The models were evaluated on a chronological holdout period, and the same customers can appear in both historical periods.
- Performance may differ when applied to other e-commerce businesses.

## 12. Future Improvements

- Hyperparameter optimisation.
- More advanced customer behaviour features.
- Testing on additional datasets.
- Improved model explainability.
- Longer-term CLV forecasting.
- Automated model monitoring.

## 13. Conclusion

This project demonstrates an end-to-end Data Science workflow, from data cleaning and exploratory analysis to feature engineering, machine learning, model evaluation and deployment.

It demonstrates how customer transaction data can be transformed into actionable insights that support customer retention and revenue growth.
