# Model Evaluation Report
## E-Commerce Customer Churn & Lifetime Value Predictor

### 1. Evaluation Methodology

The models were evaluated using chronological validation rather than relying solely on a random train-test split.

- Training period ended: 12 June 2011.
- Testing period ended: 9 December 2011.
- Churn prediction horizon: 90 days.
- CLV prediction horizon: 90 days.

This approach evaluates the models on a later period, providing a more realistic simulation of future predictions.

### 2. Churn Prediction Evaluation

**Selected model:** HistGradientBoostingClassifier

| Metric | Result |
|---|---:|
| Accuracy | 64% |
| Precision (Churn) | 56% |
| Recall (Churn) | 82% |
| F1-score (Churn) | 0.66 |
| ROC-AUC | 0.7122 |

**Interpretation**

The model achieved a churn recall of 82%, meaning it identified approximately 82% of customers who actually became inactive during the test period.

However, churn precision was 56%. This means some customers identified as high risk still made purchases.

The ROC-AUC of 0.7122 indicates that the model has useful ability to distinguish between customers who become inactive and those who remain active.

**Business implication:** The model can help businesses identify customers who may need retention campaigns. However, businesses should consider campaign costs and false-positive predictions before taking action.

### 3. Customer Lifetime Value Evaluation

**Selected model:** RandomForestRegressor with log-transformed target values.

| Metric | Result |
|---|---:|
| Mean Absolute Error | £603.06 |
| Root Mean Squared Error | £3,207.23 |
| R² Score | 0.383 |

**Interpretation**

The MAE indicates that the model's predictions differed from actual 90-day customer revenue by approximately £603 on average.

The R² score of 0.383 indicates that the model explains approximately 38.3% of the variation in future customer revenue.

### 4. Baseline Comparison

| Model | MAE |
|---|---:|
| Median baseline | £790.23 |
| HistGradientBoostingRegressor | £613.62 |
| RandomForestRegressor | £603.06 |

The Random Forest model achieved the lowest MAE among the evaluated approaches.

Its MAE was approximately 23.7% lower than the median baseline.

### 5. Model Selection

The final models selected were:

- **Churn:** HistGradientBoostingClassifier.
- **CLV:** RandomForestRegressor.

The chronological evaluation results informed the selection, and both selected models were subsequently integrated into the live Streamlit application.

### 6. Limitations

1. The dataset covers approximately one year of transactions.
2. Churn is defined as 90-day inactivity, not permanent customer loss.
3. CLV prediction is limited to a 90-day horizon.
4. The CLV model still has substantial prediction error.
5. The chronological validation includes customers who can appear in both training and testing periods. It evaluates future-period performance for the customer base, not performance on entirely unseen customers.

### 7. Conclusion

The project demonstrates how machine learning can support customer retention and revenue forecasting.

The churn model provides a useful starting point for identifying potentially inactive customers, while the CLV model estimates future customer revenue.

Although the models have limitations, the results demonstrate the practical value of customer analytics in e-commerce business decision-making.
