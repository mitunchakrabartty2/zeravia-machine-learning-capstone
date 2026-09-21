# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:19:48 2026

@author: mitun
"""

import streamlit as st
import pandas as pd
import joblib


# Load saved model and feature list
model = joblib.load("model/xgboost_churn_model.pkl")
model_features = joblib.load("model/model_features.pkl")


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Title
st.title("Customer Churn Prediction System")

st.write(
    "Enter customer information below to estimate the probability of customer churn."
)


# Customer information
st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )


with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# Billing information
st.header("Billing Information")

col4, col5, col6 = st.columns(3)


with col4:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


with col5:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)",
            "Bank transfer (automatic)"
        ]
    )


with col6:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )


# Prediction button
if st.button("Predict Churn", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Senior Citizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [1 if partner == "Yes" else 0],
        "Dependents": [1 if dependents == "Yes" else 0],
        "Tenure Months": [tenure],
        "Phone Service": [1 if phone_service == "Yes" else 0],
        "Paperless Billing": [1 if paperless_billing == "Yes" else 0],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges]
    })


    # Create engineered features
    service_columns = [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    total_services = sum(
        1 for service in service_columns
        if service == "Yes"
    )

    input_data["Total Services"] = total_services

    input_data["Average Monthly Spending"] = (
        total_charges / max(tenure, 1)
    )

    input_data["Is New Customer"] = (
        1 if tenure <= 12 else 0
    )

    input_data["High Monthly Charge"] = (
        1 if monthly_charges > 70.35 else 0
    )


    # Create encoded categorical features
    categorical_data = {
        "Multiple Lines_No phone service":
            1 if multiple_lines == "No phone service" else 0,

        "Multiple Lines_Yes":
            1 if multiple_lines == "Yes" else 0,

        "Internet Service_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "Internet Service_No":
            1 if internet_service == "No" else 0,

        "Online Security_No internet service":
            1 if online_security == "No internet service" else 0,

        "Online Security_Yes":
            1 if online_security == "Yes" else 0,

        "Online Backup_No internet service":
            1 if online_backup == "No internet service" else 0,

        "Online Backup_Yes":
            1 if online_backup == "Yes" else 0,

        "Device Protection_No internet service":
            1 if device_protection == "No internet service" else 0,

        "Device Protection_Yes":
            1 if device_protection == "Yes" else 0,

        "Tech Support_No internet service":
            1 if tech_support == "No internet service" else 0,

        "Tech Support_Yes":
            1 if tech_support == "Yes" else 0,

        "Streaming TV_No internet service":
            1 if streaming_tv == "No internet service" else 0,

        "Streaming TV_Yes":
            1 if streaming_tv == "Yes" else 0,

        "Streaming Movies_No internet service":
            1 if streaming_movies == "No internet service" else 0,

        "Streaming Movies_Yes":
            1 if streaming_movies == "Yes" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "Payment Method_Credit card (automatic)":
            1 if payment_method == "Credit card (automatic)" else 0,

        "Payment Method_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "Payment Method_Mailed check":
            1 if payment_method == "Mailed check" else 0
    }


    for feature, value in categorical_data.items():
        input_data[feature] = value


    # Make sure feature order is exactly the same as training
    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )


    # Make prediction
    probability = model.predict_proba(
        input_data
    )[0][1]


    # Display result
    st.header("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.70:
        st.error("High Risk of Churn")

    elif probability >= 0.40:
        st.warning("Medium Risk of Churn")

    else:
        st.success("Low Risk of Churn")
        
# Review Indicators
st.subheader("Review Indicators")

review_indicators = []

if contract == "Month-to-month":
    review_indicators.append("Month-to-month contract")

if tenure <= 12:
    review_indicators.append("New customer")

if monthly_charges > 70.35:
    review_indicators.append("High monthly charge")

if payment_method == "Electronic check":
    review_indicators.append("Electronic check payment")

if tech_support == "No":
    review_indicators.append("No tech support")

if online_security == "No":
    review_indicators.append("No online security")

if len(review_indicators) == 0:
    review_indicators.append("No specific review indicator identified")

for indicator in review_indicators:
    st.write("•", indicator)
    
# -------------------------------
# High-Risk Customer List
# -------------------------------

st.subheader("High-Risk Customer List")

st.write(
    "Download the ranked list of customers identified as high-risk by the XGBoost model."
)

try:
    high_risk_data = pd.read_csv("high_risk_customers.csv")

    csv_data = high_risk_data.to_csv(index=False)

    st.download_button(
        label="Download High-Risk Customer List",
        data=csv_data,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.warning(
        "High-risk customer list is not available."
    )
    
# -------------------------------
# Model Performance
# -------------------------------

st.subheader("Model Performance")

performance_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "ROC-AUC": [
        0.849,
        0.838,
        0.852
    ],
    "Precision": [
        0.519,
        0.620,
        0.520
    ],
    "Recall": [
        0.791,
        0.492,
        0.805
    ],
    "F1 Score": [
        0.627,
        0.548,
        0.632
    ]
})

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Performance metrics were calculated on the held-out test dataset."
)

# -------------------------------
# Calibration Performance
# -------------------------------

st.subheader("Calibration Performance")

calibration_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Brier Score": [
        0.1634,
        0.1390,
        0.1589
    ]
})

st.dataframe(
    calibration_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Lower Brier Score indicates better probability calibration."
)


# -------------------------------
# Feature Importance
# -------------------------------

st.subheader("Top XGBoost Features")

feature_data = pd.DataFrame({
    "Feature": [
        "Contract_Two year",
        "Internet Service_Fiber optic",
        "Contract_One year",
        "Internet Service_No",
        "Streaming Movies_Yes",
        "Dependents",
        "Payment Method_Electronic check",
        "Tenure Months",
        "Online Security_Yes",
        "Paperless Billing"
    ],
    "Importance": [
        0.2575,
        0.1422,
        0.1306,
        0.0932,
        0.0640,
        0.0531,
        0.0510,
        0.0272,
        0.0199,
        0.0182
    ]
})

st.dataframe(
    feature_data,
    use_container_width=True,
    hide_index=True
)