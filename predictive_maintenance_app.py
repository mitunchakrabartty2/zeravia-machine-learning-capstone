# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:11:29 2026

@author: mitun
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Machine Health & Early Failure Warning",
    page_icon="⚙️",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef4ff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #24104f 0%, #152d63 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #6a2cff, #1689e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #56627a;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        border: 1px solid #dfe5f5;
        box-shadow: 0 6px 20px rgba(45, 55, 100, 0.08);
        text-align: center;
    }

    .info-label {
        color: #68748a;
        font-size: 14px;
    }

    .info-value {
        color: #202b4d;
        font-size: 25px;
        font-weight: 800;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef5ff, #f7f0ff);
        border-left: 6px solid #6a4cff;
        margin-top: 20px;
    }

    .section-title {
        color: #202b4d;
        font-size: 27px;
        font-weight: 750;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("model/predictive_maintenance_model.pkl")
    scaler = joblib.load("model/predictive_maintenance_scaler.pkl")
    features = joblib.load("model/predictive_maintenance_features.pkl")
    return model, scaler, features


@st.cache_data
def load_results():
    model_comparison = pd.read_csv(
        "results/predictive_maintenance_model_comparison.csv"
    )

    high_risk = pd.read_csv(
        "results/high_risk_machines.csv"
    )

    threshold = pd.read_csv(
        "results/threshold_analysis.csv"
    )

    early_warning = pd.read_csv(
        "results/early_warning_summary.csv"
    )

    return model_comparison, high_risk, threshold, early_warning


model, scaler, feature_names = load_model()
model_comparison, high_risk, threshold, early_warning = load_results()


st.sidebar.markdown("## ⚙️ Machine Health Settings")
st.sidebar.markdown(
    "Enter the current machine sensor readings "
    "to estimate failure risk."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Recommendation Engine")
st.sidebar.markdown(
    """
    • Sensor Analysis  
    • Gradient Boosting  
    • Early Warning  
    • Risk Classification  
    • Maintenance Review
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "Predictive Maintenance & Early Failure Warning"
)


st.markdown(
    '<div class="main-title">Machine Health & Early Failure Warning System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict machine failure risk from sensor conditions and identify machines '
    'that may require maintenance review.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">🔧 Enter Machine Sensor Information</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=280.0,
        max_value=320.0,
        value=298.0,
        step=0.1
    )

with col2:
    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=290.0,
        max_value=330.0,
        value=308.0,
        step=0.1
    )

with col3:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000.0,
        max_value=3000.0,
        value=1500.0,
        step=10.0
    )


col4, col5 = st.columns(2)

with col4:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5
    )

with col5:
    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=1.0
    )


st.markdown("---")


if st.button(
    "🔍 Analyze Machine Health",
    type="primary",
    use_container_width=True
):

    temperature_difference = (
        process_temperature - air_temperature
    )

    power_proxy = (
        torque * rotational_speed
    )

    torque_speed_ratio = (
        torque / rotational_speed
    )

    sensor_values = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    input_data = {}

    for feature in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]:
        input_data[feature] = sensor_values[feature]

    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    for column in sensor_columns:
        input_data[column + " Lag1"] = sensor_values[column]
        input_data[column + " Lag2"] = sensor_values[column]
        input_data[column + " RollingMean"] = sensor_values[column]
        input_data[column + " RollingStd"] = 0.0

    input_data["Temperature Difference"] = temperature_difference
    input_data["Power Proxy"] = power_proxy
    input_data["Torque Speed Ratio"] = torque_speed_ratio

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    scaled_input = scaler.transform(input_df)

    failure_probability = model.predict_proba(
        scaled_input
    )[0, 1]

    prediction = int(
        failure_probability >= 0.50
    )

    if failure_probability >= 0.70:
        risk_level = "Critical"
    elif failure_probability >= 0.50:
        risk_level = "Warning"
    elif failure_probability >= 0.30:
        risk_level = "Watch"
    else:
        risk_level = "Normal"


    st.markdown(
        '<div class="section-title">📊 Machine Health Assessment</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Failure Probability</div>
                <div class="info-value">
                    {failure_probability * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Risk Level</div>
                <div class="info-value">
                    {risk_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        status = "Failure Risk Detected" if prediction else "No Immediate Failure Risk"

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">System Status</div>
                <div class="info-value">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    if risk_level == "Critical":
        st.error(
            "Critical condition: immediate maintenance review is recommended."
        )

    elif risk_level == "Warning":
        st.warning(
            "Warning condition: maintenance inspection should be considered."
        )

    elif risk_level == "Watch":
        st.info(
            "Watch condition: continue monitoring the machine."
        )

    else:
        st.success(
            "Normal condition: no immediate failure risk detected."
        )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">📈 Derived Sensor Indicators</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Temperature Difference",
            f"{temperature_difference:.2f} K"
        )

    with d2:
        st.metric(
            "Power Proxy",
            f"{power_proxy:,.0f}"
        )

    with d3:
        st.metric(
            "Torque Speed Ratio",
            f"{torque_speed_ratio:.5f}"
        )


st.markdown(
    '<div class="section-title">📋 Model Performance</div>',
    unsafe_allow_html=True
)

st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True
)


st.markdown(
    '<div class="section-title">🚨 High-Risk Machine List</div>',
    unsafe_allow_html=True
)

if not high_risk.empty:
    st.dataframe(
        high_risk.head(20),
        use_container_width=True,
        hide_index=True
    )


st.markdown(
    '<div class="section-title">🎯 Threshold Analysis</div>',
    unsafe_allow_html=True
)

if not threshold.empty:
    st.dataframe(
        threshold,
        use_container_width=True,
        hide_index=True
    )


st.markdown(
    '<div class="section-title">⏱️ Early Warning Summary</div>',
    unsafe_allow_html=True
)

if not early_warning.empty:
    st.dataframe(
        early_warning,
        use_container_width=True,
        hide_index=True
    )


with st.expander("ℹ️ About the Predictive Maintenance System"):

    st.write(
        "This application uses a saved Gradient Boosting model trained on "
        "machine sensor data. The modeling workflow includes lag features, "
        "rolling statistics, derived sensor indicators and imbalance-aware "
        "evaluation."
    )

    st.write(
        "The live single-reading interface uses the current sensor reading "
        "as a proxy for lag and rolling features because no live historical "
        "sensor stream is provided in this demonstration."
    )

    st.write(
        "The failure probability is a model score and should be interpreted "
        "as a risk estimate rather than a calibrated probability."
    )


st.markdown("---")

st.markdown(
    "<center>Machine Health & Early Failure Warning System</center>",
    unsafe_allow_html=True
)
