# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

import joblib


# Load dataset

df = pd.read_csv("ai4i2020.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# Check the data

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMachine failure distribution:")
print(df["Machine failure"].value_counts())

print("\nMachine failure percentage:")
print(
    df["Machine failure"].value_counts(normalize=True) * 100
)


# Sensor columns

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# Basic sensor analysis

print("\nAverage sensor values:")
print(
    df.groupby("Machine failure")[sensor_cols].mean()
)

print("\nMedian sensor values:")
print(
    df.groupby("Machine failure")[sensor_cols].median()
)

print("\nFailure rate by machine type:")
print(
    df.groupby("Type")["Machine failure"].mean() * 100
)


# Failure modes

failure_cols = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

print("\nFailure mode counts:")
print(df[failure_cols].sum())

print("\nFailure mode percentages:")
print(
    df[failure_cols].mean() * 100
)


# Plot machine failure distribution

plt.figure(figsize=(6, 4))

df["Machine failure"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Records")
plt.xticks(
    [0, 1],
    ["No Failure", "Failure"],
    rotation=0
)

plt.tight_layout()
plt.show()


# Plot sensor values for failed and non-failed machines

for col in sensor_cols:

    plt.figure(figsize=(7, 4))

    df.boxplot(
        column=col,
        by="Machine failure"
    )

    plt.title(col)
    plt.suptitle("")
    plt.xlabel("Machine Failure")
    plt.ylabel(col)

    plt.tight_layout()
    plt.show()


# Create a future failure window

df["Failure Window"] = (
    df["Machine failure"]
    .rolling(5, min_periods=1)
    .max()
    .shift(-4)
    .fillna(0)
    .astype(int)
)

print("\nFailure window distribution:")
print(
    df["Failure Window"].value_counts()
)


# Create lag features

for col in sensor_cols:

    df[col + "_Lag1"] = df[col].shift(1)
    df[col + "_Lag2"] = df[col].shift(2)


# Create rolling features

for col in sensor_cols:

    df[col + "_RollingMean"] = (
        df[col].rolling(5).mean()
    )

    df[col + "_RollingStd"] = (
        df[col].rolling(5).std()
    )


# Create a few additional features

df["Temperature Difference"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power Proxy"] = (
    df["Torque [Nm]"]
    * df["Rotational speed [rpm]"]
)

df["Torque Speed Ratio"] = (
    df["Torque [Nm]"]
    / df["Rotational speed [rpm]"]
)


# Remove rows with missing values

df = df.dropna().reset_index(drop=True)

print("\nShape after feature engineering:")
print(df.shape)

print("\nMissing values after feature engineering:")
print(df.isnull().sum().sum())


# Select features for the model

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference",
    "Power Proxy",
    "Torque Speed Ratio"
]

for col in sensor_cols:

    features.append(col + "_Lag1")
    features.append(col + "_Lag2")
    features.append(col + "_RollingMean")
    features.append(col + "_RollingStd")


X = df[features]
y = df["Machine failure"]


# Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# Scale features for Logistic Regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression model

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_scaled,
    y_train
)

lr_prob = lr_model.predict_proba(
    X_test_scaled
)[:, 1]

lr_pred = (
    lr_prob >= 0.50
).astype(int)


# Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]

rf_pred = (
    rf_prob >= 0.50
).astype(int)


# Gradient Boosting model

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_prob = gb_model.predict_proba(
    X_test
)[:, 1]

gb_pred = (
    gb_prob >= 0.50
).astype(int)


# Function to evaluate the models

def evaluate_model(name, actual, predicted, probability):

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual,
        probability
    )

    pr_auc = average_precision_score(
        actual,
        probability
    )

    print("\n" + name)
    print("-" * 35)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            actual,
            predicted
        )
    )

    return [
        name,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]


# Evaluate all models

lr_result = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred,
    lr_prob
)

rf_result = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)

gb_result = evaluate_model(
    "Gradient Boosting",
    y_test,
    gb_pred,
    gb_prob
)


# Compare model performance

results = pd.DataFrame(
    [
        lr_result,
        rf_result,
        gb_result
    ],
    columns=[
        "Model",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC"
    ]
)

print("\nModel comparison:")
print(
    results.round(4)
)


# Precision-Recall curve

precision_lr, recall_lr, _ = precision_recall_curve(
    y_test,
    lr_prob
)

precision_rf, recall_rf, _ = precision_recall_curve(
    y_test,
    rf_prob
)

precision_gb, recall_gb, _ = precision_recall_curve(
    y_test,
    gb_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_lr,
    precision_lr,
    label="Logistic Regression"
)

plt.plot(
    recall_rf,
    precision_rf,
    label="Random Forest"
)

plt.plot(
    recall_gb,
    precision_gb,
    label="Gradient Boosting"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ROC curve

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    lr_prob
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_prob
)

fpr_gb, tpr_gb, _ = roc_curve(
    y_test,
    gb_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_lr,
    tpr_lr,
    label="Logistic Regression"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label="Random Forest"
)

plt.plot(
    fpr_gb,
    tpr_gb,
    label="Gradient Boosting"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Calculate failure risk for each machine

df["Failure Probability"] = (
    gb_model.predict_proba(
        df[features]
    )[:, 1]
)

df["Risk Score"] = (
    df["Failure Probability"] * 100
)


# Create risk categories

def get_risk_level(score):

    if score >= 75:
        return "Critical"

    elif score >= 50:
        return "Warning"

    elif score >= 25:
        return "Watch"

    else:
        return "Normal"


df["Risk Level"] = (
    df["Risk Score"].apply(get_risk_level)
)


# Find high-risk machines

high_risk = df[
    df["Risk Score"] >= 75
].copy()

high_risk = high_risk.sort_values(
    by="Risk Score",
    ascending=False
)

risk_columns = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Failure Probability",
    "Risk Score",
    "Risk Level"
]

print("\nHigh-risk machines:")
print(
    high_risk[
        risk_columns
    ].head(20)
)


# Save high-risk machine list

high_risk[
    risk_columns
].to_csv(
    "high_risk_machines.csv",
    index=False
)


# Check important failure modes

failure_mode_report = (
    df[df["Machine failure"] == 1][failure_cols]
    .sum()
    .sort_values(ascending=False)
)

print("\nFailure mode report:")
print(failure_mode_report)


# Feature importance

importance = pd.DataFrame({
    "Feature": features,
    "Importance": gb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop important features:")
print(
    importance.head(15)
)


# Plot feature importance

top_features = importance.head(15)

plt.figure(figsize=(8, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Important Features for Machine Failure")
plt.tight_layout()
plt.show()


# Save model files

joblib.dump(
    gb_model,
    "predictive_maintenance_model.pkl"
)

joblib.dump(
    scaler,
    "predictive_maintenance_scaler.pkl"
)

joblib.dump(
    features,
    "predictive_maintenance_features.pkl"
)


# Save model comparison

results.to_csv(
    "predictive_maintenance_model_comparison.csv",
    index=False
)


# Final output

print("\nProject completed successfully.")

print("\nSaved files:")
print("high_risk_machines.csv")
print("predictive_maintenance_model.pkl")
print("predictive_maintenance_scaler.pkl")
print("predictive_maintenance_features.pkl")
print("predictive_maintenance_model_comparison.csv")

# Time-based validation

data = df.copy()

data = data.sort_values("UDI").reset_index(drop=True)

split_point = int(len(data) * 0.80)

train_data = data.iloc[:split_point]
test_data = data.iloc[split_point:]

X_train_time = train_data[features]
y_train_time = train_data["Machine failure"]

X_test_time = test_data[features]
y_test_time = test_data["Machine failure"]

print("\nTime-based validation")
print("Training data:", X_train_time.shape)
print("Testing data:", X_test_time.shape)

print("\nTraining failure distribution:")
print(y_train_time.value_counts())

print("\nTesting failure distribution:")
print(y_test_time.value_counts())


# Train a Random Forest using the earlier part of the data

time_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

time_model.fit(
    X_train_time,
    y_train_time
)

time_prob = time_model.predict_proba(
    X_test_time
)[:, 1]

time_pred = (
    time_prob >= 0.50
).astype(int)


# Evaluate time-based model

time_precision = precision_score(
    y_test_time,
    time_pred,
    zero_division=0
)

time_recall = recall_score(
    y_test_time,
    time_pred,
    zero_division=0
)

time_f1 = f1_score(
    y_test_time,
    time_pred,
    zero_division=0
)

time_roc_auc = roc_auc_score(
    y_test_time,
    time_prob
)

time_pr_auc = average_precision_score(
    y_test_time,
    time_prob
)

print("\nTime-based model results:")

print("Precision:", round(time_precision, 4))
print("Recall:", round(time_recall, 4))
print("F1 Score:", round(time_f1, 4))
print("ROC-AUC:", round(time_roc_auc, 4))
print("PR-AUC:", round(time_pr_auc, 4))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_time,
        time_pred
    )
)

# Check different alert thresholds

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

threshold_results = []

for threshold in thresholds:

    prediction = (
        time_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test_time,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test_time,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test_time,
        prediction,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test_time,
        prediction
    )

    false_alarms = matrix[0, 1]
    missed_failures = matrix[1, 0]

    threshold_results.append([
        threshold,
        precision,
        recall,
        f1,
        false_alarms,
        missed_failures
    ])


threshold_table = pd.DataFrame(
    threshold_results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1 Score",
        "False Alarms",
        "Missed Failures"
    ]
)

print("\nThreshold analysis:")
print(
    threshold_table.round(4)
)


# Plot threshold results

plt.figure(figsize=(7, 5))

plt.plot(
    threshold_table["Threshold"],
    threshold_table["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_table["Threshold"],
    threshold_table["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_table["Threshold"],
    threshold_table["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Performance at Different Alert Thresholds")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Save threshold results

threshold_table.to_csv(
    "threshold_analysis.csv",
    index=False
)

# Early warning alert system

alert_data = test_data[
    ["UDI", "Product ID", "Machine failure"]
].copy()

alert_data["Failure Probability"] = time_prob


def assign_alert_level(probability):

    if probability >= 0.75:
        return "Critical"

    elif probability >= 0.50:
        return "Warning"

    elif probability >= 0.25:
        return "Watch"

    else:
        return "Normal"


alert_data["Alert Level"] = alert_data[
    "Failure Probability"
].apply(assign_alert_level)


print("\nAlert level distribution:")
print(
    alert_data["Alert Level"].value_counts()
)


print("\nFailure count by alert level:")
print(
    pd.crosstab(
        alert_data["Alert Level"],
        alert_data["Machine failure"]
    )
)


alert_summary = (
    alert_data
    .groupby("Alert Level")
    .agg(
        Machines=("UDI", "count"),
        Actual_Failures=("Machine failure", "sum"),
        Average_Probability=("Failure Probability", "mean")
    )
    .reset_index()
)


alert_summary["Failure Rate"] = (
    alert_summary["Actual_Failures"]
    / alert_summary["Machines"]
)


alert_order = [
    "Normal",
    "Watch",
    "Warning",
    "Critical"
]

alert_summary["Alert Level"] = pd.Categorical(
    alert_summary["Alert Level"],
    categories=alert_order,
    ordered=True
)

alert_summary = alert_summary.sort_values(
    "Alert Level"
)


print("\nEarly warning summary:")
print(
    alert_summary.round(4)
)


# Plot alert levels

plt.figure(figsize=(8, 5))

plt.bar(
    alert_summary["Alert Level"].astype(str),
    alert_summary["Actual_Failures"]
)

plt.xlabel("Alert Level")
plt.ylabel("Actual Failures")
plt.title("Actual Failures Detected at Different Alert Levels")

plt.tight_layout()
plt.show()


# Save alert report

alert_data.to_csv(
    "early_warning_alerts.csv",
    index=False
)

alert_summary.to_csv(
    "early_warning_summary.csv",
    index=False
)

# Final early warning evaluation

total_failures = alert_data["Machine failure"].sum()

critical_failures = alert_data[
    (alert_data["Alert Level"] == "Critical") &
    (alert_data["Machine failure"] == 1)
].shape[0]

warning_failures = alert_data[
    (alert_data["Alert Level"] == "Warning") &
    (alert_data["Machine failure"] == 1)
].shape[0]

watch_failures = alert_data[
    (alert_data["Alert Level"] == "Watch") &
    (alert_data["Machine failure"] == 1)
].shape[0]

normal_failures = alert_data[
    (alert_data["Alert Level"] == "Normal") &
    (alert_data["Machine failure"] == 1)
].shape[0]

detected_failures = (
    critical_failures +
    warning_failures +
    watch_failures
)

detection_rate = (
    detected_failures / total_failures
) * 100

critical_rate = (
    critical_failures / total_failures
) * 100


print("\nFinal Early Warning Evaluation")

print("Total actual failures:", total_failures)

print("Failures in Critical:", critical_failures)
print("Failures in Warning:", warning_failures)
print("Failures in Watch:", watch_failures)
print("Failures in Normal:", normal_failures)

print("\nDetected before Normal level:", detected_failures)

print(
    "Early warning detection rate:",
    round(detection_rate, 2),
    "%"
)

print(
    "Critical alert detection rate:",
    round(critical_rate, 2),
    "%"
)


# Alert level performance table

evaluation_table = pd.DataFrame({
    "Alert Level": [
        "Critical",
        "Warning",
        "Watch",
        "Normal"
    ],
    "Actual Failures": [
        critical_failures,
        warning_failures,
        watch_failures,
        normal_failures
    ]
})

evaluation_table["Percentage"] = (
    evaluation_table["Actual Failures"]
    / total_failures
) * 100


print("\nAlert level evaluation:")
print(evaluation_table.round(2))


# Plot detected failures

plt.figure(figsize=(7,5))

plt.bar(
    evaluation_table["Alert Level"],
    evaluation_table["Actual Failures"]
)

plt.xlabel("Alert Level")
plt.ylabel("Actual Failures")
plt.title("Failure Detection Across Alert Levels")

plt.tight_layout()
plt.show()


# Save evaluation

evaluation_table.to_csv(
    "early_warning_evaluation.csv",
    index=False
)