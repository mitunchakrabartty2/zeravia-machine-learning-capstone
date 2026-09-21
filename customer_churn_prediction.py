import pandas as pd

# read the dataset
df = pd.read_excel("Telco_customer_churn.xlsx")

# see the data
print(df.head())

print("Shape of data:", df.shape)

# check columns
print(df.columns)

# check information
df.info()

# check missing values
print("\nMissing values:")
print(df.isnull().sum())

# check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# check churn
print("\nChurn:")
print(df["Churn Label"].value_counts())

print("\nChurn percentage:")
print(df["Churn Label"].value_counts(normalize=True) * 100)


# -------------------------------
# Data cleaning
# -------------------------------

# Total Charges is stored as text
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# fill missing values in Total Charges
df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)

# remove columns which are not needed
df = df.drop([
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason"
], axis=1)

# convert churn into 0 and 1
df["Churn"] = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

# remove old churn column
df = df.drop("Churn Label", axis=1)

# check the cleaned data
print("\nCleaned data:")
print(df.head())

print("\nNew shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -------------------------------
# Exploratory Data Analysis
# -------------------------------

import matplotlib.pyplot as plt

# Churn count
print("\nChurn count:")
print(df["Churn"].value_counts())

# Churn percentage
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# Churn graph
df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ["No", "Yes"], rotation=0)

plt.show()

# -------------------------------
# Churn according to Contract
# -------------------------------

print("\nChurn by Contract:")
print(pd.crosstab(df["Contract"], df["Churn"]))

pd.crosstab(df["Contract"], df["Churn"]).plot(kind="bar")

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.show()

# -------------------------------
# Churn according to Tenure
# -------------------------------

print("\nAverage tenure:")
print(df.groupby("Churn")["Tenure Months"].mean())

# Tenure graph
df.boxplot(column="Tenure Months", by="Churn")

plt.title("Tenure Months and Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure Months")
plt.xticks([1, 2], ["No", "Yes"])

plt.show()
# -------------------------------
# Churn according to Monthly Charges
# -------------------------------

print("\nAverage monthly charges:")
print(df.groupby("Churn")["Monthly Charges"].mean())

# Monthly charges graph
df.boxplot(column="Monthly Charges", by="Churn")

plt.title("Monthly Charges and Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.xticks([1, 2], ["No", "Yes"])

plt.show()


# -------------------------------
# Feature Engineering
# -------------------------------

# Total number of additional services
service_columns = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies"
]

df["Total Services"] = 0

for column in service_columns:
    df["Total Services"] = df["Total Services"] + (
        df[column] == "Yes"
    ).astype(int)


# Average monthly spending
df["Average Monthly Spending"] = (
    df["Total Charges"] / df["Tenure Months"].replace(0, 1)
)


# New customer
df["Is New Customer"] = (
    df["Tenure Months"] <= 12
).astype(int)


# High monthly charge
df["High Monthly Charge"] = (
    df["Monthly Charges"] > df["Monthly Charges"].median()
).astype(int)


# Check new features
print("\nNew features:")
print(df[
    [
        "Total Services",
        "Average Monthly Spending",
        "Is New Customer",
        "High Monthly Charge"
    ]
].head())

print("\nFeature summary:")
print(df[
    [
        "Total Services",
        "Average Monthly Spending",
        "Is New Customer",
        "High Monthly Charge"
    ]
].describe())

# -------------------------------
# Convert categorical variables
# -------------------------------

# convert Yes/No columns into 0 and 1

yes_no_columns = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing"
]

for column in yes_no_columns:
    df[column] = df[column].map({
        "No": 0,
        "Yes": 1
    })


# convert Gender
df["Gender"] = df["Gender"].map({
    "Female": 0,
    "Male": 1
})


# convert remaining categorical columns
df = pd.get_dummies(
    df,
    columns=[
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Payment Method"
    ],
    drop_first=True,
    dtype=int
)


# check the data
print("\nData after encoding:")
print(df.head())

print("\nNew shape after encoding:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

# -------------------------------
# Check for outliers
# -------------------------------

numerical_columns = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Total Services",
    "Average Monthly Spending"
]

print("\nOutlier count:")

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, ":", len(outliers))
    
    # -------------------------------
# Handle outliers
# -------------------------------

column = "Average Monthly Spending"

Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df[column] = df[column].clip(
    lower=lower_limit,
    upper=upper_limit
)

print("\nOutliers handled successfully.")
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# -------------------------------
# Split the data
# -------------------------------

from sklearn.model_selection import train_test_split

# separate input and output
X = df.drop("Churn", axis=1)
y = df["Churn"]

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nChurn in training data:")
print(y_train.value_counts())

print("\nChurn in testing data:")
print(y_test.value_counts())

# -------------------------------
# Logistic Regression
# -------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# create the model
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ))
])

# train the model
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")

# -------------------------------
# Evaluate Logistic Regression
# -------------------------------

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# predict churn
y_pred = logistic_model.predict(X_test)

# predict probability of churn
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# calculate scores
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLogistic Regression Results:")
print("ROC-AUC:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# -------------------------------
# Random Forest
# -------------------------------

from sklearn.ensemble import RandomForestClassifier

# create the model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# train the model
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")

# -------------------------------
# Evaluate Random Forest
# -------------------------------

y_pred_rf = random_forest_model.predict(X_test)

y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("ROC-AUC:", roc_auc_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)

# -------------------------------
# XGBoost Model
# -------------------------------

from xgboost import XGBClassifier

# calculate class imbalance ratio
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

print("\nXGBoost model trained successfully.")

# -------------------------------
# XGBoost Evaluation
# -------------------------------

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# predict class
y_pred_xgb = xgb_model.predict(X_test)

# predict probability
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# calculate metrics
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print("ROC-AUC:", roc_auc_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1 Score:", f1_xgb)
# -------------------------------
# Model Comparison
# -------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    
    "ROC-AUC": [
        0.848634684440311,
        0.8379679144385027,
        0.8516921129453099
    ],
    
    "Precision": [
        0.519298245614035,
        0.6195286195286195,
        0.5198618307426598
    ],
    
    "Recall": [
        0.7914438502673797,
        0.4919786096256685,
        0.8048128342245989
    ],
    
    "F1 Score": [
        0.6271186440677966,
        0.5484351713859911,
        0.6316894018887723
    ]
})

print("\nModel Comparison:")
print(results)

# -------------------------------
# Calibration Analysis
# -------------------------------

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Predicted probabilities
prob_lr = logistic_model.predict_proba(X_test)[:, 1]
prob_rf = random_forest_model.predict_proba(X_test)[:, 1]
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Brier Score
brier_lr = brier_score_loss(y_test, prob_lr)
brier_rf = brier_score_loss(y_test, prob_rf)
brier_xgb = brier_score_loss(y_test, prob_xgb)

print("\nBrier Scores:")
print("Logistic Regression:", brier_lr)
print("Random Forest:", brier_rf)
print("XGBoost:", brier_xgb)

# -------------------------------
# Calibration Curve
# -------------------------------

from sklearn.calibration import calibration_curve

prob_true_lr, prob_pred_lr = calibration_curve(
    y_test, prob_lr, n_bins=10
)

prob_true_rf, prob_pred_rf = calibration_curve(
    y_test, prob_rf, n_bins=10
)

prob_true_xgb, prob_pred_xgb = calibration_curve(
    y_test, prob_xgb, n_bins=10
)

plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred_lr,
    prob_true_lr,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    prob_pred_rf,
    prob_true_rf,
    marker="o",
    label="Random Forest"
)

plt.plot(
    prob_pred_xgb,
    prob_true_xgb,
    marker="o",
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve for Churn Prediction Models")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# High-Risk Customer Ranking
# -------------------------------

# Original dataset 
raw_data = pd.read_excel("Telco_customer_churn.xlsx")

# XGBoost churn probability
churn_probability = xgb_model.predict_proba(X_test)[:, 1]

# Customer ID 
customer_ids = raw_data.loc[X_test.index, "CustomerID"]

# Ranking dataframe make
high_risk_customers = pd.DataFrame({
    "CustomerID": customer_ids,
    "Churn Probability": churn_probability
})

# Probability অনুযায়ী descending order
high_risk_customers = high_risk_customers.sort_values(
    by="Churn Probability",
    ascending=False
)

print("\nTop 20 High-Risk Customers:")
print(high_risk_customers.head(20))

# -------------------------------
# Review Indicators
# -------------------------------

# Top 20 customers original information take
risk_data = raw_data.loc[
    high_risk_customers.head(20).index
].copy()


def get_review_indicators(row):
    
    reasons = []

    if row["Contract"] == "Month-to-month":
        reasons.append("Month-to-month contract")

    if row["Tenure Months"] <= 12:
        reasons.append("New customer")

    if row["Monthly Charges"] > raw_data["Monthly Charges"].median():
        reasons.append("High monthly charge")

    if row["Payment Method"] == "Electronic check":
        reasons.append("Electronic check payment")

    if row["Tech Support"] == "No":
        reasons.append("No tech support")

    if row["Online Security"] == "No":
        reasons.append("No online security")

    if len(reasons) == 0:
        reasons.append("General high-risk prediction")

    return ", ".join(reasons)


risk_data["Review Indicators"] = risk_data.apply(
    get_review_indicators,
    axis=1
)

# Probability addition
risk_data["Churn Probability"] = high_risk_customers.head(20)[
    "Churn Probability"
].values


risk_report = risk_data[
    [
        "CustomerID",
        "Churn Probability",
        "Contract",
        "Tenure Months",
        "Monthly Charges",
        "Payment Method",
        "Review Indicators"
    ]
]

print("\nHigh-Risk Customer Review List:")
print(risk_report)

# -------------------------------
# Confusion Matrix
# -------------------------------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
y_pred_lr = logistic_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)

ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("Random Forest Confusion Matrix")
plt.show()


# XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

print("\nXGBoost Confusion Matrix:")
print(cm_xgb)

ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb,
    display_labels=["No Churn", "Churn"]
).plot()

plt.title("XGBoost Confusion Matrix")
plt.show()

# -------------------------------
# ROC Curve Comparison
# -------------------------------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities
prob_lr = logistic_model.predict_proba(X_test)[:, 1]
prob_rf = random_forest_model.predict_proba(X_test)[:, 1]
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)

# Calculate AUC
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {auc_lr:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost (AUC = {auc_xgb:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# XGBoost Feature Importance
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

# Plot top 15 features
top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------------
# Export High-Risk Customer List
# -------------------------------

risk_report.to_csv(
    "high_risk_customers.csv",
    index=False
)

print("\nHigh-risk customer list saved successfully.")
print("File: high_risk_customers.csv")

# -------------------------------
# Save XGBoost Model
# -------------------------------

import joblib

joblib.dump(
    xgb_model,
    "xgboost_churn_model.pkl"
)

joblib.dump(
    X_train.columns.tolist(),
    "model_features.pkl"
)

print("\nModel saved successfully.")
print("File: xgboost_churn_model.pkl")
print("File: model_features.pkl")