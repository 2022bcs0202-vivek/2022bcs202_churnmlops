import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score

from preprocess import feature_engineering

# Load dataset
df = pd.read_csv("../data/churn.csv")

# -------------------------------
# TARGET COLUMN FIX
# -------------------------------
df['Churn'] = df['Churn'].astype(int)

# -------------------------------
# DROP USELESS COLUMN
# -------------------------------
if 'Phone number' in df.columns:
    df = df.drop('Phone number', axis=1)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df = feature_engineering(df)

# -------------------------------
# HANDLE CATEGORICAL DATA
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# SPLIT FEATURES & TARGET
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MLflow START
# -------------------------------
mlflow.start_run()

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------------
# METRICS
# -------------------------------
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("F1 Score:", f1)
print("ROC-AUC:", roc)
print("Precision:", precision)

# Log metrics
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc)
mlflow.log_metric("precision", precision)

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(model, "../models/model.pkl")

# Log model
mlflow.sklearn.log_model(model, name="model")

mlflow.end_run()
