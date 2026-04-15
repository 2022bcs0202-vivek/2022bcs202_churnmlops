
from fastapi import FastAPI
import joblib
import pandas as pd
import sys

sys.path.append("../src")

from preprocess import feature_engineering

app = FastAPI()

model = joblib.load("../models/model.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Feature engineering
    df = feature_engineering(df)

    # Convert categorical same as training
    df = pd.get_dummies(df)

    # Align columns with training model
    model_columns = model.feature_names_in_
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    return {"churn_prediction": int(prediction[0])}
