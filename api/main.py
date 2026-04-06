from fastapi import FastAPI
from fastapi.responses import RedirectResponse  # Add this import
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Load model once at startup
model = joblib.load("models/model.pkl")

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def home():
    # This will automatically send the browser to /docs
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
