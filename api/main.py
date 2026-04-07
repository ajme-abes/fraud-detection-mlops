from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import os
import json
import uuid
import joblib
import numpy as np
from datetime import datetime
from typing import List

from monitoring.drift import run_drift_check 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

PREDICTIONS_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.jsonl")
TRAIN_DATA_FILE = os.path.join(BASE_DIR, "data", "training_features.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Fraud Detection MLOps API")

# Serve drift report
app.mount("/reports", StaticFiles(directory="monitoring"), name="reports")

# -------------------------------
# Load Model
# -------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Model load error: {e}")
    model = None

# -------------------------------
# Schemas
# -------------------------------
class Transaction(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 31:
            raise ValueError("Expected 31 features")
        return v


class Feedback(BaseModel):
    request_id: str
    actual_label: int

# -------------------------------
# Routes
# -------------------------------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    data = np.array(transaction.features).reshape(1, -1)

    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0][1])

    log = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "features": transaction.features,
        "prediction": pred,
        "probability": prob
    }

    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")

    return {
        "request_id": request_id,
        "prediction": pred,
        "probability": round(prob, 4)
    }


@app.post("/feedback")
def feedback(data: Feedback):

    if not os.path.exists(PREDICTIONS_FILE):
        raise HTTPException(status_code=400, detail="No predictions found")

    # Load existing predictions
    with open(PREDICTIONS_FILE) as f:
        preds = {json.loads(line)["request_id"] for line in f}

    # Validate request_id
    if data.request_id not in preds:
        raise HTTPException(
            status_code=404,
            detail="Invalid request_id. No matching prediction found."
        )

    entry = {
        "request_id": data.request_id,
        "actual_label": data.actual_label,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "Success Feedback stored"}


@app.post("/monitoring/run-drift")
def run_drift():
    if not os.path.exists(PREDICTIONS_FILE):
        return {"error": "No predictions yet"}

    if not os.path.exists(TRAIN_DATA_FILE):
        return {"error": "Missing training_features.csv"}

    try:
        run_drift_check(TRAIN_DATA_FILE, PREDICTIONS_FILE)
        return {"status": "done", "report": "/reports/drift_report.html"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/monitoring/accuracy")
def accuracy():
    if not os.path.exists(PREDICTIONS_FILE) or not os.path.exists(FEEDBACK_FILE):
        return {"error": "Not enough data"}

    with open(PREDICTIONS_FILE) as f:
        preds = {json.loads(l)["request_id"]: json.loads(l)["prediction"] for l in f}

    with open(FEEDBACK_FILE) as f:
        feedback = [json.loads(l) for l in f]

    correct, total = 0, 0

    for item in feedback:
        rid = item["request_id"]
        if rid in preds:
            total += 1
            if preds[rid] == item["actual_label"]:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0

    return {"accuracy": f"{acc:.2f}%", "samples": total}