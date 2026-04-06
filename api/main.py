import sys
import os
import uuid
import json
import importlib.util
import joblib
import numpy as np
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# 1. FIX: Ensure Python can find the 'monitoring' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Setup Directories and Paths
LOG_DIR = "logs"
MONITOR_DIR = "monitoring"
DATA_DIR = "data"
PREDICTIONS_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.jsonl")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "training_features.csv")

for folder in [LOG_DIR, MONITOR_DIR, DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

app = FastAPI(title="Fraud Detection MLOps API")

# 3. Mount Static Files (to view the Drift Report in your browser)
app.mount("/reports", StaticFiles(directory=MONITOR_DIR), name="reports")

# 4. Load Model
try:
    model = joblib.load("models/model.pkl")
except Exception as e:
    print(f"CRITICAL: Model failed to load: {e}")
    model = None

# 5. Data Schemas (Pydantic v2)
class Transaction(BaseModel):
    features: List[float]

    @field_validator('features')
    @classmethod
    def validate_features_length(cls, v):
        if len(v) != 31:
            raise ValueError('Model requires exactly 31 features (Check if Class was included in training)')
        return v

class Feedback(BaseModel):
    request_id: str
    actual_label: int  # 0: Legitimate, 1: Fraud

# 6. Endpoints
@app.get("/", include_in_schema=False)
def home():
    """Redirect home to Swagger Docs"""
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(transaction: Transaction):
    """Inference + Logging"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model file missing on server.")

    request_id = str(uuid.uuid4())
    data = np.array(transaction.features).reshape(1, -1)
    
    # Run Prediction
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    # Log to JSONL
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "features": transaction.features,
        "prediction": prediction,
        "probability": probability
    }
    
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "request_id": request_id,
        "fraud_prediction": prediction,
        "fraud_probability": round(probability, 4),
        "status": "Logged"
    }

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    """Feedback Loop: Store ground truth labels"""
    feedback_entry = {
        "request_id": feedback.request_id,
        "actual_label": feedback.actual_label,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    return {"message": "Feedback stored"}

@app.get("/monitoring/accuracy")
def get_accuracy():
    """Performance Monitoring: Compare Preds vs Feedback"""
    if not os.path.exists(FEEDBACK_FILE) or not os.path.exists(PREDICTIONS_FILE):
        return {"error": "Insufficient data. Need both predictions and feedback."}

    with open(PREDICTIONS_FILE, "r") as f:
        preds = {json.loads(line)["request_id"]: json.loads(line)["prediction"] for line in f}
    
    with open(FEEDBACK_FILE, "r") as f:
        feedback_data = [json.loads(line) for line in f]

    correct = 0
    total = 0
    for item in feedback_data:
        rid = item["request_id"]
        if rid in preds:
            total += 1
            if preds[rid] == item["actual_label"]:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0
    return {"accuracy": f"{acc:.2f}%", "verified_samples": total}

@app.post("/monitoring/run-drift")
def trigger_drift():
    """Drift Detection: Generates Evidently HTML Report"""
    monitor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drift_path = os.path.join(monitor_root, "monitoring", "drift.py")
    if not os.path.exists(drift_path):
        drift_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring", "drift.py")

    if not os.path.exists(drift_path):
        return {"status": "error", "message": f"Drift module not found at {drift_path}"}

    spec = importlib.util.spec_from_file_location("monitoring.drift", drift_path)
    drift_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift_module)
    run_drift_check = drift_module.run_drift_check

    if not os.path.exists(PREDICTIONS_FILE):
        return {"error": "No live data found to compare."}
    if not os.path.exists(TRAIN_DATA_FILE):
        return {"error": f"Reference file {TRAIN_DATA_FILE} not found."}

    try:
        run_drift_check(TRAIN_DATA_FILE, PREDICTIONS_FILE)
        return {
            "status": "Report Generated",
            "view_at": "/reports/drift_report.html"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/logs/raw")
def get_raw_logs():
    """Debug only: View the last 10 logs"""
    if not os.path.exists(PREDICTIONS_FILE): return []
    with open(PREDICTIONS_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()[-10:]]
