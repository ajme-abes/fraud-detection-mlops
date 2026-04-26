from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from api.background import start_scheduler
import os
import json
import uuid
import joblib
import numpy as np
from datetime import datetime, timezone
from typing import List

from monitoring.drift import run_drift_check
from src.config.config_loader import get_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_cfg = get_config()
LOG_DIR = os.path.join(BASE_DIR, _cfg["paths"]["logs_dir"])
DATA_DIR = os.path.join(BASE_DIR, "data")
PREDICTIONS_FILE = os.path.join(BASE_DIR, _cfg["paths"]["predictions_file"])
FEEDBACK_FILE = os.path.join(BASE_DIR, _cfg["paths"]["feedback_file"])
TRAIN_DATA_FILE = os.path.join(BASE_DIR, _cfg["paths"]["training_features"])

THRESHOLD = _cfg["model"]["prediction_threshold"]

os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model Store — wraps the active model so it can be hot-reloaded in place
# ---------------------------------------------------------------------------
class ModelStore:
    """
    Holds the currently active model and its version string.

    Call reload(version) to swap in a new model from disk without
    restarting the API process.
    """

    def __init__(self):
        self.model = None
        self.version: str = ""
        self.loaded_at: str = ""

    def load(self, version: str) -> bool:
        """
        Load a model by version label (e.g. 'v1', 'v2').
        Returns True on success, False if the file is missing or corrupt.
        """
        path = os.path.join(BASE_DIR, "models", version, "model.pkl")
        if not os.path.exists(path):
            print(f"[ModelStore] Model file not found: {path}")
            return False
        try:
            self.model = joblib.load(path)
            self.version = version
            self.loaded_at = datetime.now(timezone.utc).isoformat()
            print(f"[ModelStore] Loaded model {version} from {path}")
            return True
        except Exception as e:
            print(f"[ModelStore] Failed to load model {version}: {e}")
            return False

    def reload(self, version: str) -> bool:
        """Hot-swap the active model. Same as load() but logs the swap."""
        previous = self.version
        success = self.load(version)
        if success:
            print(f"[ModelStore] Hot-reloaded: {previous} → {version}")
        return success

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Singleton — shared across all requests in this process
model_store = ModelStore()


# ---------------------------------------------------------------------------
# Lifespan — startup logic (load model + start scheduler)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    # ENV var takes priority, falls back to config.yaml, then "v1"
    default_version = os.getenv("MODEL_VERSION", cfg["model"]["default_version"])
    model_store.load(default_version)
    start_scheduler()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Fraud Detection MLOps API", lifespan=lifespan)

# Serve drift report
app.mount("/reports", StaticFiles(directory="monitoring"), name="reports")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
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

    @field_validator("actual_label")
    @classmethod
    def validate_label(cls, v):
        if v not in (0, 1):
            raise ValueError("actual_label must be 0 (legitimate) or 1 (fraud)")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_store.is_loaded,
        "model_version": model_store.version,
        "loaded_at": model_store.loaded_at,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if not model_store.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    data = np.array(transaction.features).reshape(1, -1)

    prob = float(model_store.model.predict_proba(data)[0][1])
    pred = 1 if prob > THRESHOLD else 0

    log = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": transaction.features,
        "prediction": pred,
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
        "model_version": model_store.version,
    }

    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")

    return {
        "request_id": request_id,
        "prediction": pred,
        "probability": round(prob, 4),
    }


@app.post("/feedback")
def feedback(data: Feedback):
    if not os.path.exists(PREDICTIONS_FILE):
        raise HTTPException(status_code=400, detail="No predictions found")

    with open(PREDICTIONS_FILE) as f:
        preds = {json.loads(line)["request_id"] for line in f}

    if data.request_id not in preds:
        raise HTTPException(
            status_code=404,
            detail="Invalid request_id. No matching prediction found.",
        )

    entry = {
        "request_id": data.request_id,
        "actual_label": data.actual_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "Feedback stored"}


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
        feedback_entries = [json.loads(l) for l in f]

    correct, total = 0, 0
    for item in feedback_entries:
        rid = item["request_id"]
        if rid in preds:
            total += 1
            if preds[rid] == item["actual_label"]:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0
    return {"accuracy": f"{acc:.2f}%", "samples": total}


@app.get("/logs/raw")
def get_raw_logs():
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    with open(PREDICTIONS_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]


@app.post("/monitoring/retrain")
def trigger_retraining():
    from api.train_v2 import retrain_model

    try:
        success = retrain_model()
    except Exception as e:
        import traceback
        print(f"[retrain] FATAL ERROR:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

    if not success:
        return {"message": "Retraining skipped (not enough new data)."}

    # Hot-reload the freshly saved v2 model into memory
    reloaded = model_store.reload("v2")

    if reloaded:
        return {
            "message": "Model retrained and hot-reloaded successfully.",
            "active_version": model_store.version,
            "loaded_at": model_store.loaded_at,
        }
    else:
        return {
            "message": "Model retrained but reload failed — restart the API to apply.",
            "active_version": model_store.version,
        }
