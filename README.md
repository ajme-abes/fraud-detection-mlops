# Fraud Detection MLOps System

> End-to-end production ML system for real-time credit card fraud detection — built with a full MLOps lifecycle including experiment tracking, data drift monitoring, automated retraining, and live model hot-swapping.

[![CI Pipeline](https://github.com/your-username/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/fraud-detection-mlops/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)

---

## Live Demo

| Service | URL |
|---|---|
| REST API + Swagger UI | https://fraud-detection-my6t.onrender.com/docs |
| Monitoring Dashboard | https://fraud-dashboard-29kv.onrender.com |

> Render free tier spins down after inactivity — first request may take ~30 seconds to wake up.

---

## What This System Does

A user submits a credit card transaction → the API scores it in real time → predictions are logged → users submit feedback (actual labels) → the system monitors for data drift → when enough feedback accumulates, a new model is retrained and hot-swapped into memory without restarting the server.

```
Raw Data (Kaggle)
      │
      ▼
Training Pipeline
  ├── Preprocessing (StandardScaler)
  ├── Feature Engineering (hour-of-day)
  ├── XGBoost + scale_pos_weight (handles 577:1 imbalance)
  ├── MLflow experiment tracking
  └── SHAP explainability
      │
      ▼
Model Registry (models/v1, v2, ...)
      │
      ▼
FastAPI — Real-time Serving
  ├── POST /predict        → fraud score + probability
  ├── POST /feedback       → submit actual label
  ├── GET  /monitoring/accuracy → live accuracy from feedback
  ├── POST /monitoring/run-drift → Evidently drift report
  └── POST /monitoring/retrain  → retrain + hot-reload model
      │
      ├──▶ Streamlit Dashboard (metrics, predictions, drift)
      └──▶ Scheduled drift check (every 5 min via APScheduler)
```

---

## Key Features

- **Real-time prediction** — sub-100ms fraud scoring with configurable probability threshold
- **Feedback loop** — collect ground truth labels post-prediction to measure live accuracy
- **Data drift monitoring** — Evidently-powered HTML reports comparing training vs. live distributions
- **Automated retraining** — combines original training data with feedback-labeled samples, retrains XGBoost, evaluates, and saves a new model version
- **Hot model reload** — new model swapped into memory instantly via `ModelStore`, zero downtime
- **MLflow tracking** — every training run logs hyperparameters, F1, PR-AUC, and SHAP artifacts
- **SHAP explainability** — feature importance summary generated at training time
- **Centralized config** — all hyperparameters, paths, and thresholds in one `config.yaml`
- **69 automated tests** — covering prediction logic, model loading, hot-reload, retraining, config validation, and MLflow logging correctness
- **Dockerized** — full multi-service deployment via docker-compose

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | XGBoost 2.0 |
| Experiment Tracking | MLflow 2.13 |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Dashboard | Streamlit |
| Drift Monitoring | Evidently |
| Scheduling | APScheduler |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## Project Structure

```
fraud-detection-mlops/
│
├── api/
│   ├── main.py              # FastAPI app — predict, feedback, drift, retrain
│   ├── background.py        # APScheduler — automated drift checks every 5 min
│   ├── train_v2.py          # Retraining logic — combines original + feedback data
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py               # Streamlit monitoring dashboard
│   └── Dockerfile
│
├── src/
│   ├── config/
│   │   ├── config.yaml      # Centralized config — paths, thresholds, hyperparams
│   │   └── config_loader.py # Cached YAML loader used across all modules
│   ├── data/
│   │   ├── load_data.py     # Data loading and validation
│   │   └── preprocess.py    # Scaling, splitting, saving processed splits
│   ├── features/
│   │   └── build_features.py  # Feature engineering (hour-of-day from Time)
│   ├── models/
│   │   ├── train.py         # XGBoost training with MLflow + SHAP
│   │   └── predict.py       # predict(), predict_batch(), load_model()
│   └── pipelines/
│       └── training_pipeline.py  # End-to-end training pipeline
│
├── models/
│   ├── v1/model.pkl         # Initial trained model
│   └── v2/model.pkl         # Retrained model (generated after retraining)
│
├── monitoring/
│   ├── drift.py             # Evidently drift check
│   └── drift_report.html    # Generated drift report (served via /reports)
│
├── notebooks/
│   └── 01_eda.ipynb         # EDA — class imbalance, distributions, correlations
│
├── data/
│   ├── raw/creditcard.csv   # Source dataset (not committed — download from Kaggle)
│   ├── processed/           # Train/test splits (generated by pipeline)
│   └── training_features.csv  # Feature snapshot used for drift reference + retraining
│
├── logs/
│   ├── predictions.jsonl    # All prediction logs with features + probabilities
│   └── feedback.jsonl       # User-submitted ground truth labels
│
├── tests/
│   ├── test_predict.py      # 17 tests — prediction logic, thresholds, model loading
│   ├── test_train_v2.py     # 8 tests  — retraining pipeline end-to-end
│   ├── test_model_store.py  # 17 tests — hot-reload, load/reload, failure handling
│   ├── test_config.py       # 20 tests — config loader, schema, caching, errors
│   └── test_mlflow_logging.py # 7 tests — param/metric logging correctness
│
├── scripts/
│   └── prepare_drift_reference.py
│
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for containerized run)
- [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place at `data/raw/creditcard.csv`

### 1. Clone and install

```bash
git clone https://github.com/ajme-abes/fraud-detection-mlops.git
cd fraud-detection-mlops
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.pipelines.training_pipeline
```

This preprocesses the data, engineers features, trains XGBoost with MLflow tracking, saves `models/v1/model.pkl`, and generates a SHAP summary plot.

### 3a. Run locally (no Docker)

```bash
# Terminal 1 — API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Dashboard
python -m streamlit run dashboard/app.py
```

### 3b. Run with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

---

## API Reference

### `POST /predict`

Score a transaction for fraud.

**Request:**
```json
{
  "features": [0.0, -1.35, 1.19, 0.26, 0.16, 0.45, 0.23, 0.09, 0.36, 0.09,
               -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.25,
               -0.01, 0.27, -0.11, 0.06, -0.18, -0.14, -0.05, -0.06, 0.01, 149.62, 0.0]
}
```

**Response:**
```json
{
  "request_id": "3f7a1b2c-...",
  "prediction": 0,
  "probability": 0.0312
}
```

`prediction`: `1` = fraud, `0` = legitimate. Threshold: `0.7` (configurable in `config.yaml`).

---

### `POST /feedback`

Submit the actual label for a past prediction to track live accuracy.

```json
{
  "request_id": "3f7a1b2c-...",
  "actual_label": 0
}
```

`actual_label` must be `0` or `1`.

---

### `GET /monitoring/accuracy`

Returns accuracy computed from all feedback submitted so far.

```json
{ "accuracy": "94.50%", "samples": 40 }
```

---

### `POST /monitoring/run-drift`

Triggers an Evidently drift report comparing training data vs. recent predictions.
Report available at `/reports/drift_report.html`.

---

### `POST /monitoring/retrain`

Retrains the model using original training data + feedback-labeled samples.
Requires at least 5 feedback samples. On success, hot-reloads the new model into memory.

```json
{
  "message": "Model retrained and hot-reloaded successfully.",
  "active_version": "v2",
  "loaded_at": "2026-04-26T21:00:58+00:00"
}
```

---

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1",
  "loaded_at": "2026-04-26T20:00:00+00:00"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

**69 tests across 5 modules — all passing.**

| Module | Tests | Covers |
|---|---|---|
| `test_predict.py` | 17 | Prediction logic, thresholds, batch, model loading |
| `test_model_store.py` | 17 | Hot-reload, load/reload, failure recovery |
| `test_config.py` | 20 | Config loader, schema validation, caching |
| `test_train_v2.py` | 8 | Retraining pipeline, XGBoost output, edge cases |
| `test_mlflow_logging.py` | 7 | Param/metric logging matches actual model params |

---

## MLflow Experiment Tracking

MLflow tracks every training run locally. To view:

```bash
mlflow ui
```

Open http://localhost:5000 to browse experiments, compare runs, and inspect SHAP artifacts.

MLflow is intentionally not deployed to Render — it's a training-time tool. The tracking data lives in `mlflow.db` locally.

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group

- 284,807 transactions over 2 days
- 492 fraud cases (0.17% — severely imbalanced)
- Features `V1–V28` are PCA-transformed for confidentiality
- `Time`, `Amount`, and `Class` are the original columns

The raw CSV is not committed. Download from Kaggle and place at `data/raw/creditcard.csv`.

---

## Design Decisions

**Why XGBoost?** Handles class imbalance natively via `scale_pos_weight`, fast inference, and strong performance on tabular data without feature normalization.

**Why threshold 0.7?** In fraud detection, false negatives (missed fraud) are more costly than false positives. A higher threshold reduces false positives for a production setting where flagging legitimate transactions has real user impact.

**Why hot-reload instead of restart?** Restarting a production API to deploy a new model causes downtime. The `ModelStore` pattern swaps the model object in memory atomically — zero downtime, no dropped requests.

**Why not deploy MLflow?** MLflow UI is a development tool for comparing training runs. Deploying it adds infrastructure cost and complexity with no benefit to end users. It runs locally where it's actually useful.

---

## CI/CD

GitHub Actions runs on every push to `main`:
- Sets up Python 3.11
- Installs all dependencies
- Runs the full test suite

See `.github/workflows/ci.yml`.

---

## License

MIT
