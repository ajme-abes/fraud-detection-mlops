# Fraud Detection MLOps System

A production-grade fraud detection system built with a full MLOps lifecycle — from data ingestion and model training to real-time serving, monitoring, and automated retraining.

---

## Architecture Overview

```
creditcard.csv
     │
     ▼
Training Pipeline (XGBoost + MLflow)
     │
     ▼
Model Registry (models/v1, v2)
     │
     ▼
FastAPI (predict / feedback / drift / retrain)
     │
     ├──► Streamlit Dashboard (monitoring UI)
     └──► Evidently Drift Reports (scheduled + on-demand)
```

---

## Features

- **Real-time prediction API** — POST transactions, get fraud probability + prediction
- **Feedback loop** — submit actual labels to measure live accuracy
- **Data drift monitoring** — Evidently-powered drift reports, auto-scheduled every 5 minutes
- **Automated retraining** — trigger model v2 training from the API using accumulated feedback
- **Experiment tracking** — MLflow logs params, metrics, and SHAP artifacts per run
- **SHAP explainability** — feature importance summary generated at training time
- **Streamlit dashboard** — visualize predictions, fraud rate, accuracy, and drift
- **Dockerized** — full multi-service setup via docker-compose

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Drift Monitoring | Evidently |
| Scheduling | APScheduler |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
fraud-detection-mlops/
│
├── api/
│   ├── main.py              # FastAPI app (predict, feedback, drift, retrain)
│   ├── background.py        # APScheduler for automated drift checks
│   ├── train_v2.py          # Retraining logic using feedback data
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py               # Streamlit monitoring dashboard
│   └── Dockerfile
│
├── src/
│   ├── data/
│   │   ├── load_data.py     # Data loading and validation
│   │   └── preprocess.py    # Scaling, splitting, saving
│   ├── features/
│   │   └── build_features.py  # Feature engineering
│   ├── models/
│   │   ├── train.py         # XGBoost training with MLflow tracking
│   │   └── predict.py       # Prediction utilities
│   ├── pipelines/
│   │   └── training_pipeline.py  # End-to-end training pipeline
│   └── config/
│       └── config.yaml      # Centralized configuration
│
├── models/
│   ├── v1/model.pkl         # Initial trained model
│   └── v2/model.pkl         # Retrained model (generated after retraining)
│
├── monitoring/
│   ├── drift.py             # Evidently drift check logic
│   └── drift_report.html    # Generated drift report
│
├── data/
│   ├── raw/creditcard.csv   # Source dataset (not committed)
│   └── processed/           # Train/test splits (generated)
│
├── logs/
│   ├── predictions.jsonl    # All prediction logs
│   └── feedback.jsonl       # User-submitted feedback
│
├── tests/
│   └── test_pipeline.py
│
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- The [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) placed at `data/raw/creditcard.csv`

### 1. Clone the repo

```bash
git clone https://github.com/ajme-abes/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline

```bash
python -m src.pipelines.training_pipeline
```

This will:
- Load and preprocess the raw data
- Engineer features
- Train an XGBoost model with MLflow tracking
- Save the model to `models/v1/model.pkl`
- Generate a SHAP summary plot

### 4. Start the services with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API (FastAPI) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Dashboard (Streamlit) | http://localhost:8501 |

---

## API Reference

### `POST /predict`

Submit a transaction for fraud scoring.

```json
{
  "features": [0.1, -1.2, 0.5, ..., 149.62]
}
```

**Note:** Expects exactly 31 float values (Time, V1–V28, Amount, Class).

**Response:**
```json
{
  "request_id": "uuid",
  "prediction": 0,
  "probability": 0.032
}
```

---

### `POST /feedback`

Submit the actual label for a past prediction to track live accuracy.

```json
{
  "request_id": "uuid-from-predict-response",
  "actual_label": 1
}
```

---

### `GET /monitoring/accuracy`

Returns accuracy computed from all feedback submitted so far.

```json
{
  "accuracy": "94.50%",
  "samples": 40
}
```

---

### `POST /monitoring/run-drift`

Triggers an Evidently drift report comparing training data vs. recent predictions.
Report is saved and accessible at `/reports/drift_report.html`.

---

### `POST /monitoring/retrain`

Triggers model retraining using original training data + feedback-labeled samples.
Requires at least 5 feedback samples. Saves the new model to `models/v2/`.

---

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## MLflow Tracking

MLflow is used to track every training run. To launch the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 to view experiments, metrics, and SHAP artifacts.

---

## Running Tests

```bash
pytest tests/
```

---

## CI/CD

GitHub Actions runs on every push to `main`:
- Sets up Python 3.11
- Installs all dependencies
- Runs the test suite

See `.github/workflows/ci.yml` for the full pipeline definition.

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- 284,807 transactions
- 492 fraud cases (~0.17% — highly imbalanced)
- Features V1–V28 are PCA-transformed for confidentiality

The raw CSV is not committed to this repo. Download it from Kaggle and place it at `data/raw/creditcard.csv`.

---

## License

MIT
