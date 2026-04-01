# 🚀 Fraud Detection MLOps System

A production-grade fraud detection system using:
- XGBoost
- FastAPI
- MLflow
- Docker
- CI/CD (GitHub Actions)

## Features
- Real-time fraud prediction API
- Experiment tracking
- Data drift monitoring
- Scalable deployment

## Project Structure
fraud-detection-mlops/
│
├── data/
│    ├── raw/
│    ├── processed/
│
├── notebooks/
│    └── exploration.ipynb
│
├── src/
│    ├── features/
│    │     └── build_features.py
│    │
│    ├── models/
│    │     ├── train.py
│    │     ├── predict.py
│    │
│    ├── pipelines/
│    │     └── training_pipeline.py
│    │
│    └── config/
│          └── config.yaml
│
├── api/
│    └── main.py   # FastAPI
│
├── tests/
│
├── docker/
│    └── Dockerfile
│
├── .github/
│    └── workflows/
│          └── ci.yml
│
├── requirements.txt
├── README.md
└── mlflow/


## How to Run
Instructions coming soon...