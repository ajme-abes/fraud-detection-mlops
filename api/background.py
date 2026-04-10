from apscheduler.schedulers.background import BackgroundScheduler
import os
import importlib.util

LOG_DIR = "logs"
DATA_DIR = "data"

PREDICTIONS_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "training_features.csv")


def run_drift_job():
    print("Running scheduled drift check...")

    drift_path = os.path.join("monitoring", "drift.py")

    if not os.path.exists(drift_path):
        print("Drift file not found")
        return

    spec = importlib.util.spec_from_file_location("drift", drift_path)
    drift_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift_module)

    if not os.path.exists(PREDICTIONS_FILE) or not os.path.exists(TRAIN_DATA_FILE):
        print("Missing data for drift")
        return

    try:
        drift_module.run_drift_check(TRAIN_DATA_FILE, PREDICTIONS_FILE)
        print("Drift check completed")
    except Exception as e:
        print(f"Drift error: {e}")


def start_scheduler():
    scheduler = BackgroundScheduler()
    
    # Run every 5 minutes (change later)
    scheduler.add_job(run_drift_job, 'interval', minutes=5)
    
    scheduler.start()
    print("Scheduler started...")