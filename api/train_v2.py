import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier # Or your specific model
from datetime import datetime

# Paths - Ensure these match your main.py setup
DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "models/v2" # Saving to v2 folder
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "training_features.csv")
PREDICTIONS_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.jsonl")

def retrain_model():
    print(f"[{datetime.now()}] Starting automated retraining...")
    
    # 1. Load Original Training Data
    if not os.path.exists(TRAIN_DATA_FILE):
        print("Error: Original training data not found.")
        return False
   
    df_orig = pd.read_csv(TRAIN_DATA_FILE)
    
    # 2. Load New Feedback Data
    if not os.path.exists(FEEDBACK_FILE) or not os.path.exists(PREDICTIONS_FILE):
        print("Error: No feedback or predictions found to retrain with.")
        return False

    # Load predictions into a dict for quick lookup
    with open(PREDICTIONS_FILE, "r") as f:
        preds = {json.loads(line)["request_id"]: json.loads(line)["features"] for line in f}
    
    # Match feedback with their original features
    new_data = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            fb = json.loads(line)
            rid = fb["request_id"]
            if rid in preds:
                features = list(preds[rid])
                # REPLACE the last element (the "cheating" feature) with the actual label
                features[-1] = fb["actual_label"] 
                new_data.append(features)
    
    if len(new_data) < 5: # Threshold: Don't retrain with less than 5 new samples
        print(f"Only {len(new_data)} new samples found. Skipping retraining.")
        return False

    # 3. Combine Data
    original_columns = df_orig.columns.tolist() 

    df_new = pd.DataFrame(new_data, columns=original_columns)
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    
    # 4. Train New Model
    X = df_combined.iloc[:, :-1] # All columns except last
    y = df_combined.iloc[:, -1]  # Last column (target)
    
    # Use same parameters as your original model
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X, y)
    
    # 5. Save Version 2
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(new_model, os.path.join(MODEL_DIR, "model.pkl"))
    
    print(f"Success! New model version saved to {MODEL_DIR}")
    return True

if __name__ == "__main__":
    retrain_model()
