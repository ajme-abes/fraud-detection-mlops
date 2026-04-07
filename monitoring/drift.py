import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset

def run_drift_check(reference_csv_path, logs_jsonl_path):
    # 1. Load Data
    ref = pd.read_csv(reference_csv_path)
    
    with open(logs_jsonl_path, 'r') as f:
        # Extract features from the log entry
        log_data = [json.loads(line)['features'] for line in f]
    
    curr = pd.DataFrame(log_data, columns=ref.columns)

    # 2. Setup Report
    report = Report(metrics=[DataDriftPreset()])
    
    # 3. Capture the RUN result (The Fix)
    result = report.run(reference_data=ref, current_data=curr)

    # 4. Save using the result object
    # Saving to 'monitoring/' so the FastAPI mount can find it
    result.save_html("monitoring/drift_report.html")
