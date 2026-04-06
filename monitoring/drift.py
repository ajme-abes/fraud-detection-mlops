import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset


def run_drift_check(reference_csv_path, logs_jsonl_path):
    # 1. Load Training Data
    ref = pd.read_csv(reference_csv_path)

    # 2. Load API Logs
    with open(logs_jsonl_path, 'r') as f:
        log_data = [json.loads(line)['features'] for line in f]
    
    # Create DataFrame (Ensure column names match the Reference data!)
    curr = pd.DataFrame(log_data, columns=ref.columns)

    # 3. Run Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    report.save_html("monitoring/drift_report.html")
    print("Drift report generated at monitoring/drift_report.html")
