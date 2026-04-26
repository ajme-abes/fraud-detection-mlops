import pandas as pd
import joblib
import json
import os
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import f1_score, precision_recall_curve, auc
from datetime import datetime

from src.config.config_loader import get_config

# ---------------------------------------------------------------------------
# Paths — resolved at import time from config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _abs(relative_path: str) -> str:
    return os.path.join(BASE_DIR, relative_path)


def retrain_model() -> bool:
    """
    Retrain the fraud detection model using original training data
    combined with feedback-labeled samples.

    All paths and hyperparameters are read from src/config/config.yaml.

    Returns:
        True  — new model saved to models/v2/model.pkl
        False — retraining skipped (missing data or not enough feedback)
    """
    cfg = get_config()
    paths_cfg = cfg["paths"]
    xgb_cfg = cfg["xgboost"]
    mlflow_cfg = cfg["mlflow"]
    model_cfg = cfg["model"]

    train_data_file = _abs(paths_cfg["training_features"])
    predictions_file = _abs(paths_cfg["predictions_file"])
    feedback_file = _abs(paths_cfg["feedback_file"])
    model_dir = _abs(os.path.join(paths_cfg["models_dir"], "v2"))
    min_samples = model_cfg["min_feedback_samples"]
    label_col = model_cfg.get("label_column", "Class")  # explicit, never positional

    print(f"[{datetime.now()}] Starting automated retraining (v2)...")

    # ------------------------------------------------------------------
    # 1. Load original training data
    # ------------------------------------------------------------------
    if not os.path.exists(train_data_file):
        print("Error: Original training data not found at", train_data_file)
        return False

    df_orig = pd.read_csv(train_data_file)
    original_columns = df_orig.columns.tolist()

    if label_col not in original_columns:
        print(f"Error: label column '{label_col}' not found in training data.")
        print(f"  Available columns: {original_columns}")
        return False

    # ------------------------------------------------------------------
    # 2. Load and validate feedback data
    # ------------------------------------------------------------------
    if not os.path.exists(feedback_file) or not os.path.exists(predictions_file):
        print("Error: No feedback or predictions found.")
        return False

    feedback_rows = _load_feedback_rows(original_columns, predictions_file, feedback_file, label_col)

    if len(feedback_rows) < min_samples:
        print(
            f"Only {len(feedback_rows)} feedback sample(s) found. "
            f"Need at least {min_samples}. Skipping retraining."
        )
        return False

    print(f"Found {len(feedback_rows)} feedback sample(s). Proceeding...")

    # ------------------------------------------------------------------
    # 3. Combine original + feedback data
    # ------------------------------------------------------------------
    df_feedback = pd.DataFrame(feedback_rows, columns=original_columns)
    df_combined = pd.concat([df_orig, df_feedback], ignore_index=True)

    X = df_combined.drop(columns=[label_col])
    y = df_combined[label_col].astype(int)  # ensure integer labels

    # ------------------------------------------------------------------
    # 4. Train XGBoost — same architecture as v1
    # ------------------------------------------------------------------
    scale_pos_weight = (len(y) - y.sum()) / y.sum()

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        eval_metric=xgb_cfg["eval_metric"],
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)

    # ------------------------------------------------------------------
    # 5. Evaluate on the feedback slice
    # ------------------------------------------------------------------
    X_feedback = df_feedback.drop(columns=[label_col])
    y_feedback = df_feedback[label_col]

    y_prob = model.predict_proba(X_feedback)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_feedback, y_pred, zero_division=0)
    precision, recall, _ = precision_recall_curve(y_feedback, y_prob)
    pr_auc = auc(recall, precision)

    print(f"\n📊 Retrain Evaluation (on {len(y_feedback)} feedback samples):")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   PR-AUC   : {pr_auc:.4f}")

    # ------------------------------------------------------------------
    # 6. Save model to disk — always happens, regardless of MLflow
    # ------------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ New model saved to {model_path}")

    # ------------------------------------------------------------------
    # 7. Log to MLflow — optional, never blocks model saving
    # ------------------------------------------------------------------
    try:
        mlflow.set_experiment(mlflow_cfg["experiment_v2"])
        with mlflow.start_run():
            mlflow.log_param("n_estimators", xgb_cfg["n_estimators"])
            mlflow.log_param("max_depth", xgb_cfg["max_depth"])
            mlflow.log_param("learning_rate", xgb_cfg["learning_rate"])
            mlflow.log_param("scale_pos_weight", round(float(scale_pos_weight), 4))
            mlflow.log_param("feedback_samples", len(feedback_rows))
            mlflow.log_param("total_training_samples", len(df_combined))
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.xgboost.log_model(model, name="model")
            print("📈 MLflow run logged successfully.")
    except Exception as e:
        print(f"⚠️  MLflow logging skipped (non-fatal): {e}")

    return True


def _load_feedback_rows(
    original_columns: list,
    predictions_file: str,
    feedback_file: str,
    label_col: str = "Class",
) -> list:
    """
    Match feedback labels to their original prediction features.
    Returns a list of dicts with columns matching original_columns.
    """
    with open(predictions_file, "r") as f:
        pred_features = {
            json.loads(line)["request_id"]: json.loads(line)["features"]
            for line in f
        }

    feature_columns = [c for c in original_columns if c != label_col]

    rows = []
    with open(feedback_file, "r") as f:
        for line in f:
            fb = json.loads(line)
            rid = fb["request_id"]
            if rid not in pred_features:
                continue

            features = pred_features[rid]

            # API logs all features — match against feature_columns length
            if len(features) >= len(feature_columns):
                feature_values = features[:len(feature_columns)]
            else:
                print(f"Skipping {rid}: expected {len(feature_columns)} features, got {len(features)}")
                continue

            row = dict(zip(feature_columns, feature_values))
            row[label_col] = int(fb["actual_label"])
            rows.append(row)

    return rows


if __name__ == "__main__":
    retrain_model()
