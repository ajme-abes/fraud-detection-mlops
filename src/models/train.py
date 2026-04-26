import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc
)
import shap
import matplotlib.pyplot as plt
import joblib
import os

from src.config.config_loader import get_config

os.makedirs("models", exist_ok=True)


def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with MLflow tracking.
    Hyperparameters and experiment name are read from config.yaml.
    """
    cfg = get_config()
    xgb_cfg = cfg["xgboost"]
    mlflow_cfg = cfg["mlflow"]
    paths_cfg = cfg["paths"]

    # Handle class imbalance dynamically
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        eval_metric=xgb_cfg["eval_metric"],
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(paths_cfg["models_dir"], "v1", "model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print("\n📊 Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")

    # Log to MLflow — optional, never blocks training
    try:
        mlflow.set_experiment(mlflow_cfg["experiment_v1"])
        with mlflow.start_run():
            mlflow.log_param("n_estimators", xgb_cfg["n_estimators"])
            mlflow.log_param("max_depth", xgb_cfg["max_depth"])
            mlflow.log_param("learning_rate", xgb_cfg["learning_rate"])
            mlflow.log_param("scale_pos_weight", round(float(scale_pos_weight), 4))
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.xgboost.log_model(model, name="model")
            explain_model(model, X_test, paths_cfg["shap_plot"])
            print("📈 MLflow run logged successfully.")
    except Exception as e:
        print(f"⚠️  MLflow logging skipped (non-fatal): {e}")

    return model


def explain_model(model, X_sample, shap_plot_path: str = "shap_summary.png"):
    """
    Generate and save a SHAP summary plot, then log it as an MLflow artifact.
    """
    cfg = get_config()
    sample_size = cfg["monitoring"]["shap_sample_size"]

    print("\n🔍 Generating SHAP explanations...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample[:sample_size])

    plt.figure()
    shap.summary_plot(shap_values, X_sample[:sample_size], show=False)
    plt.savefig(shap_plot_path)
    plt.close()

    mlflow.log_artifact(shap_plot_path)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using the production prediction threshold from config.
    """
    cfg = get_config()
    threshold = cfg["model"]["prediction_threshold"]

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n🚨 Model Evaluation (Threshold: {threshold}):")
    print(classification_report(y_test, y_pred))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
