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
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("models", exist_ok=True)

def train_model(X_train, y_train, X_test, y_test):
    """
    Train model with MLflow tracking
    """

    # Handle imbalance dynamically
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    mlflow.set_experiment("Fraud_Detection_v1")

    with mlflow.start_run():

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False
        )

        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, "models/model.pkl")

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        f1 = f1_score(y_test, y_pred)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        print("\n📊 Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"F1 Score: {f1}")
        print(f"PR-AUC: {pr_auc}")

        # Log metrics
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # SHAP Explainability
        explain_model(model, X_test)

        return model

def explain_model(model, X_sample):
    """
    Generate SHAP explanations
    """
    print("\n🔍 Generating SHAP explanations...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample[:100])  # sample for speed

    plt.figure()
    shap.summary_plot(shap_values, X_sample[:100], show=False)
    plt.savefig("shap_summary.png")

    mlflow.log_artifact("shap_summary.png")
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model
    """
    # Get probabilities instead of just 0 or 1
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # AGGRESSIVE THRESHOLD: set to 0.1
    threshold = 0.1
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n🚨 Model Evaluation (Aggressive Threshold {threshold}):")
    print(classification_report(y_test, y_pred))
    
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
