import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix


def train_model(X_train, y_train):
    """
    Train XGBoost model
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,  # handle imbalance
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


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
