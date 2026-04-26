import os
import joblib
import numpy as np
from typing import Tuple

# Default model path — can be overridden by passing model_path explicitly
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "v1", "model.pkl"
)

DEFAULT_THRESHOLD = 0.7


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the .pkl model file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model cannot be loaded.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    return model


def predict(
    features: list,
    model,
    threshold: float = DEFAULT_THRESHOLD
) -> Tuple[int, float]:
    """
    Run fraud prediction on a single transaction.

    Args:
        features: List of 31 floats (Time, V1–V28, Amount, Class).
        model:    Trained model with a predict_proba method.
        threshold: Probability cutoff above which a transaction is flagged as fraud.

    Returns:
        Tuple of (prediction, probability):
            - prediction: 1 (fraud) or 0 (legitimate)
            - probability: float between 0 and 1

    Raises:
        ValueError: If features length is not 31.
    """
    if len(features) != 31:
        raise ValueError(f"Expected 31 features, got {len(features)}")

    data = np.array(features, dtype=float).reshape(1, -1)
    probability = float(model.predict_proba(data)[0][1])
    prediction = 1 if probability > threshold else 0

    return prediction, round(probability, 4)


def predict_batch(
    features_list: list,
    model,
    threshold: float = DEFAULT_THRESHOLD
) -> list:
    """
    Run fraud prediction on a batch of transactions.

    Args:
        features_list: List of feature lists, each with 31 floats.
        model:         Trained model with a predict_proba method.
        threshold:     Probability cutoff for fraud classification.

    Returns:
        List of dicts with keys 'prediction' and 'probability'.

    Raises:
        ValueError: If any row does not have exactly 31 features.
    """
    results = []
    for i, features in enumerate(features_list):
        if len(features) != 31:
            raise ValueError(f"Row {i}: expected 31 features, got {len(features)}")

        data = np.array(features, dtype=float).reshape(1, -1)
        probability = float(model.predict_proba(data)[0][1])
        prediction = 1 if probability > threshold else 0

        results.append({
            "prediction": prediction,
            "probability": round(probability, 4)
        })

    return results
