"""
Tests for api/train_v2.py

All tests are self-contained — no real dataset or model file needed.
We build minimal CSV / JSONL fixtures in tmp_path and patch the config
so retrain_model() uses tmp_path locations instead of real project paths.
"""
import json
import os
import uuid
import joblib
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [f"V{i}" for i in range(1, 31)] + ["Amount"]
LABEL_COL = "Class"
ALL_COLS = FEATURE_COLS + [LABEL_COL]
N_FEATURES = len(FEATURE_COLS)   # 31


def _make_training_csv(path: str, n_rows: int = 50) -> None:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, N_FEATURES))
    labels = np.array([0] * (n_rows // 2) + [1] * (n_rows // 2))
    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df[LABEL_COL] = labels
    df.to_csv(path, index=False)


def _make_logs(pred_path: str, feedback_path: str, n: int = 10) -> list:
    rng = np.random.default_rng(0)
    request_ids = []
    with open(pred_path, "w") as pf, open(feedback_path, "w") as ff:
        for i in range(n):
            rid = str(uuid.uuid4())
            request_ids.append(rid)
            features = rng.standard_normal(N_FEATURES + 1).tolist()
            pf.write(json.dumps({"request_id": rid, "features": features,
                                  "prediction": 0, "probability": 0.1}) + "\n")
            ff.write(json.dumps({"request_id": rid, "actual_label": i % 2}) + "\n")
    return request_ids


def _patched_config(tmp_path, min_feedback=5):
    """Return a config dict pointing all paths at tmp_path."""
    return {
        "paths": {
            "training_features": str(tmp_path / "data" / "training_features.csv"),
            "predictions_file": str(tmp_path / "logs" / "predictions.jsonl"),
            "feedback_file": str(tmp_path / "logs" / "feedback.jsonl"),
            "models_dir": str(tmp_path / "models"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "model": {
            "default_version": "v1",
            "prediction_threshold": 0.7,
            "min_feedback_samples": min_feedback,
        },
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        },
        "mlflow": {
            "experiment_v1": "test_v1",
            "experiment_v2": "test_v2",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    """
    Set up a self-contained file environment and patch get_config + BASE_DIR
    so retrain_model() resolves all paths to tmp_path.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()

    cfg = _patched_config(tmp_path)

    import api.train_v2 as tv2
    monkeypatch.setattr(tv2, "get_config", lambda: cfg)
    monkeypatch.setattr(tv2, "BASE_DIR", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI",
                       f"sqlite:///{tmp_path}/mlflow_test.db")

    return {
        "train_csv": cfg["paths"]["training_features"],
        "pred_jsonl": cfg["paths"]["predictions_file"],
        "fb_jsonl": cfg["paths"]["feedback_file"],
        "model_dir": str(tmp_path / "models" / "v2"),
        "tmp_path": tmp_path,
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrainModel:

    def test_returns_false_when_training_csv_missing(self, tmp_env):
        from api.train_v2 import retrain_model
        result = retrain_model()
        assert result is False

    def test_returns_false_when_feedback_files_missing(self, tmp_env):
        from api.train_v2 import retrain_model
        _make_training_csv(tmp_env["train_csv"])
        result = retrain_model()
        assert result is False

    def test_returns_false_when_not_enough_feedback(self, tmp_env, monkeypatch):
        from api.train_v2 import retrain_model
        import api.train_v2 as tv2

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=3)

        cfg = _patched_config(tmp_env["tmp_path"], min_feedback=5)
        monkeypatch.setattr(tv2, "get_config", lambda: cfg)

        result = retrain_model()
        assert result is False

    def test_returns_true_and_saves_model_with_enough_feedback(self, tmp_env):
        from api.train_v2 import retrain_model

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=10)

        result = retrain_model()

        assert result is True
        model_path = os.path.join(tmp_env["model_dir"], "model.pkl")
        assert os.path.exists(model_path), "model.pkl was not saved"

    def test_saved_model_is_xgboost(self, tmp_env):
        import xgboost as xgb
        from api.train_v2 import retrain_model

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=10)
        retrain_model()

        model = joblib.load(os.path.join(tmp_env["model_dir"], "model.pkl"))
        assert isinstance(model, xgb.XGBClassifier), (
            f"Expected XGBClassifier, got {type(model).__name__}"
        )

    def test_saved_model_can_predict(self, tmp_env):
        from api.train_v2 import retrain_model

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=10)
        retrain_model()

        model = joblib.load(os.path.join(tmp_env["model_dir"], "model.pkl"))
        sample = np.zeros((1, N_FEATURES))
        proba = model.predict_proba(sample)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0][1] <= 1.0

    def test_min_feedback_threshold_respected(self, tmp_env, monkeypatch):
        """Exactly min_feedback_samples should be enough to proceed."""
        from api.train_v2 import retrain_model
        import api.train_v2 as tv2

        cfg = _patched_config(tmp_env["tmp_path"], min_feedback=5)
        monkeypatch.setattr(tv2, "get_config", lambda: cfg)

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=5)

        result = retrain_model()
        assert result is True

    def test_unmatched_request_ids_are_skipped(self, tmp_env):
        """Feedback entries with no matching prediction should be ignored gracefully."""
        from api.train_v2 import retrain_model

        _make_training_csv(tmp_env["train_csv"])
        _make_logs(tmp_env["pred_jsonl"], tmp_env["fb_jsonl"], n=10)

        with open(tmp_env["fb_jsonl"], "a") as f:
            for _ in range(3):
                f.write(json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "actual_label": 1
                }) + "\n")

        result = retrain_model()
        assert result is True
