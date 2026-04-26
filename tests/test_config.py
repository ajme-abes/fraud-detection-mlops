"""
Tests for src/config/config_loader.py

Verifies the loader, schema, value types, caching, and error handling.
No real config file is modified — all tests use tmp_path fixtures.
"""
import os
import pytest
import yaml
from src.config.config_loader import get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: str, data: dict) -> str:
    """Write a dict as YAML to path and return the path."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


MINIMAL_CONFIG = {
    "paths": {
        "raw_data": "data/raw/creditcard.csv",
        "processed_dir": "data/processed",
        "training_features": "data/training_features.csv",
        "models_dir": "models",
        "logs_dir": "logs",
        "predictions_file": "logs/predictions.jsonl",
        "feedback_file": "logs/feedback.jsonl",
        "shap_plot": "shap_summary.png",
        "drift_report": "monitoring/drift_report.html",
    },
    "model": {
        "default_version": "v1",
        "prediction_threshold": 0.7,
        "min_feedback_samples": 5,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
    },
    "mlflow": {
        "experiment_v1": "Fraud_Detection_v1",
        "experiment_v2": "Fraud_Detection_v2",
    },
    "preprocessing": {
        "test_size": 0.2,
        "random_state": 42,
        "scale_columns": ["Amount", "Time"],
    },
    "monitoring": {
        "drift_schedule_minutes": 5,
        "shap_sample_size": 100,
    },
}


# ---------------------------------------------------------------------------
# Tests — real config.yaml
# ---------------------------------------------------------------------------

class TestRealConfig:
    """Smoke tests against the actual config.yaml in the repo."""

    def test_loads_without_error(self):
        cfg = get_config()
        assert cfg is not None

    def test_returns_dict(self):
        cfg = get_config()
        assert isinstance(cfg, dict)

    def test_has_required_top_level_keys(self):
        cfg = get_config()
        for key in ("paths", "model", "xgboost", "mlflow", "preprocessing", "monitoring"):
            assert key in cfg, f"Missing top-level key: {key}"

    # paths section
    def test_paths_section_complete(self):
        paths = get_config()["paths"]
        for key in ("raw_data", "processed_dir", "training_features",
                    "models_dir", "logs_dir", "predictions_file",
                    "feedback_file", "shap_plot", "drift_report"):
            assert key in paths, f"Missing paths key: {key}"

    # model section
    def test_threshold_is_float_between_0_and_1(self):
        threshold = get_config()["model"]["prediction_threshold"]
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0

    def test_min_feedback_samples_is_positive_int(self):
        val = get_config()["model"]["min_feedback_samples"]
        assert isinstance(val, int)
        assert val > 0

    def test_default_version_is_string(self):
        val = get_config()["model"]["default_version"]
        assert isinstance(val, str)
        assert len(val) > 0

    # xgboost section
    def test_xgboost_n_estimators_is_positive_int(self):
        val = get_config()["xgboost"]["n_estimators"]
        assert isinstance(val, int)
        assert val > 0

    def test_xgboost_learning_rate_is_positive_float(self):
        val = get_config()["xgboost"]["learning_rate"]
        assert isinstance(val, float)
        assert val > 0

    def test_xgboost_max_depth_is_positive_int(self):
        val = get_config()["xgboost"]["max_depth"]
        assert isinstance(val, int)
        assert val > 0

    # preprocessing section
    def test_test_size_between_0_and_1(self):
        val = get_config()["preprocessing"]["test_size"]
        assert 0.0 < val < 1.0

    def test_scale_columns_is_list(self):
        val = get_config()["preprocessing"]["scale_columns"]
        assert isinstance(val, list)
        assert len(val) > 0

    # monitoring section
    def test_shap_sample_size_is_positive_int(self):
        val = get_config()["monitoring"]["shap_sample_size"]
        assert isinstance(val, int)
        assert val > 0


# ---------------------------------------------------------------------------
# Tests — loader behaviour with custom paths
# ---------------------------------------------------------------------------

class TestConfigLoader:

    def test_loads_custom_config_file(self, tmp_path):
        path = _write_config(str(tmp_path / "cfg.yaml"), MINIMAL_CONFIG)
        # bypass lru_cache by calling with explicit path
        cfg = get_config(path)
        assert cfg["model"]["prediction_threshold"] == 0.7

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        from src.config.config_loader import get_config
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            get_config(str(tmp_path / "nonexistent.yaml"))

    def test_raises_value_error_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="Config file is empty"):
            get_config(str(empty))

    def test_caching_returns_same_object(self):
        """Two calls with the same path must return the identical dict object."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_custom_path_returns_correct_values(self, tmp_path):
        custom = {**MINIMAL_CONFIG}
        custom["model"] = {**MINIMAL_CONFIG["model"], "prediction_threshold": 0.85}
        path = _write_config(str(tmp_path / "custom.yaml"), custom)
        cfg = get_config(path)
        assert cfg["model"]["prediction_threshold"] == 0.85

    def test_xgboost_params_match_expected_values(self, tmp_path):
        path = _write_config(str(tmp_path / "cfg.yaml"), MINIMAL_CONFIG)
        cfg = get_config(path)
        assert cfg["xgboost"]["n_estimators"] == 300
        assert cfg["xgboost"]["max_depth"] == 8
        assert cfg["xgboost"]["learning_rate"] == 0.05

    def test_scale_columns_contains_amount_and_time(self, tmp_path):
        path = _write_config(str(tmp_path / "cfg.yaml"), MINIMAL_CONFIG)
        cfg = get_config(path)
        assert "Amount" in cfg["preprocessing"]["scale_columns"]
        assert "Time" in cfg["preprocessing"]["scale_columns"]
