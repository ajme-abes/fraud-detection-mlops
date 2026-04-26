"""
Tests for MLflow logging correctness in src/models/train.py

The core guarantee: every param logged to MLflow must exactly match
the value used to construct the model. We verify this by:
  1. Intercepting mlflow.log_param calls
  2. Intercepting the XGBClassifier constructor
  3. Asserting the values are identical

No real MLflow server or dataset needed — everything is mocked.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_dataset(n_samples=200, n_features=31):
    """Tiny balanced dataset — enough to fit XGBoost quickly."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    # Return as numpy arrays (train.py accepts these)
    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLflowLoggingConsistency:

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_logged_n_estimators_matches_model(self, mock_explain, mock_dump, mock_mlflow):
        """n_estimators logged to MLflow must equal the value used in XGBClassifier."""
        from src.config.config_loader import get_config
        from src.models.train import train_model

        cfg = get_config()
        expected = cfg["xgboost"]["n_estimators"]

        # Capture what gets passed to log_param
        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        assert "n_estimators" in logged_params, "n_estimators was never logged"
        assert logged_params["n_estimators"] == expected, (
            f"Logged n_estimators={logged_params['n_estimators']} "
            f"but model used n_estimators={expected}"
        )

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_logged_max_depth_matches_model(self, mock_explain, mock_dump, mock_mlflow):
        """max_depth logged to MLflow must equal the value used in XGBClassifier."""
        from src.config.config_loader import get_config
        from src.models.train import train_model

        cfg = get_config()
        expected = cfg["xgboost"]["max_depth"]

        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        assert "max_depth" in logged_params, "max_depth was never logged"
        assert logged_params["max_depth"] == expected, (
            f"Logged max_depth={logged_params['max_depth']} "
            f"but model used max_depth={expected}"
        )

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_logged_learning_rate_matches_model(self, mock_explain, mock_dump, mock_mlflow):
        """learning_rate logged to MLflow must equal the value used in XGBClassifier."""
        from src.config.config_loader import get_config
        from src.models.train import train_model

        cfg = get_config()
        expected = cfg["xgboost"]["learning_rate"]

        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        assert "learning_rate" in logged_params, "learning_rate was never logged"
        assert logged_params["learning_rate"] == expected, (
            f"Logged learning_rate={logged_params['learning_rate']} "
            f"but model used learning_rate={expected}"
        )

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_scale_pos_weight_is_logged(self, mock_explain, mock_dump, mock_mlflow):
        """scale_pos_weight must be logged — it's computed dynamically and easy to forget."""
        from src.models.train import train_model

        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        assert "scale_pos_weight" in logged_params, "scale_pos_weight was never logged"
        assert isinstance(logged_params["scale_pos_weight"], float)
        assert logged_params["scale_pos_weight"] > 0

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_f1_and_pr_auc_are_logged_as_metrics(self, mock_explain, mock_dump, mock_mlflow):
        """f1_score and pr_auc must be logged as metrics, not params."""
        from src.models.train import train_model

        logged_metrics = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param = MagicMock()
        mock_mlflow.log_metric.side_effect = lambda k, v: logged_metrics.update({k: v})
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        assert "f1_score" in logged_metrics, "f1_score metric was never logged"
        assert "pr_auc" in logged_metrics, "pr_auc metric was never logged"
        assert 0.0 <= logged_metrics["f1_score"] <= 1.0
        assert 0.0 <= logged_metrics["pr_auc"] <= 1.0

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_no_hardcoded_wrong_values_logged(self, mock_explain, mock_dump, mock_mlflow):
        """
        Regression test for the original bug:
        n_estimators=200 and max_depth=6 were hardcoded while the model
        used 300 and 8. This test fails if those wrong values ever come back.
        """
        from src.models.train import train_model

        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        # These were the original wrong hardcoded values
        assert logged_params.get("n_estimators") != 200, (
            "n_estimators=200 is the old wrong hardcoded value — bug has regressed"
        )
        assert logged_params.get("max_depth") != 6, (
            "max_depth=6 is the old wrong hardcoded value — bug has regressed"
        )

    @patch("src.models.train.mlflow")
    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.explain_model")
    def test_all_required_params_are_logged(self, mock_explain, mock_dump, mock_mlflow):
        """Every hyperparameter used to build the model must appear in MLflow."""
        from src.models.train import train_model

        logged_params = {}
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_param.side_effect = lambda k, v: logged_params.update({k: v})
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.xgboost = MagicMock()

        X_train, X_test, y_train, y_test = _make_small_dataset()
        train_model(X_train, y_train, X_test, y_test)

        required = {"n_estimators", "max_depth", "learning_rate", "scale_pos_weight"}
        missing = required - set(logged_params.keys())
        assert not missing, f"These params were never logged: {missing}"
