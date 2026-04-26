"""
Tests for src/models/predict.py

Uses a lightweight mock model so no real .pkl file is needed.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.models.predict import predict, predict_batch, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_model(fraud_prob: float = 0.9):
    """Return a mock model whose predict_proba always returns fraud_prob."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - fraud_prob, fraud_prob]])
    return mock


VALID_FEATURES = [0.0] * 31  # 31 zeros — valid input shape


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_fraud_when_prob_above_threshold(self):
        model = make_mock_model(fraud_prob=0.95)
        prediction, probability = predict(VALID_FEATURES, model, threshold=0.7)
        assert prediction == 1
        assert probability == 0.95

    def test_returns_legit_when_prob_below_threshold(self):
        model = make_mock_model(fraud_prob=0.3)
        prediction, probability = predict(VALID_FEATURES, model, threshold=0.7)
        assert prediction == 0
        assert probability == 0.3

    def test_boundary_exactly_at_threshold_is_legit(self):
        # prob == threshold should NOT trigger fraud (strictly greater than)
        model = make_mock_model(fraud_prob=0.7)
        prediction, _ = predict(VALID_FEATURES, model, threshold=0.7)
        assert prediction == 0

    def test_probability_is_rounded_to_4_decimals(self):
        model = make_mock_model(fraud_prob=0.123456789)
        _, probability = predict(VALID_FEATURES, model)
        assert probability == round(0.123456789, 4)

    def test_raises_on_wrong_feature_count_too_few(self):
        model = make_mock_model()
        with pytest.raises(ValueError, match="Expected 31 features"):
            predict([0.0] * 10, model)

    def test_raises_on_wrong_feature_count_too_many(self):
        model = make_mock_model()
        with pytest.raises(ValueError, match="Expected 31 features"):
            predict([0.0] * 50, model)

    def test_custom_threshold_low(self):
        """A low threshold means even low-prob transactions are flagged."""
        model = make_mock_model(fraud_prob=0.2)
        prediction, _ = predict(VALID_FEATURES, model, threshold=0.1)
        assert prediction == 1

    def test_custom_threshold_high(self):
        """A very high threshold means even high-prob transactions pass."""
        model = make_mock_model(fraud_prob=0.8)
        prediction, _ = predict(VALID_FEATURES, model, threshold=0.99)
        assert prediction == 0


# ---------------------------------------------------------------------------
# predict_batch()
# ---------------------------------------------------------------------------

class TestPredictBatch:

    def test_returns_correct_number_of_results(self):
        model = make_mock_model(fraud_prob=0.9)
        batch = [VALID_FEATURES] * 5
        results = predict_batch(batch, model)
        assert len(results) == 5

    def test_each_result_has_prediction_and_probability(self):
        model = make_mock_model(fraud_prob=0.9)
        results = predict_batch([VALID_FEATURES], model)
        assert "prediction" in results[0]
        assert "probability" in results[0]

    def test_all_fraud_predictions(self):
        model = make_mock_model(fraud_prob=0.95)
        results = predict_batch([VALID_FEATURES] * 3, model)
        assert all(r["prediction"] == 1 for r in results)

    def test_all_legit_predictions(self):
        model = make_mock_model(fraud_prob=0.1)
        results = predict_batch([VALID_FEATURES] * 3, model)
        assert all(r["prediction"] == 0 for r in results)

    def test_raises_on_bad_row(self):
        model = make_mock_model()
        bad_batch = [VALID_FEATURES, [0.0] * 10]  # second row is wrong
        with pytest.raises(ValueError, match="Row 1"):
            predict_batch(bad_batch, model)

    def test_empty_batch_returns_empty_list(self):
        model = make_mock_model()
        results = predict_batch([], model)
        assert results == []


# ---------------------------------------------------------------------------
# load_model()
# ---------------------------------------------------------------------------

class TestLoadModel:

    def test_raises_file_not_found_for_missing_path(self):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model("/nonexistent/path/model.pkl")

    def test_raises_runtime_error_on_corrupt_file(self, tmp_path):
        # Write a corrupt (non-pickle) file
        corrupt = tmp_path / "model.pkl"
        corrupt.write_text("this is not a valid pickle")
        with pytest.raises(RuntimeError, match="Failed to load model"):
            load_model(str(corrupt))

    def test_loads_valid_model(self, tmp_path):
        import joblib
        from sklearn.dummy import DummyClassifier
        # Use a real sklearn object — MagicMock can't be pickled by joblib
        fake_model = DummyClassifier()
        model_path = tmp_path / "model.pkl"
        joblib.dump(fake_model, str(model_path))

        loaded = load_model(str(model_path))
        assert loaded is not None
