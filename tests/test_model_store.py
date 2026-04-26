"""
Tests for the ModelStore hot-reload mechanism in api/main.py.

No real model file or running server needed — we use tmp_path fixtures
and a DummyClassifier saved as a .pkl to simulate real model files.
"""
import os
import joblib
import pytest
import numpy as np
from sklearn.dummy import DummyClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_dummy_model(path: str) -> None:
    """Save a fitted DummyClassifier to path (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clf = DummyClassifier(strategy="constant", constant=0)
    # DummyClassifier needs to be fitted before predict_proba works
    clf.fit(np.zeros((4, 31)), [0, 0, 1, 1])
    joblib.dump(clf, path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    """Return a fresh ModelStore instance for each test."""
    from api.main import ModelStore
    return ModelStore()


@pytest.fixture()
def model_dir(tmp_path):
    """
    Create a temporary models/ directory with v1 and v2 sub-folders,
    each containing a valid model.pkl.
    """
    v1_path = tmp_path / "models" / "v1" / "model.pkl"
    v2_path = tmp_path / "models" / "v2" / "model.pkl"
    _save_dummy_model(str(v1_path))
    _save_dummy_model(str(v2_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestModelStoreInitialState:

    def test_model_is_none_before_load(self, store):
        assert store.model is None

    def test_is_loaded_false_before_load(self, store):
        assert store.is_loaded is False

    def test_version_is_empty_before_load(self, store):
        assert store.version == ""


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestModelStoreLoad:

    def test_load_returns_true_for_valid_model(self, store, model_dir, monkeypatch):
        monkeypatch.setattr(
            "api.main.BASE_DIR", str(model_dir)
        )
        assert store.load("v1") is True

    def test_load_sets_model_object(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        assert store.model is not None

    def test_load_sets_version_string(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        assert store.version == "v1"

    def test_load_sets_loaded_at_timestamp(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        assert store.loaded_at != ""

    def test_load_returns_false_for_missing_file(self, store, tmp_path, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(tmp_path))
        assert store.load("v99") is False

    def test_load_does_not_change_model_on_failure(self, store, tmp_path, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(tmp_path))
        store.load("v99")
        assert store.model is None

    def test_load_returns_false_for_corrupt_file(self, store, tmp_path, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(tmp_path))
        corrupt_path = tmp_path / "models" / "v1" / "model.pkl"
        os.makedirs(corrupt_path.parent, exist_ok=True)
        corrupt_path.write_text("not a pickle")
        assert store.load("v1") is False

    def test_is_loaded_true_after_successful_load(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        assert store.is_loaded is True


# ---------------------------------------------------------------------------
# reload() — hot-swap behaviour
# ---------------------------------------------------------------------------

class TestModelStoreReload:

    def test_reload_swaps_version(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        assert store.version == "v1"

        store.reload("v2")
        assert store.version == "v2"

    def test_reload_replaces_model_object(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        original_model = store.model

        store.reload("v2")
        # The object in memory should have been replaced
        assert store.model is not original_model

    def test_reload_updates_loaded_at(self, store, model_dir, monkeypatch):
        import time
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        ts_before = store.loaded_at

        time.sleep(0.01)  # ensure timestamp differs
        store.reload("v2")
        assert store.loaded_at != ts_before

    def test_reload_returns_false_for_missing_version(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        result = store.reload("v99")
        assert result is False

    def test_reload_preserves_old_model_on_failure(self, store, model_dir, monkeypatch):
        """If reload fails, the previously loaded model stays active."""
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        original_model = store.model

        store.reload("v99")  # non-existent version
        assert store.model is original_model
        assert store.version == "v1"

    def test_reloaded_model_can_predict(self, store, model_dir, monkeypatch):
        monkeypatch.setattr("api.main.BASE_DIR", str(model_dir))
        store.load("v1")
        store.reload("v2")

        sample = np.zeros((1, 31))
        proba = store.model.predict_proba(sample)
        assert proba.shape == (1, 2)
