"""
Centralised config loader.

Usage anywhere in the project:
    from src.config.config_loader import get_config
    cfg = get_config()
    threshold = cfg["model"]["prediction_threshold"]
"""
import os
import yaml
from functools import lru_cache

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


@lru_cache(maxsize=1)
def get_config(path: str = CONFIG_PATH) -> dict:
    """
    Load and return the project config as a plain dict.

    Result is cached after the first call so the file is only
    read once per process. Pass a different path in tests to
    override with a fixture config.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config
