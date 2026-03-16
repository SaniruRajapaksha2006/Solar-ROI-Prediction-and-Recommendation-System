from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = None) -> dict:
    """
    Load and cache config.yaml.
    """
    target = Path(path) if path else _CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"config.yaml not found at: {target}\n"
            "Make sure config.yaml is in the project root."
        )
    with open(target, "r") as f:
        return yaml.safe_load(f)


def get_roi_constants() -> dict[str, Any]:
    """
    Return ROI/financial constants as a flat dict.
    Used by predict.py and any ROI calculation.

    Returns:
        {
            "net_plus_tariff_lkr": 37.0,
            "mae_kwh_per_kw":      18.0,
        }
    """
    return load_config()["roi"]


def get_physics_constants() -> dict[str, Any]:
    """
    Return solar physics constants.
    Used by feature_engineering.py and fetch_future_weather.py.

    Returns:
        {
            "performance_ratio": 0.80,
            "temp_coefficient":  -0.005,
            "temp_reference":    25.0,
            "humidity_default":  75.0,
        }
    """
    return load_config()["physics"]


def get_nasa_params() -> dict[str, str]:
    """
    Return the NASA API key → column name mapping.
    Used by nasa_power.py, data_pipeline.py, handle_missing.py.
    """
    return load_config()["nasa"]["params"]


def get_training_config() -> dict[str, Any]:
    """
    Return training/evaluation constants.
    Used by splitter.py, tuner.py, evaluator.py.

    Returns:
        {
            "test_size":         0.2,
            "random_state":      42,
            "n_folds":           5,
            "overfit_threshold": 15.0,
            "cv_leak_threshold": -5.0,
        }
    """
    return load_config()["training"]


def get_c3s_config() -> dict[str, Any]:
    """
    Return C3S SEAS5 API constants.
    Used by fetch_future_weather.py.
    """
    return load_config()["c3s"]


def get_outlier_config() -> dict[str, Any]:
    """Return outlier detection config. Used by outliers.py."""
    return load_config()["outlier_detection"]


def get_similarity_config() -> dict[str, Any]:
    """Return similarity model config. Used by similarity_engine.py."""
    return load_config()["similarity"]


def get_feature_selection_config() -> dict[str, Any]:
    """Return feature selection config. Used by selection.py."""
    return load_config()["feature_selection"]
