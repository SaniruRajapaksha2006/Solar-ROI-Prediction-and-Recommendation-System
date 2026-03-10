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
        )
    with open(target, "r") as f:
        return yaml.safe_load(f)


def get_roi_constants() -> dict[str, Any]:
    """
    Used by predict11.py and ROI calculation.

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
    Return the NASA API key -> column name mapping.
    Used by nasa_power.py, data_pipeline.py, handle_missing.py.
    """
    return load_config()["nasa"]["params"]
