import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
import hashlib
import pickle
import numpy as np
import pandas as pd


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        # Create log directory if needed
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded configuration from {config_path}")

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")

def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    logger = logging.getLogger(__name__)
    logger.info(f"Saved data to {filepath}")

def load_json(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data

def save_pickle(data: Any, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data