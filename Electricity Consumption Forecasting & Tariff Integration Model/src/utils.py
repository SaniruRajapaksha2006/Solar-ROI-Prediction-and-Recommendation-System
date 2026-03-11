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

def validate_user_input(user_data: Dict) -> Dict:
    cleaned = user_data.copy()

    # Required fields
    required_fields = ['latitude', 'longitude']
    for field in required_fields:
        if field not in cleaned:
            raise ValueError(f"Missing required field: {field}")

    # Convert numeric fields
    numeric_fields = ['latitude', 'longitude', 'has_solar', 'household_size']
    for field in numeric_fields:
        if field in cleaned:
            try:
                cleaned[field] = float(cleaned[field]) if field in ['latitude', 'longitude'] else int(cleaned[field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field}: must be a number")

    # Validate latitude/longitude ranges (Sri Lanka)
    if not (5.9 <= cleaned['latitude'] <= 9.8):
        raise ValueError(f"Latitude {cleaned['latitude']} outside Sri Lanka range (5.9-9.8)")

    if not (79.6 <= cleaned['longitude'] <= 81.9):
        raise ValueError(f"Longitude {cleaned['longitude']} outside Sri Lanka range (79.6-81.9)")

    # Clean consumption months
    if 'consumption_months' in cleaned:
        clean_months = {}
        for month_str, consumption in cleaned['consumption_months'].items():
            try:
                month = int(month_str)
                if not (1 <= month <= 12):
                    raise ValueError(f"Month {month} must be between 1 and 12")

                consumption = float(consumption)
                if consumption < 0 or consumption > 2000:
                    raise ValueError(f"Consumption {consumption} kWh out of range (0-2000)")

                clean_months[month] = consumption
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid consumption data: {e}")

        cleaned['consumption_months'] = clean_months

        # Validate minimum data
        if len(clean_months) < 2:
            raise ValueError(f"At least 2 months of consumption data required, got {len(clean_months)}")

    # Set defaults
    cleaned.setdefault('tariff', 'D1')
    cleaned.setdefault('phase', 'SP')
    cleaned.setdefault('has_solar', 0)
    cleaned.setdefault('household_size', 4)

    # Validate tariff
    valid_tariffs = ['D1', 'GP1', 'GP2']
    if cleaned['tariff'] not in valid_tariffs:
        raise ValueError(f"Invalid tariff: {cleaned['tariff']}. Must be one of {valid_tariffs}")

    # Validate phase
    valid_phases = ['SP', 'TP']
    if cleaned['phase'] not in valid_phases:
        raise ValueError(f"Invalid phase: {cleaned['phase']}. Must be one of {valid_phases}")

    return cleaned

def format_currency(amount: float) -> str:
    #Format currency for display
    if amount >= 1000000:
        return f"Rs. {amount / 1000000:.2f}M"
    elif amount >= 1000:
        return f"Rs. {amount / 1000:.1f}K"
    else:
        return f"Rs. {amount:,.0f}"


def format_consumption(consumption_kwh: float) -> str:
    #Format consumption for display
    if consumption_kwh >= 1000:
        return f"{consumption_kwh / 1000:.1f} MWh"
    else:
        return f"{consumption_kwh:,.0f} kWh"


def get_month_name(month_number: int) -> str:
    #Get month name from number
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    return month_names[month_number - 1] if 1 <= month_number <= 12 else f"Month {month_number}"

def create_timestamp() -> str:
    #Create ISO format timestamp
    return datetime.now().isoformat()


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    #Calculate percentage change
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100


def get_sri_lanka_holiday_months() -> List[int]:
    # Thai Pongal, Avurudu, Vesak, Christmas
    return [1, 4, 5, 12]

def create_results_directory(base_path: str = "results") -> Path:
    #Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def create_cache_key(data: Dict) -> str:
    #Create cache key from dictionary
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode()).hexdigest()


class Timer:
    #Context manager for timing code execution

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger = logging.getLogger(__name__)
        logger.info(f"{self.name} completed in {self.elapsed:.2f} seconds")


def log_execution_time(func):
    #To log function execution time
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executed in {elapsed:.2f} seconds")
        return result
    return wrapper
