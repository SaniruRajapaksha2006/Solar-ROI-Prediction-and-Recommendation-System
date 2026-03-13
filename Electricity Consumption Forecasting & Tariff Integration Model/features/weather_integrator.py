import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class WeatherIntegrator:
    """
    Fetches and integrates weather data from NASA POWER API
    Includes caching to avoid redundant API calls
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path("cache/weather")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = config['features']['weather'].get('cache_days', 30)
        self.source = config['features']['weather'].get('source', 'NASA_POWER')

        # NASA POWER API endpoint
        self.base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

        # Weather parameters we care about
        self.parameters = {
            'temperature': 'T2M',           # Temperature at 2 Meters
            'humidity': 'RH2M',              # Relative Humidity at 2 Meters
            'rainfall': 'PRECTOTCORR',       # Precipitation Corrected
            'cloud_cover': 'CLRSKY_SFC_SW_DWN', # Clear Sky Surface Shortwave Downward Irradiance
            'solar_irradiance': 'ALLSKY_SFC_SW_DWN'  # All Sky Surface Shortwave Downward Irradiance
        }

        # In-memory cache
        self._cache = {}

        logger.info(f"Weather integrator initialized with source: {self.source}")

    def _create_cache_key(self, lat: float, lon: float, year: int) -> str:
        # Create cache key from location and year
        key_str = f"{lat:.4f}_{lon:.4f}_{year}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_file_cache(self, cache_key: str) -> Optional[Dict]:
        # Load weather data from file cache
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                # Check if cache is still valid
                mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age_days = (datetime.now() - mod_time).days

                if age_days <= self.cache_days:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                else:
                    logger.debug(f"Cache expired for {cache_key}")
                    cache_file.unlink()  # Delete expired cache
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")

        return None

    def _save_to_file_cache(self, cache_key: str, data: Dict):
        # Save weather data to file cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
