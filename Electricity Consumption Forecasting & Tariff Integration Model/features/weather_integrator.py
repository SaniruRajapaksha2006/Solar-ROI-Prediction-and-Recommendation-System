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

    def get_features(self, lat: float, lon: float,
                    months: Dict[int, float]) -> Dict:
        # Get weather features for location and months

        if not months:
            return {}

        # Get year from config or use current
        year = 2025  # Your data year

        # Create cache key
        cache_key = self._create_cache_key(lat, lon, year)

        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Using cached weather data for {lat}, {lon}")
            return self._cache[cache_key]

        # Try to load from file cache
        cached_data = self._load_from_file_cache(cache_key)
        if cached_data is not None:
            self._cache[cache_key] = cached_data
            return cached_data

        # Fetch from API
        weather_data = self._fetch_weather_data(lat, lon, year)

        if weather_data:
            # Process into features
            features = self._process_weather_features(weather_data, months)

            # Cache the result
            self._cache[cache_key] = features
            self._save_to_file_cache(cache_key, features)

            return features
        else:
            logger.warning(f"Could not fetch weather data for {lat}, {lon}")
            return self._get_fallback_features(months)

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

    def _fetch_weather_data(self, lat: float, lon: float, year: int) -> Optional[Dict]:
        # Fetch weather data from NASA POWER API

        try:
            # Prepare parameters
            params = {
                'parameters': ','.join(self.parameters.values()),
                'community': 'RE',
                'longitude': lon,
                'latitude': lat,
                'start': year,
                'end': year,
                'format': 'JSON'
            }

            logger.info(f"Fetching weather data for {lat}, {lon} for year {year}")

            # Make API request with timeout
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30,
                headers={'Accept': 'application/json'}
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched weather data")
                return data
            else:
                logger.warning(f"API returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Connection error to NASA POWER API")
            return None
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def _process_weather_features(self, raw_data: Dict,
                                  user_months: Dict[int, float]) -> Dict:
        # Process raw weather data into features

        features = {}

        try:
            # Extract properties from API response
            properties = raw_data.get('properties', {})
            parameter_data = properties.get('parameter', {})

            # Process each weather parameter
            for feature_name, param_key in self.parameters.items():
                if param_key in parameter_data:
                    monthly_data = parameter_data[param_key]

                    # Create month mapping
                    monthly_values = {}
                    for month in range(1, 13):
                        month_str = f"{month:02d}"
                        if month_str in monthly_data:
                            monthly_values[month] = monthly_data[month_str]

                    # Add statistics
                    features[feature_name] = {
                        'monthly': monthly_values,
                        'mean': np.mean(list(monthly_values.values())),
                        'std': np.std(list(monthly_values.values())),
                        'min': min(monthly_values.values()),
                        'max': max(monthly_values.values())
                    }

                    # Add features for user's months specifically
                    for month in user_months.keys():
                        if month in monthly_values:
                            features[f"{feature_name}_month_{month}"] = monthly_values[month]

            # Calculate derived features
            if 'temperature' in features and 'humidity' in features:
                # Heat index (simplified)
                temp = features['temperature']['monthly']
                hum = features['humidity']['monthly']

                heat_index = {}
                for month in range(1, 13):
                    if month in temp and month in hum:
                        # Simple heat index approximation
                        heat_index[month] = temp[month] + 0.1 * hum[month]

                features['heat_index'] = {
                    'monthly': heat_index,
                    'mean': np.mean(list(heat_index.values())) if heat_index else 0
                }

            # Add metadata
            features['location'] = {
                'lat': raw_data.get('geometry', {}).get('coordinates', [])[1],
                'lon': raw_data.get('geometry', {}).get('coordinates', [])[0]
            }

            features['data_source'] = 'NASA_POWER'
            features['data_year'] = 2025

        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            return self._get_fallback_features(user_months)

        return features

    def _get_fallback_features(self, user_months: Dict[int, float]) -> Dict:
        """
        Generate fallback weather features when API fails
        Uses Sri Lanka average seasonal patterns
        """
        logger.info("Using fallback weather features")

        features = {}

        # Sri Lanka typical monthly patterns (based on climate data)
        # Temperatures in Celsius
        temp_pattern = {
            1: 26.5, 2: 27.0, 3: 28.0, 4: 28.5, 5: 28.5, 6: 28.0,
            7: 27.5, 8: 27.5, 9: 27.5, 10: 27.0, 11: 26.5, 12: 26.0
        }

        # Humidity percentage
        humidity_pattern = {
            1: 75, 2: 70, 3: 70, 4: 75, 5: 80, 6: 80,
            7: 80, 8: 80, 9: 80, 10: 80, 11: 80, 12: 80
        }

        # Rainfall in mm
        rainfall_pattern = {
            1: 100, 2: 80, 3: 120, 4: 200, 5: 300, 6: 200,
            7: 150, 8: 150, 9: 200, 10: 300, 11: 300, 12: 200
        }

        # Create temperature features
        features['temperature'] = {
            'monthly': temp_pattern,
            'mean': np.mean(list(temp_pattern.values())),
            'std': np.std(list(temp_pattern.values()))
        }

        # Create humidity features
        features['humidity'] = {
            'monthly': humidity_pattern,
            'mean': np.mean(list(humidity_pattern.values())),
            'std': np.std(list(humidity_pattern.values()))
        }

        # Create rainfall features
        features['rainfall'] = {
            'monthly': rainfall_pattern,
            'mean': np.mean(list(rainfall_pattern.values())),
            'std': np.std(list(rainfall_pattern.values()))
        }

        # Add features for user's months
        for month in user_months.keys():
            if month in temp_pattern:
                features[f'temperature_month_{month}'] = temp_pattern[month]
            if month in humidity_pattern:
                features[f'humidity_month_{month}'] = humidity_pattern[month]
            if month in rainfall_pattern:
                features[f'rainfall_month_{month}'] = rainfall_pattern[month]

        features['data_source'] = 'fallback_pattern'
        features['is_fallback'] = True

        return features

    def clear_cache(self):
        # Clear all caches
        self._cache.clear()

        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

        logger.info("Weather cache cleared")