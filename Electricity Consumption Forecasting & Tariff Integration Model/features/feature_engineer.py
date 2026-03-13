import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
import sys
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def find_project_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / 'src').exists():
            return current
        current = current.parent
    return Path.cwd()

project_root = find_project_root()
sys.path.insert(0, str(project_root))

from features.cyclical_encoder import CyclicalFeatureEncoder
from features.weather_integrator import WeatherIntegrator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.cyclical_encoder = CyclicalFeatureEncoder(config)
        self.weather_integrator = WeatherIntegrator(config) if config['features']['weather']['enabled'] else None

        # Scalers for different feature types
        self.scalers = {
            'consumption': RobustScaler(),
            'weather': StandardScaler(),
            'count': MinMaxScaler()
        }

        self.is_fitted = False
        self.feature_names = []

    def _create_user_features(self, user_data: Dict) -> Dict:
        # Create features from user profile
        features = {
            'has_solar': user_data.get('has_solar', 0),
            'phase_TP': 1 if user_data.get('phase') == 'TP' else 0,
            'phase_SP': 1 if user_data.get('phase') == 'SP' else 0,
            'household_size': user_data.get('household_size', 4),
            'latitude': user_data['latitude'],
            'longitude': user_data['longitude']
        }

        # One-hot encode tariff
        tariff = user_data.get('tariff', 'D1')
        features[f'tariff_{tariff}'] = 1

        return features

    def _create_temporal_features(self, user_data: Dict) -> Dict:
        # Create temporal features
        user_months = user_data.get('consumption_months', {})

        if not user_months:
            return {}

        months = sorted(user_months.keys())

        # Basic temporal features
        features = {
            'first_month': months[0],
            'last_month': months[-1],
            'n_months': len(months),
            'month_span': months[-1] - months[0] + 1
        }

        # Cyclical encoding for each month
        for month in months:
            cyclic = self.cyclical_encoder.encode_month(month)
            features[f'month_{month}_sin'] = cyclic['month_sin']
            features[f'month_{month}_cos'] = cyclic['month_cos']

        return features

    def _create_statistical_features(self, user_data: Dict,
                                     similar_households: List) -> Dict:
        # Create statistical features from user's consumption
        user_months = user_data.get('consumption_months', {})

        if not user_months:
            return {}

        values = list(user_months.values())
        months = list(user_months.keys())

        features = {
            'user_mean': np.mean(values),
            'user_median': np.median(values),
            'user_std': np.std(values) if len(values) > 1 else 0,
            'user_min': np.min(values),
            'user_max': np.max(values),
            'user_cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
        }

        # Trend if enough data
        if len(values) >= 3:
            from scipy import stats
            slope, _, r_value, _, _ = stats.linregress(months, values)
            features['user_trend_slope'] = slope
            features['user_trend_r2'] = r_value ** 2

        return features