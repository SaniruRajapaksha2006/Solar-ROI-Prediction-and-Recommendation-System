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
    """
    Comprehensive feature engineering for consumption forecasting
    """

    def __init__(self, config: Dict):
        """
        Initialize feature engineer

        Args:
            config: Configuration dictionary
        """
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
        """Create features from user profile"""
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