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

    def create_all_features(self, user_data: Dict,
                            similar_households: List,
                            df: pd.DataFrame) -> Dict:
        # Create all features for forecasting
        features = {}

        # 1. User features
        features['user'] = self._create_user_features(user_data)

        # 2. Temporal features
        features['temporal'] = self._create_temporal_features(user_data)

        # 3. Historical features from similar households
        features['historical'] = self._create_historical_features(
            similar_households, user_data, df
        )

        # 4. Weather features
        if self.weather_integrator:
            features['weather'] = self.weather_integrator.get_features(
                user_data['latitude'],
                user_data['longitude'],
                user_data.get('consumption_months', {})
            )

        # 5. Statistical features
        features['statistical'] = self._create_statistical_features(
            user_data, similar_households
        )

        # 6. LSTM input (now uses the new method)
        features['lstm_input'] = self._prepare_lstm_input(
            features, user_data, similar_households, df
        )

        return features

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

    def _create_historical_features(self, similar_households: List,
                                    user_data: Dict,
                                    df: pd.DataFrame) -> Dict:
        # Create features from historical data of similar households
        if not similar_households:
            return {}

        accounts = [acc for acc, _ in similar_households[:5]]  # Top 5
        user_months = user_data.get('consumption_months', {})

        features = {}
        month_values = {m: [] for m in range(1, 13)}

        # Collect historical data
        for account in accounts:
            account_data = df[df['ACCOUNT_NO'] == account]
            for month in user_months.keys():
                month_data = account_data[account_data['MONTH'] == month]
                if not month_data.empty:
                    month_values[month].extend(month_data['NET_CONSUMPTION_kWh'].tolist())

        # Calculate statistics for each month
        for month in user_months.keys():
            if month_values[month]:
                values = month_values[month]
                features[f'hist_month_{month}_mean'] = np.mean(values)
                features[f'hist_month_{month}_median'] = np.median(values)
                features[f'hist_month_{month}_std'] = np.std(values)
                features[f'hist_month_{month}_min'] = np.min(values)
                features[f'hist_month_{month}_max'] = np.max(values)

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