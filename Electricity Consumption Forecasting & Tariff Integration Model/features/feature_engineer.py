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

from .cyclical_encoder import CyclicalFeatureEncoder
from .weather_integrator import WeatherIntegrator

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

    def _prepare_lstm_input(self, features: Dict, user_data: Dict,
                            similar_households: List, df: pd.DataFrame) -> Optional[np.ndarray]:
        # Prepare a proper 12-month sequence for LSTM input with exactly 30 features.


        # Get user's months
        user_months = user_data.get('consumption_months', {})
        if not user_months:
            return None

        # Get top similar households
        top_similar = [acc for acc, _ in similar_households[:5]] if similar_households else []
        if not top_similar:
            return None

        sequence = []

        # Get user's average for scaling
        user_avg = np.mean(list(user_months.values())) if user_months else 400

        # User's static features
        has_solar = float(user_data.get('has_solar', 0))
        inv_capacity = float(user_data.get('inv_capacity', 0))
        distance = 0.0  # User's distance to transformer (if available)

        # Phase encoding
        phase = user_data.get('phase', 'SP')
        phase_SP = 1.0 if phase == 'SP' else 0.0
        phase_TP = 1.0 if phase == 'TP' else 0.0

        # For each month (1-12)
        for month in range(1, 13):

            # GET CONSUMPTION FOR THIS MONTH
            if month in user_months:
                consumption = user_months[month]
            else:
                # Get from similar households
                month_values = []
                for account in top_similar:
                    account_data = df[df['ACCOUNT_NO'] == account]
                    month_data = account_data[account_data['MONTH'] == month]
                    if not month_data.empty:
                        month_values.append(month_data['NET_CONSUMPTION_kWh'].iloc[0])

                if month_values:
                    consumption = np.median(month_values)
                    # Scale to match user's average
                    pattern_avg = np.mean(month_values)
                    if pattern_avg > 0:
                        consumption *= (user_avg / pattern_avg)
                else:
                    # Fallback to seasonal average
                    seasonal_factors = self.config['features']['sri_lanka']['seasonal_factors']
                    consumption = 400 * seasonal_factors.get(month, 1.0)

            # calculate lag features
            def get_lag(lag_months):
                if month - lag_months >= 1:
                    if (month - lag_months) in user_months:
                        return user_months[month - lag_months]
                    else:
                        lag_values = []
                        for account in top_similar:
                            account_data = df[df['ACCOUNT_NO'] == account]
                            lag_data = account_data[account_data['MONTH'] == (month - lag_months)]
                            if not lag_data.empty:
                                lag_values.append(lag_data['NET_CONSUMPTION_kWh'].iloc[0])
                        return np.median(lag_values) if lag_values else consumption * 0.9
                return consumption * 0.9

            lag_1 = get_lag(1)
            lag_2 = get_lag(2)
            lag_3 = get_lag(3)
            lag_6 = get_lag(6)
            lag_12 = get_lag(12)

            # calculate rolling statistics
            def get_rolling(window, stat):
                values = []
                for w in range(1, window + 1):
                    past_month = month - w
                    if past_month >= 1:
                        if past_month in user_months:
                            values.append(user_months[past_month])
                        else:
                            for account in top_similar:
                                account_data = df[df['ACCOUNT_NO'] == account]
                                month_data = account_data[account_data['MONTH'] == past_month]
                                if not month_data.empty:
                                    values.append(month_data['NET_CONSUMPTION_kWh'].iloc[0])
                                    break

                if len(values) == 0:
                    return consumption

                if stat == 'mean':
                    return np.mean(values)
                elif stat == 'std':
                    return np.std(values) if len(values) > 1 else consumption * 0.1
                elif stat == 'min':
                    return np.min(values)
                elif stat == 'max':
                    return np.max(values)
                return consumption

            rolling_mean_3 = get_rolling(3, 'mean')
            rolling_std_3 = get_rolling(3, 'std')
            rolling_min_3 = get_rolling(3, 'min')
            rolling_max_3 = get_rolling(3, 'max')

            rolling_mean_6 = get_rolling(6, 'mean')
            rolling_std_6 = get_rolling(6, 'std')
            rolling_min_6 = get_rolling(6, 'min')
            rolling_max_6 = get_rolling(6, 'max')

            rolling_mean_12 = get_rolling(12, 'mean')
            rolling_std_12 = get_rolling(12, 'std')
            rolling_min_12 = get_rolling(12, 'min')
            rolling_max_12 = get_rolling(12, 'max')

            # differenced features
            diff_1 = consumption - lag_1
            diff_12 = consumption - lag_12 if lag_12 > 0 else 0
            pct_change_1 = (diff_1 / lag_1) * 100 if lag_1 > 0 else 0
            pct_change_12 = (diff_12 / lag_12) * 100 if lag_12 > 0 else 0

            # ratio features
            hist_avg = np.mean([lag_1, lag_2, lag_3]) if lag_1 > 0 else consumption
            consumption_vs_avg = consumption / hist_avg if hist_avg > 0 else 1.0
            consumption_vs_rolling_3 = consumption / rolling_mean_3 if rolling_mean_3 > 0 else 1.0
            consumption_vs_rolling_6 = consumption / rolling_mean_6 if rolling_mean_6 > 0 else 1.0
            consumption_vs_rolling_12 = consumption / rolling_mean_12 if rolling_mean_12 > 0 else 1.0

            # Sri Lanka season flags
            is_wet_season = 1 if month in [5, 6, 7, 8, 9] else 0
            is_intermediate = 1 if month in [10, 11] else 0

            # final 30 features
            month_features = [
                is_wet_season,
                is_intermediate,
                lag_1,
                lag_2,
                lag_3,
                lag_6,
                lag_12,
                rolling_mean_3,
                rolling_std_3,
                rolling_min_3,
                rolling_max_3,
                rolling_mean_6,
                rolling_std_6,
                rolling_min_6,
                rolling_max_6,
                rolling_mean_12,
                rolling_std_12,
                rolling_min_12,
                rolling_max_12,
                diff_1,
                diff_12,
                pct_change_1,
                pct_change_12,
                consumption_vs_avg,
                consumption_vs_rolling_3,
                consumption_vs_rolling_6,
                consumption_vs_rolling_12,
                inv_capacity,  # INV_CAPACITY
                distance,  # DISTANCE_FROM_TF_M
                phase_SP  # phase_SP
            ]

            sequence.append(month_features)

        # Convert to numpy array and reshape for LSTM
        X = np.array(sequence).reshape(1, 12, 30)

        print(f"✅ Created LSTM input sequence with shape {X.shape}")
        return X

    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Prepare features for model training

        # Group by account
        X_list = []
        y_list = []

        for account in df['ACCOUNT_NO'].unique():
            account_data = df[df['ACCOUNT_NO'] == account].sort_values(['YEAR', 'MONTH'])

            if len(account_data) < 24:  # Need at least 2 years
                continue

            # Create features
            values = account_data['NET_CONSUMPTION_kWh'].values

            # Create sequences
            lookback = self.config['forecasting']['lstm']['lookback_window']
            horizon = self.config['forecasting']['lstm']['forecast_horizon']

            for i in range(len(values) - lookback - horizon):
                X_seq = values[i:i + lookback]
                y_seq = values[i + lookback:i + lookback + horizon]

                # Add simple features (expand in production)
                X_with_features = np.column_stack([
                    X_seq,
                    np.sin(2 * np.pi * np.arange(lookback) / 12),  # month sin
                    np.cos(2 * np.pi * np.arange(lookback) / 12)  # month cos
                ])

                X_list.append(X_with_features)
                y_list.append(y_seq)

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")

        self.feature_names = [f'feature_{i}' for i in range(X.shape[2])]
        self.is_fitted = True

        return X, y

    def fit_scalers(self, X_train: np.ndarray):
        # Fit scalers on training data
        # Reshape to 2D for scaling
        n_samples, n_timesteps, n_features = X_train.shape
        X_2d = X_train.reshape(-1, n_features)

        # Fit consumption scaler (first feature)
        self.scalers['consumption'].fit(X_2d[:, 0:1])

        # Fit other scalers as needed
        logger.info("Scalers fitted on training data")

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Transform features using fitted scalers
        if not self.is_fitted:
            raise ValueError("Must fit scalers before transform")

        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = X_2d.copy()

        # Scale consumption (first feature)
        X_scaled[:, 0:1] = self.scalers['consumption'].transform(X_2d[:, 0:1])

        return X_scaled.reshape(n_samples, n_timesteps, n_features)

    def get_feature_names(self) -> List[str]:
        # Get list of feature names
        return self.feature_names