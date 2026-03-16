import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

def find_project_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / 'src').exists():
            return current
        current = current.parent
    return Path.cwd()

project_root = find_project_root()
sys.path.insert(0, str(project_root))

from .time_series_split import TemporalSplitter
from ..models.lstm_model import LSTMForecaster
from ..features.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.splitter = TemporalSplitter(config)
        self.results_history = []

    def evaluate_forecast(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_true_series: Optional[np.ndarray] = None) -> Dict:
        metrics = {}

        # 1. MAE (Mean Absolute Error)
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))

        # 2. RMSE (Root Mean Square Error)
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # 3. MAPE (Mean Absolute Percentage Error)
        mask = y_true > 0
        if np.any(mask):
            metrics['mape'] = float(np.mean(
                np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            ) * 100)
        else:
            metrics['mape'] = np.nan

        # 4. R² (Coefficient of Determination)
        metrics['r2'] = float(r2_score(y_true, y_pred))

        # 5. Bias (Mean Error)
        metrics['bias'] = float(np.mean(y_pred - y_true))
        metrics['bias_pct'] = float(metrics['bias'] / np.mean(y_true) * 100 if np.mean(y_true) > 0 else 0)

        # 6. Peak Error
        peak_idx = np.argmax(y_true)
        metrics['peak_error'] = float(abs(y_true[peak_idx] - y_pred[peak_idx]))
        metrics['peak_error_pct'] = float(metrics['peak_error'] / y_true[peak_idx] * 100)

        # 7. MASE (Mean Absolute Scaled Error)
        if y_true_series is not None and len(y_true_series) > 12:
            # Seasonal naive forecast (same month last year)
            naive_errors = np.abs(y_true_series[12:] - y_true_series[:-12])
            naive_mae = np.mean(naive_errors)
            if naive_mae > 0:
                metrics['mase'] = metrics['mae'] / naive_mae
            else:
                metrics['mase'] = np.nan

        return metrics

    def compare_with_baselines(self, df: pd.DataFrame) -> Dict:
        baselines = {
            'persistence': self._persistence_forecast,
            'seasonal_naive': self._seasonal_naive_forecast,
            'historical_average': self._historical_average_forecast
        }

        results = {}

        for name, baseline_fn in baselines.items():
            try:
                metrics = baseline_fn(df)
                results[name] = metrics
                logger.info(f"Baseline {name}: MAE={metrics.get('mae', 0):.2f} kWh")
            except Exception as e:
                logger.error(f"Error in baseline {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def _persistence_forecast(self, df: pd.DataFrame) -> Dict:
        # Persistence forecast: next month = same as last month

        errors = []

        for account in df['ACCOUNT_NO'].unique():
            account_data = df[df['ACCOUNT_NO'] == account].sort_values(['YEAR', 'MONTH'])
            values = account_data['NET_CONSUMPTION_kWh'].values

            if len(values) < 2:
                continue

            # Predict next month = last month
            y_true = values[1:]
            y_pred = values[:-1]

            errors.extend(np.abs(y_true - y_pred))

        return {
            'mae': float(np.mean(errors)),
            'description': 'Next month = last month'
        }

    def _seasonal_naive_forecast(self, df: pd.DataFrame) -> Dict:
        # Seasonal naive: next month = same month last year

        errors = []

        for account in df['ACCOUNT_NO'].unique():
            account_data = df[df['ACCOUNT_NO'] == account].sort_values(['YEAR', 'MONTH'])

            # Need at least 13 months for seasonal naive
            if len(account_data) < 13:
                continue

            values = account_data['NET_CONSUMPTION_kWh'].values

            # Predict using value from 12 months ago
            y_true = values[12:]
            y_pred = values[:-12]

            errors.extend(np.abs(y_true - y_pred))

        return {
            'mae': float(np.mean(errors)) if errors else 0,
            'description': 'Next month = same month last year'
        }

    def _historical_average_forecast(self, df: pd.DataFrame) -> Dict:
        # Historical average: predict using monthly averages

        # Calculate monthly averages
        monthly_avg = df.groupby('MONTH')['NET_CONSUMPTION_kWh'].mean()

        errors = []

        for _, row in df.iterrows():
            month = row['MONTH']
            actual = row['NET_CONSUMPTION_kWh']
            predicted = monthly_avg[month]

            errors.append(abs(actual - predicted))

        return {
            'mae': float(np.mean(errors)),
            'description': 'Historical monthly average'
        }

    def analyze_residuals(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict:
        residuals = y_true - y_pred

        analysis = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
        }

        # Normality test
        if len(residuals) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for large samples
            analysis['shapiro_p_value'] = float(shapiro_p)
            analysis['is_normal'] = shapiro_p > 0.05

        # Autocorrelation test
        if len(residuals) > 1:
            # Durbin-Watson statistic for lag-1 autocorrelation
            dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
            analysis['durbin_watson'] = float(dw)

        # Heteroscedasticity test (simplified)
        abs_residuals = np.abs(residuals)
        if len(abs_residuals) > 1:
            # Correlation between |residuals| and predicted values
            corr = np.corrcoef(abs_residuals, y_pred)[0, 1]
            analysis['heteroscedasticity_corr'] = float(corr)
            analysis['constant_variance'] = abs(corr) < 0.3

        return analysis

    def save_validation_report(self, results: Dict, filepath: str):
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {filepath}")