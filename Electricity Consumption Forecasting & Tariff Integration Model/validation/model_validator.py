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

    def cross_validate(self, df: pd.DataFrame, model: Optional[LSTMForecaster] = None,
                       n_folds: int = 5) -> Dict:
        # Get time-ordered indices
        df = df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)

        # Create time index
        df['time_idx'] = df['YEAR'] * 12 + df['MONTH']
        unique_times = df['time_idx'].unique()

        fold_size = len(unique_times) // (n_folds + 1)

        cv_results = {
            'fold_metrics': [],
            'average_metrics': {},
            'std_metrics': {}
        }

        for fold in range(n_folds):
            # Define train/test periods
            train_end_idx = (fold + 1) * fold_size
            test_start_idx = train_end_idx
            test_end_idx = min((fold + 2) * fold_size, len(unique_times))

            train_times = unique_times[:train_end_idx]
            test_times = unique_times[test_start_idx:test_end_idx]

            # Split data
            train_df = df[df['time_idx'].isin(train_times)]
            test_df = df[df['time_idx'].isin(test_times)]

            logger.info(f"Fold {fold + 1}: Train {train_times[0]}-{train_times[-1]}, "
                        f"Test {test_times[0]}-{test_times[-1]}")

            # Train model on this fold
            fold_model = self._train_fold_model(train_df)

            # Make predictions on test set
            fold_metrics = self._evaluate_fold(fold_model, test_df)

            cv_results['fold_metrics'].append({
                'fold': fold + 1,
                'train_range': (int(train_times[0]), int(train_times[-1])),
                'test_range': (int(test_times[0]), int(test_times[-1])),
                'metrics': fold_metrics
            })

        # Calculate average metrics
        metrics_list = [f['metrics'] for f in cv_results['fold_metrics']]
        if metrics_list:
            for metric in metrics_list[0].keys():
                values = [m[metric] for m in metrics_list if metric in m]
                if values:
                    cv_results['average_metrics'][metric] = float(np.mean(values))
                    cv_results['std_metrics'][metric] = float(np.std(values))

        return cv_results

    def _train_fold_model(self, train_df: pd.DataFrame) -> LSTMForecaster:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(self.config)

        # Prepare training data
        X_train, y_train = feature_engineer.prepare_for_training(train_df)

        # Initialize and train model
        model = LSTMForecaster(self.config)

        # Build model with correct input shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        model.build_model(input_shape)

        # Train the model
        # Using 30 epochs for CV to keep it fast
        model.train(X_train, y_train, None, None, epochs=30)

        return model

    def _evaluate_fold(self, model: LSTMForecaster, test_df: pd.DataFrame) -> Dict:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(self.config)

        # Prepare test data
        X_test, y_test = feature_engineer.prepare_for_training(test_df)

        # Make predictions
        y_pred = model.predict(X_test)

        # Flatten for metrics
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        # Calculate metrics
        return self.evaluate_forecast(y_test_flat, y_pred_flat, y_test_flat)

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

    def diebold_mariano_test(self, df: pd.DataFrame,
                             model1_pred: np.ndarray,
                             model2_pred: np.ndarray,
                             y_true: np.ndarray) -> Dict:
        # Diebold-Mariano test for forecast comparison
        # Calculate loss differential
        loss1 = np.abs(y_true - model1_pred)
        loss2 = np.abs(y_true - model2_pred)
        d = loss1 - loss2

        # Calculate test statistic
        n = len(d)
        d_bar = np.mean(d)

        # Estimate variance (using Newey-West for autocorrelation)
        gamma = []
        for k in range(min(4, n - 1)):
            cov = np.cov(d[:-k - 1], d[k + 1:])[0, 1] if k + 1 < n else 0
            gamma.append(cov)

        var_d = gamma[0] + 2 * sum(gamma[1:]) if gamma else 0

        if var_d <= 0:
            return {'dm_statistic': 0, 'p_value': 1.0, 'result': 'Cannot compute'}

        dm_stat = d_bar / np.sqrt(var_d / n)

        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'better_model': 'model1' if d_bar < 0 else 'model2'
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

    def validate_on_household_types(self, df: pd.DataFrame,
                                    predictions: Dict) -> Dict:
        # Define household types based on consumption
        df['consumption_level'] = pd.cut(
            df['NET_CONSUMPTION_kWh'],
            bins=[0, 90, 150, 300, 2000],
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )

        results = {}

        for level in ['Low', 'Medium-Low', 'Medium-High', 'High']:
            level_accounts = df[df['consumption_level'] == level]['ACCOUNT_NO'].unique()

            level_errors = []
            for account in level_accounts:
                if account in predictions:
                    level_errors.extend(predictions[account].get('errors', []))

            if level_errors:
                results[level] = {
                    'n_accounts': len(level_accounts),
                    'mae': float(np.mean(level_errors)),
                    'rmse': float(np.sqrt(np.mean(np.square(level_errors))))
                }
            else:
                results[level] = {
                    'n_accounts': len(level_accounts),
                    'mae': None,
                    'rmse': None
                }

        return results

    def validate_by_season(self, df: pd.DataFrame,
                           predictions: Dict) -> Dict:
        # Define seasons
        seasons = {
            'NE_Monsoon': [12, 1, 2],
            'Dry_Season': [3, 4],
            'SW_Monsoon': [5, 6, 7, 8, 9],
            'Inter_Monsoon': [10, 11]
        }

        results = {}

        for season_name, months in seasons.items():
            season_errors = []

            for account in predictions:
                for month in months:
                    if month in predictions[account]:
                        season_errors.append(predictions[account][month].get('error', 0))

            if season_errors:
                results[season_name] = {
                    'n_samples': len(season_errors),
                    'mae': float(np.mean(season_errors)),
                    'rmse': float(np.sqrt(np.mean(np.square(season_errors))))
                }

        return results

    def save_validation_report(self, results: Dict, filepath: str):
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {filepath}")