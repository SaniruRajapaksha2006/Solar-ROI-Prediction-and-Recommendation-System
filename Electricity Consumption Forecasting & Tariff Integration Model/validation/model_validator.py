import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
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

logger = logging.getLogger(__name__)


class ModelValidator:
    # Comprehensive model validation and evaluation


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

        # 8. Prediction Interval Coverage
        # This would require prediction intervals from the model

        return metrics