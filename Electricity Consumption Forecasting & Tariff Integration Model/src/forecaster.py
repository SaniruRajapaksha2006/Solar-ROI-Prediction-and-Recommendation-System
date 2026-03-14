import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple forecasting methods
    - LSTM as primary
    - Pattern-based as fallback
    """

    def __init__(self, config: Dict):
        self.config = config
        self.forecasters = {}
        self.weights = {}
        self.performance_history = {}

    def add_forecaster(self, name: str, forecaster, weight: float):
        # Add a forecasting method to the ensemble

        self.forecasters[name] = forecaster
        self.weights[name] = weight
        self.performance_history[name] = []

        logger.info(f"Added forecaster: {name} with weight {weight}")

    def _combine_forecasts(self, forecasts: Dict[str, Dict[int, float]], confidences: Dict[str, float],
                          user_data: Dict) -> Dict[int, float]:
        # Combine forecasts using dynamic weights

        combined = {month: 0.0 for month in range(1, 13)}

        # Calculate dynamic weights based on confidence
        total_confidence = sum(confidences.values())
        if total_confidence > 0:
            dynamic_weights = {
                name: conf * self.weights.get(name, 0.5) / total_confidence
                for name, conf in confidences.items()
            }
        else:
            dynamic_weights = self.weights

        # Normalize weights
        weight_sum = sum(dynamic_weights.values())
        if weight_sum > 0:
            dynamic_weights = {k: v / weight_sum for k, v in dynamic_weights.items()}

        # Combine
        for month in range(1, 13):
            weighted_sum = 0.0
            for name, forecast in forecasts.items():
                if forecast and month in forecast:
                    weighted_sum += forecast[month] * dynamic_weights[name]
            combined[month] = weighted_sum

        logger.info(f"Ensemble weights: {dynamic_weights}")
        return combined

    def _calculate_combined_confidence(self, confidences: Dict[str, float]) -> float:
        # Calculate combined confidence score
        if not confidences:
            return 0.3
        return float(np.mean(list(confidences.values())))

    def _calculate_uncertainty(self, forecasts: Dict[str, Dict[int, float]]) -> Dict:
        # Calculate uncertainty ranges from ensemble spread
        uncertainty = {}

        for month in range(1, 13):
            month_values = []
            for name, forecast in forecasts.items():
                if forecast and month in forecast:
                    month_values.append(forecast[month])

            if len(month_values) >= 2:
                uncertainty[month] = {
                    'lower_bound': float(np.min(month_values)),
                    'upper_bound': float(np.max(month_values)),
                    'std_dev': float(np.std(month_values)),
                    'q25': float(np.percentile(month_values, 25)),
                    'q75': float(np.percentile(month_values, 75)),
                    'n_methods': len(month_values)
                }
            elif len(month_values) == 1:
                # Single method, use 20% uncertainty
                val = month_values[0]
                uncertainty[month] = {
                    'lower_bound': val * 0.8,
                    'upper_bound': val * 1.2,
                    'std_dev': val * 0.2,
                    'q25': val * 0.9,
                    'q75': val * 1.1,
                    'n_methods': 1
                }
            else:
                uncertainty[month] = {
                    'lower_bound': 0,
                    'upper_bound': 0,
                    'std_dev': 0,
                    'q25': 0,
                    'q75': 0,
                    'n_methods': 0
                }

        return uncertainty

    def _get_sri_lanka_season(self, month: int) -> str:
        # sGet Sri Lankan season for month
        if month in [12, 1, 2]:
            return "NE Monsoon"
        elif month in [3, 4]:
            return "Dry Season"
        elif month in [5, 6, 7, 8, 9]:
            return "SW Monsoon"
        else:
            return "Dry Season"