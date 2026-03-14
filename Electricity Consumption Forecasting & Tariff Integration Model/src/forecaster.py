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

    def _create_ensemble_result(self, combined_forecast: Dict[int, float],
                               confidence: float,
                               uncertainty: Dict,
                               individual_forecasts: Dict,
                               confidences: Dict,
                               user_data: Dict) -> Dict:
        # Create formatted ensemble result
        # Calculate statistics
        values = list(combined_forecast.values())
        annual_total = sum(values)
        annual_avg = annual_total / 12
        peak_month = max(range(1, 13), key=lambda m: combined_forecast[m])
        low_month = min(range(1, 13), key=lambda m: combined_forecast[m])

        # Calculate trend
        monthly_diffs = [combined_forecast[m+1] - combined_forecast[m] for m in range(1, 12)]
        avg_change = np.mean(monthly_diffs)

        if avg_change > 5:
            trend = 'increasing'
        elif avg_change < -5:
            trend = 'decreasing'
        else:
            trend = 'stable'

        # Calculate seasonality index
        seasonality = np.std(values) / annual_avg if annual_avg > 0 else 0

        forecast_stats = {
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': peak_month,
            'peak_consumption': combined_forecast[peak_month],
            'low_month': low_month,
            'low_consumption': combined_forecast[low_month],
            'trend': trend,
            'monthly_variability': float(np.std(values)),
            'seasonality_index': seasonality,
            'overall_confidence': confidence
        }

        # Monthly details for display
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_details = []
        for month in range(1, 13):
            monthly_details.append({
                'month': month,
                'month_name': month_names[month - 1],
                'consumption_kwh': round(combined_forecast[month], 1),
                'confidence': confidence * (1 - 0.1 * abs(month - peak_month) / 6),  # Decay from peak
                'lower_bound': uncertainty[month]['lower_bound'],
                'upper_bound': uncertainty[month]['upper_bound'],
                'season': self._get_sri_lanka_season(month),
                'is_holiday_month': month in self.config['features']['sri_lanka']['holiday_months']
            })

        result = {
            'forecast': {
                'monthly_values': combined_forecast,
                'monthly_confidence': {m: d['confidence'] for m, d in enumerate(monthly_details, 1)},
                'monthly_details': monthly_details,
                'uncertainty_ranges': uncertainty,
                'statistics': forecast_stats
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'user_months_provided': len(user_data.get('consumption_months', {})),
                'ensemble_methods': list(individual_forecasts.keys()),
                'method_weights': self.weights,
                'method_confidences': confidences,
                'forecast_method': 'ensemble'
            }
        }

        return result

    def _create_forecast_result(self, forecast: Dict[int, float],
                               confidence: float,
                               user_data: Dict,
                               method: str) -> Dict:
        # Create result for single-method forecast
        # Similar to above but with single method
        values = list(forecast.values())
        annual_total = sum(values)
        annual_avg = annual_total / 12
        peak_month = max(range(1, 13), key=lambda m: forecast[m])

        forecast_stats = {
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': peak_month,
            'peak_consumption': forecast[peak_month],
            'overall_confidence': confidence
        }

        # Simple uncertainty (20%)
        uncertainty = {}
        for month in range(1, 13):
            val = forecast[month]
            uncertainty[month] = {
                'lower_bound': val * 0.8,
                'upper_bound': val * 1.2,
                'std_dev': val * 0.2
            }

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_details = []
        for month in range(1, 13):
            monthly_details.append({
                'month': month,
                'month_name': month_names[month - 1],
                'consumption_kwh': round(forecast[month], 1),
                'confidence': confidence,
                'season': self._get_sri_lanka_season(month),
                'is_holiday_month': month in self.config['features']['sri_lanka']['holiday_months']
            })

        result = {
            'forecast': {
                'monthly_values': forecast,
                'monthly_confidence': {m: confidence for m in range(1, 13)},
                'monthly_details': monthly_details,
                'uncertainty_ranges': uncertainty,
                'statistics': forecast_stats
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'user_months_provided': len(user_data.get('consumption_months', {})),
                'forecast_method': method
            }
        }

        return result

    def _create_fallback_forecast(self, user_data: Dict) -> Dict:
        # Create fallback forecast when all methods fail
        logger.warning("Using fallback forecast (national average)")

        seasonal_factors = self.config['features']['sri_lanka']['seasonal_factors']
        national_avg = 300  # kWh/month

        forecast = {}
        for month in range(1, 13):
            forecast[month] = national_avg * seasonal_factors[month]

        return self._create_forecast_result(
            forecast,
            0.3,
            user_data,
            'fallback_national_average'
        )

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