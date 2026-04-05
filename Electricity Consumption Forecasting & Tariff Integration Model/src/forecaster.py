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

    def forecast(self, user_data: Dict, similar_households: List,
                 features: Optional[Dict] = None) -> Dict:

        logger.info("Generating ensemble forecast")

        forecasts = {}
        confidences = {}
        errors = []

        # Get forecasts from each method
        for name, forecaster in self.forecasters.items():
            try:
                if name == 'lstm' and features is not None:
                    # LSTM forecast
                    result = forecaster.forecast(user_data, features)
                    if result and 'forecast' in result:
                        forecasts[name] = result['forecast']['monthly_values']
                        confidences[name] = result['forecast']['statistics']['overall_confidence']
                        logger.info(f"LSTM forecast successful, confidence: {confidences[name]:.2f}")
                    else:
                        logger.warning("LSTM forecast failed, falling back")
                        forecasts[name] = None
                        confidences[name] = 0.0

                elif name == 'pattern':
                    # Pattern-based forecast
                    result = forecaster.extract_pattern_with_forecast(similar_households, user_data)
                    if result and 'forecast' in result:
                        forecasts[name] = result['forecast']['monthly_values']
                        confidences[name] = result['forecast']['statistics']['overall_confidence']
                        logger.info(f"Pattern forecast successful, confidence: {confidences[name]:.2f}")
                    else:
                        forecasts[name] = None
                        confidences[name] = 0.0

            except Exception as e:
                logger.error(f"Error in {name} forecast: {e}")
                forecasts[name] = None
                confidences[name] = 0.0

        # Check if we have any valid forecasts
        valid_forecasts = {name: f for name, f in forecasts.items() if f is not None}

        if not valid_forecasts:
            logger.error("All forecasting methods failed")
            return self._create_fallback_forecast(user_data)

        # If only one method succeeded, use it
        if len(valid_forecasts) == 1:
            name = list(valid_forecasts.keys())[0]
            logger.info(f"Only {name} forecast available, using it")
            return self._create_forecast_result(
                valid_forecasts[name],
                confidences[name],
                user_data,
                method=f"{name}_only"
            )

        # Combine forecasts with dynamic weights
        combined_forecast = self._combine_forecasts(
            valid_forecasts,
            confidences,
            user_data
        )

        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(confidences)

        # Calculate uncertainty ranges from ensemble spread
        uncertainty = self._calculate_uncertainty(valid_forecasts)

        # Prepare result
        result = self._create_ensemble_result(
            combined_forecast,
            combined_confidence,
            uncertainty,
            valid_forecasts,
            confidences,
            user_data
        )

        logger.info(f"Ensemble forecast complete: annual={result['forecast']['statistics']['annual_total']:.0f} kWh")

        return result

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

        user_months = user_data.get('consumption_months', {})
        if user_months:
            user_avg = sum(user_months.values()) / len(user_months)
            model_avg = sum(combined.values()) / 12
            if model_avg > 0 and user_avg > 0:
                scale_factor = user_avg / model_avg
                for m in combined:
                    combined[m] *= scale_factor

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

        user_months = user_data.get('consumption_months', {})
        if user_months:
            user_avg = sum(user_months.values()) / len(user_months)
            model_avg = sum(forecast.values()) / 12
            if model_avg > 0 and user_avg > 0:
                scale_factor = user_avg / model_avg
                for m in forecast:
                    forecast[m] *= scale_factor

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

    def update_performance(self, method: str, actual: Dict[int, float],
                          predicted: Dict[int, float]):
        # Update performance history for a method
        if method not in self.performance_history:
            self.performance_history[method] = []

        # Calculate error
        months = set(actual.keys()) & set(predicted.keys())
        if months:
            errors = [abs(actual[m] - predicted[m]) / actual[m] for m in months]
            mape = np.mean(errors) * 100
            self.performance_history[method].append(mape)

            # Adjust weight based on recent performance
            if len(self.performance_history[method]) >= 3:
                recent_mape = np.mean(self.performance_history[method][-3:])
                self._adjust_weight(method, recent_mape)

    def _adjust_weight(self, method: str, recent_error: float):
        # Adjust ensemble weight based on recent performance
        # Lower error = higher weight
        if method in self.weights:
            # Inverse relationship: weight ∝ 1/error
            base_weight = 1.0 / recent_error if recent_error > 0 else 1.0

            # Normalize with other methods
            other_methods = [m for m in self.weights.keys() if m != method]
            if other_methods:
                other_weights = [self.weights[m] for m in other_methods]
                total_other = sum(other_weights)

                # Set new weight, keeping total = 1.0
                self.weights[method] = base_weight / (base_weight + total_other)

                # Adjust others proportionally
                scale = (1.0 - self.weights[method]) / total_other if total_other > 0 else 0
                for m in other_methods:
                    self.weights[m] *= scale

            logger.info(f"Adjusted weight for {method} to {self.weights[method]:.3f}")