import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class ConsumptionPatternExtractor:
    """
    Extracts consumption patterns from similar households
    Creates a representative 12-month pattern for forecasting
    """

    def __init__(self, data_loader, config: Dict):
        self.data_loader = data_loader
        self.config = config
        self.seasonal_factors = config['features']['sri_lanka']['seasonal_factors']

    def extract_pattern_safe(self, similar_households: List[Tuple[str, float]],
                            user_data: Dict) -> Dict:
        # Extract 12-month consumption pattern using ONLY historical data
        logger.info("Extracting consumption pattern from similar households (safe mode)")

        if not similar_households:
            logger.warning("No similar households found, using default pattern")
            return self._create_default_pattern()

        accounts = [acc for acc, _ in similar_households]
        similarity_scores = [score for _, score in similar_households]

        # Get user's last month
        user_months = user_data.get('consumption_months', {})
        last_user_month = max(user_months.keys()) if user_months else 12

        logger.info(f"Using {len(accounts)} similar households, last user month: {last_user_month}")

        # Collect monthly data using ONLY months up to last_user_month
        monthly_data = self._collect_monthly_data_safe(accounts, similarity_scores, last_user_month)

        # Calculate robust pattern using median
        pattern = self._calculate_robust_pattern(monthly_data)

        # Adjust pattern based on user's consumption (using ONLY their months)
        if user_months:
            pattern = self._adjust_for_user_consumption_safe(pattern, user_months)

        # Apply Sri Lankan seasonal adjustments
        pattern = self._apply_sri_lanka_seasonal_pattern(pattern)

        # Calculate confidence
        confidence = self._calculate_pattern_confidence(similarity_scores, pattern, monthly_data)

        # Add metadata
        result = self._add_metadata(pattern, confidence, accounts, similarity_scores, monthly_data)

        logger.info(f"Pattern extraction complete: confidence={confidence:.3f}")

        return result

    def extract_pattern_with_forecast(self, similar_households: List[Tuple[str, float]],
                                      user_data: Dict) -> Dict:
        # Extract pattern and generate forecast in one step

        pattern_result = self.extract_pattern_safe(similar_households, user_data)

        # Generate forecast from pattern
        forecast = self._generate_forecast_from_pattern(pattern_result, user_data)

        return forecast

    def _collect_monthly_data_safe(self, accounts: List[str], similarity_scores: List[float],
                                  last_month: int) -> Dict[int, List[Dict]]:
        # Collect monthly data using ONLY months up to last_month

        monthly_data = {month: [] for month in range(1, 13)}

        # Normalize similarity scores to weights
        total_score = sum(similarity_scores)
        weights = [score / total_score for score in similarity_scores] if total_score > 0 else [1/len(accounts)] * len(accounts)

        for idx, account in enumerate(accounts):
            profile = self.data_loader.get_customer_profile(account)
            if not profile:
                continue

            weight = weights[idx]

            # Get household's values for all months
            household_values = []
            for month in range(1, 13):
                if month in profile['monthly_pattern']:
                    household_values.append(profile['monthly_pattern'][month]['net_consumption'])

            if not household_values:
                continue

            household_median = np.median(household_values)

            # Only use months UP TO last_month
            for month in range(1, last_month + 1):
                if month in profile['monthly_pattern']:
                    consumption = profile['monthly_pattern'][month]['net_consumption']

                    # Filter outliers
                    if 10 <= consumption <= 2000:
                        # Check if within reasonable range
                        if household_median * 0.3 <= consumption <= household_median * 3.0:
                            monthly_data[month].append({
                                'value': consumption,
                                'weight': weight,
                                'account': account,
                                'household_median': household_median
                            })

            # Months after last_month are NOT used (avoid look-ahead)

        # Log data collection statistics
        total_points = sum(len(data) for data in monthly_data.values())
        logger.info(f"Collected {total_points} data points after filtering (months 1-{last_month})")

        return monthly_data

    def _calculate_robust_pattern(self, monthly_data: Dict[int, List[Dict]]) -> Dict:
        # Calculate robust pattern using median and IQR

        monthly_median = {}
        monthly_iqr = {}
        monthly_min_reasonable = {}
        monthly_max_reasonable = {}
        data_points_per_month = {}

        for month in range(1, 13):
            month_data = monthly_data.get(month, [])
            data_points_per_month[month] = len(month_data)

            if len(month_data) < 3:
                # Insufficient data, use seasonal default
                seasonal_factor = self.seasonal_factors.get(month, 1.0)
                base_value = 350 * seasonal_factor
                monthly_median[month] = base_value
                monthly_iqr[month] = base_value * 0.4
                monthly_min_reasonable[month] = base_value * 0.6
                monthly_max_reasonable[month] = base_value * 1.4
                continue

            # Extract values and weights
            values = [d['value'] for d in month_data]
            weights = [d['weight'] for d in month_data]

            # Use weighted median
            monthly_median[month] = self._weighted_median(values, weights)

            # Calculate IQR
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            monthly_iqr[month] = float(q75 - q25)

            # Calculate reasonable bounds (10th and 90th percentiles)
            monthly_min_reasonable[month] = float(np.percentile(values, 10))
            monthly_max_reasonable[month] = float(np.percentile(values, 90))

        # Apply smoothing
        monthly_median = self._smooth_monthly_values(monthly_median)

        # Calculate annual statistics
        annual_total = sum(monthly_median.values())
        annual_avg = annual_total / 12

        # Find peak month
        peak_month = max(range(1, 13), key=lambda m: monthly_median[m])

        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(data_points_per_month, monthly_iqr, annual_avg)

        return {
            'monthly_median': monthly_median,
            'monthly_iqr': monthly_iqr,
            'monthly_min_reasonable': monthly_min_reasonable,
            'monthly_max_reasonable': monthly_max_reasonable,
            'data_points_per_month': data_points_per_month,
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': peak_month,
            'peak_consumption': monthly_median[peak_month],
            'data_quality_score': data_quality_score
        }

    def _weighted_median(self, values: List[float], weights: List[float]) -> float:
        # Calculate weighted median
        df = sorted(zip(values, weights), key=lambda x: x[0])
        values_sorted, weights_sorted = zip(*df)

        total_weight = sum(weights_sorted)
        cum_weight = 0

        for val, w in zip(values_sorted, weights_sorted):
            cum_weight += w
            if cum_weight >= total_weight / 2:
                return val

        return values_sorted[-1] if values_sorted else 0

    def _smooth_monthly_values(self, monthly_values: Dict[int, float]) -> Dict[int, float]:
        # Apply smoothing to avoid drastic month-to-month changes
        smoothed = monthly_values.copy()

        for month in range(1, 13):
            prev_month = month - 1 if month > 1 else 12
            next_month = month + 1 if month < 12 else 1

            smoothed[month] = (
                monthly_values[prev_month] * 0.25 +
                monthly_values[month] * 0.5 +
                monthly_values[next_month] * 0.25
            )

        return smoothed

    def _calculate_data_quality_score(self, data_points: Dict[int, int], monthly_iqr: Dict[int, float],
                                     annual_avg: float) -> float:
        # Calculate overall data quality score
        # Score 1: Sufficient data points
        sufficient_months = sum(1 for month in range(1, 13) if data_points.get(month, 0) >= 3)
        score1 = sufficient_months / 12

        # Score 2: Low variability
        if annual_avg > 0:
            avg_iqr = np.mean(list(monthly_iqr.values()))
            variability = avg_iqr / annual_avg
            score2 = max(0, 1 - (variability / 0.5))
        else:
            score2 = 0.5

        # Score 3: Consistency
        monthly_medians = [monthly_iqr[m] for m in range(1, 13)]
        if len(monthly_medians) > 1:
            consistency = np.std(monthly_medians) / np.mean(monthly_medians)
            score3 = max(0, 1 - (consistency / 0.3))
        else:
            score3 = 0.5

        # Weighted combination
        final_score = (score1 * 0.6) + (score2 * 0.2) + (score3 * 0.2)

        return min(1.0, max(0.0, final_score))

    def _adjust_for_user_consumption_safe(self, pattern: Dict, user_months_data: Dict) -> Dict:
        # Adjust pattern based on user's consumption (using ONLY their months)

        logger.info("Adjusting pattern based on user's consumption")

        # Find overlapping months
        common_months = [m for m in user_months_data.keys() if m in pattern['monthly_median']]

        if not common_months:
            logger.warning("No overlapping months for adjustment")
            return pattern

        # Calculate user vs pattern ratio using ONLY overlapping months
        user_values = [user_months_data[m] for m in common_months]
        pattern_values = [pattern['monthly_median'][m] for m in common_months]

        user_avg = np.median(user_values)
        pattern_avg = np.median(pattern_values)

        if pattern_avg > 0:
            adjustment_factor = user_avg / pattern_avg
        else:
            adjustment_factor = 1.0

        # Limit adjustment to reasonable range
        adjustment_factor = max(0.8, min(1.2, adjustment_factor))

        logger.info(f"Adjustment factor: {adjustment_factor:.2f} (based on {len(common_months)} months)")
        logger.info(f"User median: {user_avg:.1f} kWh, Pattern median: {pattern_avg:.1f} kWh")

        # Apply adjustment to all months
        adjusted_pattern = pattern.copy()

        for month in range(1, 13):
            adjusted_pattern['monthly_median'][month] *= adjustment_factor
            adjusted_pattern['monthly_iqr'][month] *= adjustment_factor
            adjusted_pattern['monthly_min_reasonable'][month] *= adjustment_factor
            adjusted_pattern['monthly_max_reasonable'][month] *= adjustment_factor

        # Recalculate statistics
        adjusted_pattern['annual_total'] = sum(adjusted_pattern['monthly_median'].values())
        adjusted_pattern['annual_average'] = adjusted_pattern['annual_total'] / 12

        # Update peak month
        peak_month = max(range(1, 13), key=lambda m: adjusted_pattern['monthly_median'][m])
        adjusted_pattern['peak_month'] = peak_month
        adjusted_pattern['peak_consumption'] = adjusted_pattern['monthly_median'][peak_month]

        return adjusted_pattern

    def _apply_sri_lanka_seasonal_pattern(self, pattern: Dict) -> Dict:
        # Apply Sri Lankan seasonal pattern knowledge

        adjusted_pattern = pattern.copy()
        monthly_median = pattern['monthly_median'].copy()
        annual_avg = pattern['annual_average']

        # Blend with Sri Lanka template
        blend_weight = 0.7

        for month in range(1, 13):
            template_value = annual_avg * self.seasonal_factors[month]
            blended_value = (monthly_median[month] * blend_weight) + (template_value * (1 - blend_weight))
            monthly_median[month] = blended_value

        # Enforce key Sri Lankan patterns:

        # 1. April should be peak or near-peak
        current_peak = max(range(1, 13), key=lambda m: monthly_median[m])
        if current_peak != 4:
            # Boost April
            current_peak_val = monthly_median[current_peak]
            april_val = monthly_median[4]
            if april_val < current_peak_val * 0.95:
                boost_factor = (current_peak_val * 1.05) / april_val
                monthly_median[4] *= boost_factor

        # 2. July/August should be low (monsoon)
        july_val = monthly_median[7]
        august_val = monthly_median[8]
        avg_val = annual_avg

        if july_val > avg_val * 1.1:
            monthly_median[7] *= 0.9
        if august_val > avg_val * 1.1:
            monthly_median[8] *= 0.9

        # Smooth transitions
        monthly_median = self._smooth_monthly_values(monthly_median)

        # Recalculate statistics
        adjusted_pattern['monthly_median'] = monthly_median
        adjusted_pattern['annual_total'] = sum(monthly_median.values())
        adjusted_pattern['annual_average'] = adjusted_pattern['annual_total'] / 12

        # Update peak month
        new_peak = max(range(1, 13), key=lambda m: monthly_median[m])
        adjusted_pattern['peak_month'] = new_peak
        adjusted_pattern['peak_consumption'] = monthly_median[new_peak]

        return adjusted_pattern

    def _calculate_pattern_confidence(self, similarity_scores: List[float], pattern: Dict,
                                    monthly_data: Dict[int, List[Dict]]) -> float:
        # Calculate confidence in the extracted pattern
        if not similarity_scores:
            return 0.3

        # Confidence based on similarity
        avg_similarity = np.mean(similarity_scores)
        similarity_confidence = avg_similarity

        # Confidence based on data quality
        data_quality_confidence = pattern['data_quality_score']

        # Confidence based on data sufficiency
        total_points = sum(len(data) for data in monthly_data.values())
        points_per_month = total_points / 12
        sufficiency_confidence = min(1.0, points_per_month / 5)

        # Combined confidence
        confidence = (
            similarity_confidence * 0.4 +
            data_quality_confidence * 0.3 +
            sufficiency_confidence * 0.3
        )

        return float(max(0.3, min(0.95, confidence)))

    def _add_metadata(self, pattern: Dict, confidence: float, accounts: List[str], similarity_scores: List[float],
                     monthly_data: Dict[int, List[Dict]]) -> Dict:
        # Add metadata to the pattern
        # Create sample households info
        sample_households = []
        for account in accounts[:5]:
            profile = self.data_loader.get_customer_profile(account)
            if profile:
                # Count data points for this account
                data_points = sum(1 for month_data in monthly_data.values()
                                for d in month_data if d['account'] == account)

                sample_households.append({
                    'account_no': account[:8] + '...',
                    'similarity': similarity_scores[accounts.index(account)],
                    'annual_consumption': profile['annual_stats']['total'],
                    'monthly_average': profile['annual_stats']['average'],
                    'tariff': profile['tariff'],
                    'has_solar': profile['has_solar'],
                    'data_points': data_points
                })

        # Calculate data statistics
        total_points = sum(len(data) for data in monthly_data.values())
        months_with_data = sum(1 for month in range(1, 13) if len(monthly_data.get(month, [])) >= 3)

        result = {
            'pattern': pattern,
            'confidence': confidence,
            'metadata': {
                'similar_households_count': len(accounts),
                'average_similarity': float(np.mean(similarity_scores)) if similarity_scores else 0,
                'total_data_points': total_points,
                'months_with_sufficient_data': months_with_data,
                'data_quality_score': pattern['data_quality_score'],
                'sample_households': sample_households,
                'extraction_method': 'robust_median_safe',
                'notes': 'No look-ahead bias - uses only historical data'
            }
        }

        return result

    def _generate_forecast_from_pattern(self, pattern_result: Dict, user_data: Dict) -> Dict:
        # Generate 12-month forecast from extracted pattern with proper month alignment
        pattern = pattern_result['pattern']
        confidence = pattern_result['confidence']

        # Get user's last month to align forecast correctly
        user_months = user_data.get('consumption_months', {})
        if user_months:
            last_user_month = max(user_months.keys())
            # You might want to pass year information in user_data
            last_user_year = user_data.get('year', 2025)  # Default to 2025 if not provided
        else:
            last_user_month = 12
            last_user_year = 2025

        logger.info(f"Aligning forecast: user's last month = {last_user_month}/{last_user_year}")

        # Calculate starting month (next month after user's last)
        start_month = (last_user_month % 12) + 1
        start_year = last_user_year + (1 if start_month == 1 else 0)

        logger.info(f"Forecast will start from: {start_month}/{start_year}")

        # Get the base pattern (months 1-12)
        base_pattern = pattern['monthly_median'].copy()

        # Reorder the pattern to start from the correct month
        monthly_forecast = {}
        monthly_confidence = {}
        uncertainty_ranges = {}
        monthly_details = []

        # Month names for display
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        values = []

        for i in range(12):
            # Calculate current month and year in the forecast period
            current_offset = i
            current_month = ((start_month - 1 + current_offset) % 12) + 1
            current_year = start_year + ((start_month - 1 + current_offset) // 12)

            # Get consumption from pattern (pattern is already in month order 1-12)
            pattern_month = current_month  # The pattern value for this month number
            consumption = base_pattern[pattern_month]
            values.append(consumption)
            monthly_forecast[current_month] = consumption

            # Calculate confidence for this month
            if current_month in user_months:
                monthly_confidence[current_month] = 0.9
            else:
                # Distance from nearest user month (in month space, not accounting for year)
                if user_months:
                    # Calculate distance considering month wraparound
                    distances = []
                    for um in user_months.keys():
                        # Simple month difference (1-12 scale)
                        diff = abs(current_month - um)
                        # Consider wrap-around
                        diff = min(diff, 12 - diff)
                        distances.append(diff)
                    min_distance = min(distances)
                    distance_factor = np.exp(-min_distance / 6)
                    monthly_confidence[current_month] = confidence * distance_factor
                else:
                    monthly_confidence[current_month] = confidence * 0.7

            # Calculate uncertainty ranges
            median = consumption
            iqr = pattern['monthly_iqr'][pattern_month]
            uncertainty_ranges[current_month] = {
                'lower_bound': max(0, median - iqr),
                'upper_bound': median + iqr,
                'confidence_interval': 0.5
            }

            # Create detailed month info
            monthly_details.append({
                'month': current_month,
                'month_name': month_names[current_month - 1],
                'year': current_year,
                'consumption_kwh': round(consumption, 1),
                'confidence': monthly_confidence[current_month],
                'season': self._get_sri_lanka_season(current_month),
                'is_holiday_month': current_month in self.config['features']['sri_lanka']['holiday_months'],
                'seasonal_factor': consumption / np.mean(list(base_pattern.values())) if np.mean(
                    list(base_pattern.values())) > 0 else 1.0
            })

        # Calculate statistics
        annual_total = sum(values)
        annual_avg = annual_total / 12
        peak_month = max(range(1, 13), key=lambda m: monthly_forecast[m])
        low_month = min(range(1, 13), key=lambda m: monthly_forecast[m])

        # Calculate trend (using the reordered values)
        x = np.arange(12)
        if len(values) > 1:
            slope, _ = np.polyfit(x, values, 1)
            if slope > 2:
                trend = 'increasing'
            elif slope < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        forecast_stats = {
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': peak_month,
            'peak_consumption': monthly_forecast[peak_month],
            'low_month': low_month,
            'low_consumption': monthly_forecast[low_month],
            'trend': trend,
            'monthly_variability': float(np.std(values)),
            'seasonality_index': np.std(values) / annual_avg if annual_avg > 0 else 0,
            'overall_confidence': np.mean(list(monthly_confidence.values())),
            'forecast_period': {
                'start': f"{month_names[start_month - 1]} {start_year}",
                'end': f"{month_names[((start_month + 10) % 12)]} {start_year + (1 if start_month > 1 else 0)}"
            }
        }

        result = {
            'forecast': {
                'monthly_values': monthly_forecast,
                'monthly_confidence': monthly_confidence,
                'monthly_details': monthly_details,
                'uncertainty_ranges': uncertainty_ranges,
                'statistics': forecast_stats
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'user_months_provided': len(user_months),
                'user_last_month': f"{last_user_month}/{last_user_year}",
                'forecast_start': f"{month_names[start_month - 1]} {start_year}",
                'pattern_confidence': confidence,
                'pattern_quality_score': pattern['data_quality_score'],
                'similar_households_count': pattern_result['metadata']['similar_households_count'],
                'total_data_points': pattern_result['metadata']['total_data_points'],
                'forecast_method': 'pattern_based_safe',
                'notes': 'Pattern-based forecast with no look-ahead bias'
            }
        }

        return result

    def _get_sri_lanka_season(self, month: int) -> str:
        # Get Sri Lankan season for month
        if month in [12, 1, 2]:
            return "NE Monsoon"
        elif month in [3, 4]:
            return "Dry Season"
        elif month in [5, 6, 7, 8, 9]:
            return "SW Monsoon"
        else:  # 10, 11
            return "Dry Season"

    def _create_default_pattern(self) -> Dict:
        # Create default Sri Lanka consumption pattern
        monthly_median = {}
        for month in range(1, 13):
            monthly_median[month] = 350 * self.seasonal_factors[month]

        monthly_iqr = {m: 80 for m in range(1, 13)}
        monthly_min = {m: monthly_median[m] * 0.7 for m in range(1, 13)}
        monthly_max = {m: monthly_median[m] * 1.3 for m in range(1, 13)}

        annual_total = sum(monthly_median.values())
        annual_avg = annual_total / 12

        pattern = {
            'monthly_median': monthly_median,
            'monthly_iqr': monthly_iqr,
            'monthly_min_reasonable': monthly_min,
            'monthly_max_reasonable': monthly_max,
            'data_points_per_month': {m: 0 for m in range(1, 13)},
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': 4,
            'peak_consumption': monthly_median[4],
            'data_quality_score': 0.3
        }

        result = {
            'pattern': pattern,
            'confidence': 0.3,
            'metadata': {
                'similar_households_count': 0,
                'average_similarity': 0,
                'total_data_points': 0,
                'months_with_sufficient_data': 0,
                'data_quality_score': 0.3,
                'sample_households': [],
                'extraction_method': 'default',
                'notes': 'Default pattern when no similar households found'
            }
        }

        return result