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

    def _calculate_data_quality_score(self, data_points: Dict[int, int],
                                     monthly_iqr: Dict[int, float],
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