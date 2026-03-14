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