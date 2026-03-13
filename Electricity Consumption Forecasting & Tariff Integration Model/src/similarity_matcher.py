import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import correlation
from sklearn.preprocessing import StandardScaler
import hashlib
import json
from functools import lru_cache

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """
    Finds similar households using weighted similarity scoring
    """

    def __init__(self, data_loader, config: Dict):
        #Initialize similarity matcher

        self.data_loader = data_loader
        self.config = config
        self.similarity_weights = config['similarity']['weights']

        # Validate weights sum to 1.0
        weight_sum = sum(self.similarity_weights.values())
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            raise ValueError(f"Similarity weights must sum to 1.0, got {weight_sum}")

        # Cache for similarity calculations
        self._similarity_cache = {}

        logger.info(f"Similarity matcher initialized with weights: {self.similarity_weights}")

    def _create_cache_key(self, user_data: Dict) -> str:
        """Create cache key from user data"""
        # Only use fields that affect similarity
        key_dict = {
            'lat': round(user_data['latitude'], 4),
            'lon': round(user_data['longitude'], 4),
            'months': user_data.get('consumption_months', {}),
            'tariff': user_data.get('tariff', 'D1'),
            'phase': user_data.get('phase', 'SP'),
            'has_solar': user_data.get('has_solar', 0)
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear similarity cache"""
        self._similarity_cache.clear()
        logger.info("Similarity cache cleared")

    def _filter_by_profile_compatibility(self, user_data: Dict, profile: Dict) -> bool:
        #Filter households based on profile compatibility
        # 1. Solar compatibility
        user_has_solar = user_data.get('has_solar', 0)
        profile_has_solar = profile['has_solar']

        if user_has_solar != profile_has_solar:
            logger.debug(f"Filtered - solar mismatch: user={user_has_solar}, profile={profile_has_solar}")
            return False

        # 2. Tariff consistency (very important)
        user_tariff = user_data.get('tariff', 'D1')
        profile_tariff = profile['tariff']

        if user_tariff != profile_tariff:
            logger.debug(f"Filtered - tariff mismatch: {user_tariff} vs {profile_tariff}")
            return False

        # 3. Data quality check
        # Count months with data for this household
        months_with_data = sum(1 for month in range(1, 13)
                              if month in profile['monthly_pattern'])

        if months_with_data < 6:
            logger.debug(f"Filtered - insufficient data: {months_with_data} months")
            return False

        return True

    def _calculate_data_quality_score(self, profile: Dict) -> float:
        #Calculate data quality score for a household

        score = 1.0

        # Check data completeness
        months_with_data = sum(1 for month in range(1, 13)
                              if month in profile['monthly_pattern'])

        if months_with_data < 12:
            score *= months_with_data / 12

        # Check for zero consumption months
        zero_count = 0
        for month in range(1, 13):
            if month in profile['monthly_pattern']:
                if profile['monthly_pattern'][month]['net_consumption'] < 10:
                    zero_count += 1

        if zero_count > 0:
            score *= (12 - zero_count) / 12

        return min(1.0, score)

    @staticmethod
    def _normalize_array(arr: List[float]) -> List[float]:
        #Normalize array to 0-1 range
        if not arr:
            return []

        arr_min = min(arr)
        arr_max = max(arr)

        if arr_max == arr_min:
            return [0.5] * len(arr)

        return [(x - arr_min) / (arr_max - arr_min) for x in arr]