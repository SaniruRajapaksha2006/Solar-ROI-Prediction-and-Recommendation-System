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

    def find_similar_households_safe(self, user_data: Dict) -> List[Tuple[str, float]]:
        """
        Find households similar to the user
        No look-ahead bias - only uses months the user has
        """
        logger.info(f"Finding similar households for user at {user_data['latitude']}, {user_data['longitude']}")

        # Get user's last month
        user_months = user_data.get('consumption_months', {})
        last_user_month = max(user_months.keys()) if user_months else 12

        # Create cache key
        cache_key = self._create_cache_key(user_data)
        if cache_key in self._similarity_cache:
            logger.info("Returning cached similarity results")
            return self._similarity_cache[cache_key]

        # Step 1: Validate user location
        if not self.data_loader.validate_user_location(user_data['latitude'], user_data['longitude']):
            logger.warning(f"User location outside Sri Lanka bounds")
            # Still proceed, but log warning

        # Step 2: Get households within geographic radius
        max_distance_km = self.config['similarity']['max_distance_km']
        nearby_profiles = self.data_loader.get_profiles_by_location(
            user_data['latitude'], user_data['longitude'],
            radius_km=max_distance_km
        )

        if not nearby_profiles:
            logger.warning(f"No households found within {max_distance_km}km radius")
            return []

        logger.info(f"Found {len(nearby_profiles)} households within {max_distance_km}km")

        # Step 3: Calculate similarity scores with filtering
        similarity_scores = []
        filtered_count = 0
        filtered_reasons = {'profile': 0, 'consumption': 0, 'other': 0}

        for account_no, distance in nearby_profiles:
            profile = self.data_loader.get_customer_profile(account_no)
            if not profile:
                continue

            # Filter by profile compatibility
            if not self._filter_by_profile_compatibility(user_data, profile):
                filtered_count += 1
                filtered_reasons['profile'] += 1
                continue

            # Calculate similarity score using ONLY user's months
            score = self._calculate_similarity_score_safe(user_data, profile, distance, last_user_month)

            if score >= self.config['similarity']['min_similarity_score']:
                similarity_scores.append((account_no, score))
            else:
                filtered_count += 1
                filtered_reasons['other'] += 1

        # Log filtering statistics
        if filtered_count > 0:
            logger.info(f"Filtering statistics:")
            logger.info(f"  • Total filtered: {filtered_count} households")
            logger.info(f"  • Profile incompatibility: {filtered_reasons['profile']}")
            logger.info(f"  • Low similarity score: {filtered_reasons['other']}")

        # Step 4: Sort by similarity score (highest first)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Return top N similar households
        top_n = self.config['similarity']['top_n_similar']
        top_matches = similarity_scores[:top_n]

        # Cache results
        self._similarity_cache[cache_key] = top_matches

        if top_matches:
            logger.info(f"Found {len(top_matches)} similar households after filtering")
            logger.info(f"Top similarity score: {top_matches[0][1]:.3f}")
        else:
            logger.info("No households met minimum similarity threshold")

        return top_matches

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

    def _calculate_similarity_score_safe(self, user_data: Dict, profile: Dict,
                                        distance: float, last_user_month: int) -> float:
        #Calculate weighted similarity score using ONLY user's months

        scores = {}

        # 1. Geographic similarity
        max_distance = self.config['similarity']['max_distance_km'] * 1000
        geo_score = max(0, 1 - (distance / max_distance))
        scores['geographic'] = geo_score

        # 2. Consumption pattern similarity (using ONLY user's months)
        if 'consumption_months' in user_data and user_data['consumption_months']:
            pattern_score = self._calculate_consumption_similarity_safe(
                user_data['consumption_months'],
                profile['monthly_pattern'],
                last_user_month
            )
        else:
            pattern_score = 0.5
        scores['consumption_pattern'] = pattern_score

        # 3. Technical similarity
        tech_score = self._calculate_technical_similarity(user_data, profile)
        scores['technical'] = tech_score

        # 4. Seasonal compatibility
        seasonal_score = self._calculate_seasonal_similarity_safe(user_data, profile)
        scores['seasonal_compatibility'] = seasonal_score

        # Calculate weighted average
        total_score = 0
        for category, weight in self.similarity_weights.items():
            if category in scores:
                total_score += scores[category] * weight

        return min(1.0, total_score)

    def _calculate_technical_similarity(self, user_data: Dict, profile: Dict) -> float:
        #Calculate technical similarity (static features - safe)

        score = 0
        total_weight = 0

        # 1. Solar status
        if user_data.get('has_solar', 0) == profile['has_solar']:
            score += 0.4
        total_weight += 0.4

        # 2. Tariff
        if user_data.get('tariff', 'D1') == profile['tariff']:
            score += 0.3
        total_weight += 0.3

        # 3. Phase
        if user_data.get('phase', 'SP') == profile['phase']:
            score += 0.2
        total_weight += 0.2

        # 4. Data quality bonus
        quality_score = self._calculate_data_quality_score(profile)
        score += quality_score * 0.1
        total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0.3

    def _calculate_seasonal_similarity_safe(self, user_data: Dict, profile: Dict) -> float:
        #Calculate seasonal pattern similarity using ONLY user's months

        if 'consumption_months' not in user_data:
            return 0.5

        user_months = user_data['consumption_months']
        if len(user_months) < 2:
            return 0.5

        # Get user's months sorted
        months = sorted(user_months.keys())
        user_values = [user_months[m] for m in months]

        # Get profile values for SAME months only
        profile_values = []
        for month in months:
            if month in profile['monthly_pattern']:
                profile_values.append(profile['monthly_pattern'][month]['net_consumption'])
            else:
                # Month not in profile - can't compare
                return 0.5

        if len(profile_values) < 2:
            return 0.5

        # Calculate trends using only these months
        user_trend = np.polyfit(months, user_values, 1)[0]
        profile_trend = np.polyfit(months, profile_values, 1)[0]

        # Compare trends
        trend_diff = abs(user_trend - profile_trend)
        max_expected_trend = max(abs(user_trend), abs(profile_trend), 50)

        similarity = 1 - (trend_diff / max_expected_trend)
        return max(0, min(1, similarity))

    def _calculate_consumption_similarity_safe(self, user_months: Dict,
                                               profile_pattern: Dict,
                                               last_user_month: int) -> float:
        #Calculate consumption pattern similarity using BOTH shape AND magnitude
        # Find overlapping months (user's months only)
        common_months = [m for m in user_months.keys()
                         if m in profile_pattern and m <= last_user_month]

        if not common_months or len(common_months) < 2:
            return 0.5

        # Extract consumption values for overlapping months
        user_values = [user_months[m] for m in common_months]
        profile_values = [profile_pattern[m]['net_consumption'] for m in common_months]

        # Calculate magnitude similarity (how close the actual values are)
        user_avg = np.mean(user_values)
        profile_avg = np.mean(profile_values)

        # STRICTER magnitude scoring
        ratio = user_avg / profile_avg if profile_avg > 0 else 1.0

        # Much stricter ranges to find households with similar consumption
        if 0.95 <= ratio <= 1.05:  # Within 5% - perfect match
            magnitude_score = 1.0
        elif 0.9 <= ratio <= 1.1:  # Within 10% - good match
            magnitude_score = 0.8
        elif 0.85 <= ratio <= 1.15:  # Within 15% - moderate match
            magnitude_score = 0.5
        elif 0.8 <= ratio <= 1.2:  # Within 20% - weak match
            magnitude_score = 0.2
        else:  # Outside 20% - no match
            magnitude_score = 0.0

        # Shape similarity (correlation of normalized patterns)
        user_normalized = self._normalize_array(user_values)
        profile_normalized = self._normalize_array(profile_values)

        try:
            corr = 1 - correlation(user_normalized, profile_normalized)
            shape_score = (corr + 1) / 2
        except:
            shape_score = 0.5

        # Combine shape and magnitude (80% magnitude, 20% shape)
        # This strongly prioritizes consumption level over pattern shape
        similarity = (magnitude_score * 0.8) + (shape_score * 0.2)

        return max(0, min(1, similarity))

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

    def get_similarity_breakdown(self, user_data: Dict, account_no: str,
                                distance: float) -> Dict[str, Any]:
        #Get detailed similarity breakdown for a specific household

        profile = self.data_loader.get_customer_profile(account_no)
        if not profile:
            return {}

        breakdown = {
            'geographic': {
                'score': max(0, 1 - (distance / (self.config['similarity']['max_distance_km'] * 1000))),
                'weight': self.similarity_weights.get('geographic', 0.25),
                'distance_m': distance
            },
            'consumption_pattern': {
                'score': self._calculate_consumption_similarity_safe(
                user_data.get('consumption_months', {}),
                profile['monthly_pattern'],
                max(user_data.get('consumption_months', {}).keys()) if user_data.get('consumption_months', {}) else 12
            ),
            'weight': self.similarity_weights.get('consumption_pattern', 0.40)
        },
            'technical': {
                'score': self._calculate_technical_similarity(user_data, profile),
                'weight': self.similarity_weights.get('technical', 0.20)
            },
            'seasonal_compatibility': {
                'score': self._calculate_seasonal_similarity_safe(user_data, profile),
                'weight': self.similarity_weights.get('seasonal_compatibility', 0.15)
            }
        }

        # Calculate total weighted score
        total_score = 0
        for category, cat_data in breakdown.items():
            if 'score' in cat_data and 'weight' in cat_data:
                total_score += cat_data['score'] * cat_data['weight']

        breakdown['total_score'] = total_score
        breakdown['profile_info'] = {
            'account_no': account_no,
            'tariff': profile['tariff'],
            'phase': profile['phase'],
            'has_solar': profile['has_solar'],
            'annual_consumption': profile['annual_stats']['total'],
            'monthly_average': profile['annual_stats']['average'],
            'data_quality_score': self._calculate_data_quality_score(profile)
        }

        return breakdown

    def clear_cache(self):
        """Clear similarity cache"""
        self._similarity_cache.clear()
        logger.info("Similarity cache cleared")