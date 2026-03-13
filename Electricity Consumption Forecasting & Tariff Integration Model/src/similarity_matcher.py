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