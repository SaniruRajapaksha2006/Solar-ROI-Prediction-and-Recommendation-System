import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class CyclicalFeatureEncoder:
    # Encodes cyclical features (month, day of week, hour) using sin/cos


    def __init__(self, config: Dict):
        self.config = config
        self.encoding_info = {}

    def encode_month(self, month: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        if isinstance(month, (int, float)):
            months = np.array([month])
        else:
            months = np.array(month)

        # Convert to radians (0 to 2π)
        radians = 2 * np.pi * (months - 1) / 12

        result = {
            'month_sin': np.sin(radians),
            'month_cos': np.cos(radians)
        }

        # If single value, return scalars
        if len(months) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def decode_month(self, sin_val: float, cos_val: float) -> int:
        # Decode sin/cos back to month number

        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi

        month = int(angle * 12 / (2 * np.pi)) + 1
        return month