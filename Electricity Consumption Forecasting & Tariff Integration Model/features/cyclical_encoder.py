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

    def encode_day_of_week(self, day: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        # Encode day of week (0=Monday, 6=Sunday)

        if isinstance(day, (int, float)):
            days = np.array([day])
        else:
            days = np.array(day)

        radians = 2 * np.pi * days / 7

        result = {
            'dow_sin': np.sin(radians),
            'dow_cos': np.cos(radians)
        }

        if len(days) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def encode_hour(self, hour: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        # Encode hour (0-23)

        if isinstance(hour, (int, float)):
            hours = np.array([hour])
        else:
            hours = np.array(hour)

        radians = 2 * np.pi * hours / 24

        result = {
            'hour_sin': np.sin(radians),
            'hour_cos': np.cos(radians)
        }

        if len(hours) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def encode_day_of_month(self, day: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        # Encode day of month (1-31) - approximate

        if isinstance(day, (int, float)):
            days = np.array([day])
        else:
            days = np.array(day)

        # Normalize to 0-1 within month
        normalized = (days - 1) / 30
        radians = 2 * np.pi * normalized

        result = {
            'dom_sin': np.sin(radians),
            'dom_cos': np.cos(radians)
        }

        if len(days) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def encode_week_of_year(self, week: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        # Encode week of year (1-52)

        if isinstance(week, (int, float)):
            weeks = np.array([week])
        else:
            weeks = np.array(week)

        radians = 2 * np.pi * (weeks - 1) / 52

        result = {
            'week_sin': np.sin(radians),
            'week_cos': np.cos(radians)
        }

        if len(weeks) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def encode_quarter(self, quarter: Union[int, List[int], np.ndarray]) -> Dict[str, float]:
        # Encode quarter (1-4)

        if isinstance(quarter, (int, float)):
            quarters = np.array([quarter])
        else:
            quarters = np.array(quarter)

        radians = 2 * np.pi * (quarters - 1) / 4

        result = {
            'quarter_sin': np.sin(radians),
            'quarter_cos': np.cos(radians)
        }

        if len(quarters) == 1:
            result = {k: float(v[0]) for k, v in result.items()}

        return result

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Encode all cyclical features in a dataframe

        result = df.copy()

        if 'MONTH' in df.columns:
            encoded = self.encode_month(df['MONTH'].values)
            result['month_sin'] = encoded['month_sin']
            result['month_cos'] = encoded['month_cos']

        if 'DAY_OF_WEEK' in df.columns:
            encoded = self.encode_day_of_week(df['DAY_OF_WEEK'].values)
            result['dow_sin'] = encoded['dow_sin']
            result['dow_cos'] = encoded['dow_cos']

        if 'HOUR' in df.columns:
            encoded = self.encode_hour(df['HOUR'].values)
            result['hour_sin'] = encoded['hour_sin']
            result['hour_cos'] = encoded['hour_cos']

        if 'DAY' in df.columns:
            encoded = self.encode_day_of_month(df['DAY'].values)
            result['dom_sin'] = encoded['dom_sin']
            result['dom_cos'] = encoded['dom_cos']

        if 'WEEK' in df.columns:
            encoded = self.encode_week_of_year(df['WEEK'].values)
            result['week_sin'] = encoded['week_sin']
            result['week_cos'] = encoded['week_cos']

        if 'QUARTER' in df.columns:
            encoded = self.encode_quarter(df['QUARTER'].values)
            result['quarter_sin'] = encoded['quarter_sin']
            result['quarter_cos'] = encoded['quarter_cos']

        return result

    def decode_month(self, sin_val: float, cos_val: float) -> int:
        # Decode sin/cos back to month number

        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi

        month = int(angle * 12 / (2 * np.pi)) + 1
        return month

    def get_encoding_info(self) -> Dict:
        # Get information about encodings
        return {
            'month': {'period': 12, 'min': 1, 'max': 12},
            'day_of_week': {'period': 7, 'min': 0, 'max': 6},
            'hour': {'period': 24, 'min': 0, 'max': 23},
            'day_of_month': {'period': 30, 'min': 1, 'max': 31},
            'week_of_year': {'period': 52, 'min': 1, 'max': 52},
            'quarter': {'period': 4, 'min': 1, 'max': 4}
        }