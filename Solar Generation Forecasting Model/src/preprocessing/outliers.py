"""
src/preprocessing/outliers.py
IQR-based outlier removal on EXPORT_kWh, applied per month.
"""

import numpy as np
import pandas as pd

from utils.utils_config import get_outlier_config


MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


class OutlierRemover:

    def __init__(self, iqr_threshold: float = None):
        cfg = get_outlier_config()
        self.iqr_threshold = iqr_threshold if iqr_threshold is not None \
                             else cfg["iqr_threshold"]

    def remove(self, df: pd.DataFrame,
               column: str = "EXPORT_kWh") -> pd.DataFrame:
        """
        Flag and remove IQR outliers in `column` for each month independently.

        Using per-month IQR avoids penalising the seasonality pattern — summer
        months genuinely export more, so a global IQR would be too aggressive.

        Args:
            df     : DataFrame with Month and target column
            column : Column to detect outliers in (default: EXPORT_kWh)

        Returns:
            Cleaned DataFrame with outlier rows removed
        """
        print(f"\nOutlier removal on '{column}' (IQR × {self.iqr_threshold}) per month...")
        print("-" * 60)

        df_result     = df.copy()
        df_result["_outlier"] = False
        total_outliers = 0

        for month, group in df.groupby("Month"):
            Q1  = group[column].quantile(0.25)
            Q3  = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - self.iqr_threshold * IQR
            hi  = Q3 + self.iqr_threshold * IQR

            mask  = (group[column] < lo) | (group[column] > hi)
            count = mask.sum()
            total_outliers += count
            df_result.loc[group.index[mask], "_outlier"] = True

            label = MONTH_NAMES[month - 1]
            if count > 0:
                print(f"  {label}: {count}/{len(group)} outliers  "
                      f"bounds=[{lo:.0f}, {hi:.0f}] kWh")
            else:
                print(f"  {label}: 0/{len(group)}")

        before  = len(df_result)
        cleaned = df_result[~df_result["_outlier"]].drop(columns=["_outlier"]).copy()
        removed = before - len(cleaned)

        print(f"\n  Removed : {removed:,} ({removed / before * 100:.1f}%)")
        print(f"  Remaining: {len(cleaned):,}")
        print("-" * 60)
        return cleaned
