"""
src/preprocessing/outliers.py
IQR-based outlier removal on EXPORT_kWh, applied per month.
"""

import joblib
from pathlib import Path
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

    def fit(self, df: pd.DataFrame, column: str = "EXPORT_kWh") -> "OutlierRemover":
        """Compute IQR bounds per month from training data only."""
        self._bounds = {}
        self._column = column
        for month, group in df.groupby("Month"):
            Q1  = group[column].quantile(0.25)
            Q3  = group[column].quantile(0.75)
            IQR = Q3 - Q1
            self._bounds[month] = (Q1 - self.iqr_threshold * IQR,
                                   Q3 + self.iqr_threshold * IQR)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using fitted bounds. Call on train split only."""
        if not hasattr(self, "_bounds"):
            raise RuntimeError("Call fit() before transform().")
        col = self._column
        print(f"\nOutlier removal on '{col}' (IQR × {self.iqr_threshold}) per month...")
        print("-" * 60)
        df_result = df.copy()
        df_result["_outlier"] = False
        for month, group in df_result.groupby("Month"):
            if month not in self._bounds:
                continue
            lo, hi = self._bounds[month]
            mask   = (group[col] < lo) | (group[col] > hi)
            count  = mask.sum()
            df_result.loc[group.index[mask], "_outlier"] = True
            label  = MONTH_NAMES[month - 1]
            if count > 0:
                print(f"  {label}: {count}/{len(group)} outliers  bounds=[{lo:.0f}, {hi:.0f}] kWh")
            else:
                print(f"  {label}: 0/{len(group)}")
        before  = len(df_result)
        cleaned = df_result[~df_result["_outlier"]].drop(columns=["_outlier"]).copy()
        print(f"\n  Removed : {before - len(cleaned):,} ({(before - len(cleaned)) / before * 100:.1f}%)")
        print(f"  Remaining: {len(cleaned):,}")
        print("-" * 60)
        return cleaned

    def fit_transform(self, df: pd.DataFrame, column: str = "EXPORT_kWh") -> pd.DataFrame:
        """Fit on df and immediately remove outliers. Use on train split only."""
        return self.fit(df, column).transform(df)

    def save(self, path: str | Path) -> None:
        """Save fitted bounds"""
        if not hasattr(self, "_bounds"):
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"bounds": self._bounds, "column": self._column,
                     "iqr_threshold": self.iqr_threshold}, path)
        print(f"  OutlierRemover saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "OutlierRemover":
        """Load a previously fitted OutlierRemover"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OutlierRemover not found: {path}")
        data = joblib.load(path)
        remover = cls(iqr_threshold=data["iqr_threshold"])
        remover._bounds = data["bounds"]
        remover._column = data["column"]
        print(f"  OutlierRemover loaded: {path}")
        return remover
