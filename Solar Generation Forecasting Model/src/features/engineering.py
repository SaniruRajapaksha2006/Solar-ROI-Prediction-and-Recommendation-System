"""
All feature creation and the Efficiency training target.
"""

import numpy as np
import pandas as pd

from utils.utils_config import get_physics_constants


DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30,  5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


class FeatureEngineer:

    def __init__(self):
        self._created: list[str] = []
        phys = get_physics_constants()
        self._pr        = phys["performance_ratio"]   # 0.80
        self._tc        = phys["temp_coefficient"]    # -0.005
        self._t_ref     = phys["temp_reference"]      # 25.0

    # -- Individual feature groups ----------------------------------------------

    def _temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temp_Efficiency, Temp_Range, GHI_Adjusted."""
        df["Temp_Efficiency"] = (
            1 + (df["Temperature"] - self._t_ref) * self._tc
        ).round(3)
        df["Temp_Range"]   = (df["Max_Temperature"] - df["Min_Temperature"]).round(2)
        df["GHI_Adjusted"] = (df["Solar_Irradiance_GHI"] * df["Temp_Efficiency"]).round(2)
        self._created.extend(["Temp_Efficiency", "Temp_Range", "GHI_Adjusted"])
        print(" Temp_Efficiency, Temp_Range, GHI_Adjusted")
        return df

    def _cloud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cloud_Factor = actual GHI / clear-sky GHI."""
        df["Cloud_Factor"] = (
            df["Solar_Irradiance_GHI"] / df["Clear_Sky_GHI"].replace(0, np.nan)
        ).fillna(1.0).round(2)
        self._created.append("Cloud_Factor")
        print(" Cloud_Factor")
        return df

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Month_Sin, Month_Cos (cyclical encoding), Days_In_Month."""
        df["Month_Sin"]     = np.sin(2 * np.pi * df["Month"] / 12).round(4)
        df["Month_Cos"]     = np.cos(2 * np.pi * df["Month"] / 12).round(4)
        df["Days_In_Month"] = df["Month"].map(DAYS_IN_MONTH)
        self._created.extend(["Month_Sin", "Month_Cos", "Days_In_Month"])
        print(" Month_Sin, Month_Cos, Days_In_Month")
        return df

    def _system_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expected_Generation : GHI × capacity × PR × days  [kWh]
            Physics-informed feature that guides model scale.

        Physics_Pred : GHI × PR × days  [kWh/kW]
            Deterministic baseline in Efficiency units.
            Used by baseline.py to benchmark ML improvement.
        """
        df["Expected_Generation"] = (
            df["Solar_Irradiance_GHI"] * df["INV_CAPACITY"] *
            self._pr * df["Days_In_Month"]
        ).round(2)
        df["Physics_Pred"] = (
            df["Solar_Irradiance_GHI"] * self._pr * df["Days_In_Month"]
        ).round(2)
        self._created.extend(["Expected_Generation", "Physics_Pred"])
        print(" Expected_Generation, Physics_Pred")
        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Efficiency = EXPORT_kWh / INV_CAPACITY  [kWh/kW]
        Training path only — only called when EXPORT_kWh is in the df.
        """
        df["Efficiency"] = (df["EXPORT_kWh"] / df["INV_CAPACITY"]).round(4)
        print(f" Efficiency  "
              f"mean={df['Efficiency'].mean():.2f}  "
              f"min={df['Efficiency'].min():.2f}  "
              f"max={df['Efficiency'].max():.2f}")
        return df

    # -- Public -----------------------------------------------------------------

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature groups in sequence.
        Creates Efficiency target automatically when EXPORT_kWh is present.
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        df = df.copy()
        df = self._temperature_features(df)
        df = self._cloud_features(df)
        df = self._temporal_features(df)
        df = self._system_features(df)

        if "EXPORT_kWh" in df.columns:
            df = self._create_target(df)

        print(f"\n  {len(self._created)} features created")
        print("=" * 60)
        return df

    @property
    def feature_names(self) -> list[str]:
        return list(self._created)
