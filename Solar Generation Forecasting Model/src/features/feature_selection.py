import numpy as np
import pandas as pd


NON_FEATURE_COLS = ["ACCOUNT_NO", "EXPORT_kWh", "Efficiency"]

COLS_TO_DROP = [
    "TRANSFORMER_CODE", "TRANSFORMER_LAT", "TRANSFORMER_LON",
    "CUSTOMER_LAT", "CUSTOMER_LON", "DISTANCE_FROM_TF_M",
    "DATA_QUALITY", "SOURCE", "CAL_TARIFF", "PHASE", "YEAR",
    "HAS_SOLAR", "NET_CONSUMPTION_kWh", "IMPORT_kWh",
    "Total_Generation_kWh",
    "Max_Temperature",   # correlated with Temperature
    "Min_Temperature",   # correlated with Temperature
    "Clear_Sky_GHI",     # correlated with Solar_Irradiance_GHI
]

MODEL_FEATURES = [
    "Month", "INV_CAPACITY",
    "Solar_Irradiance_GHI", "Temperature", "Humidity",
    "Precipitation", "Wind_Speed",
    "Temp_Efficiency", "Temp_Range", "GHI_Adjusted", "Cloud_Factor",
    "Month_Sin", "Month_Cos", "Days_In_Month",
    "Expected_Generation", "Physics_Pred",
]


class FeatureSelector:

    def __init__(self):
        self._dropped_non_informative: list[str] = []
        self._dropped_low_corr: list[str] = []

    def select_features(self, df: pd.DataFrame,
                        correlation_threshold: float = 0.05) -> pd.DataFrame:
        """
        Drop non-informative columns then drop low-correlation columns.
        NON_FEATURE_COLS are always kept regardless of correlation.

        Args:
            df                     : Post-engineering DataFrame (Efficiency must exist)
            correlation_threshold  : Min |r| with Efficiency to keep a feature

        Returns:
            DataFrame with MODEL_FEATURES + NON_FEATURE_COLS (those present)
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)

        initial = df.shape[1]
        df = self._drop_non_informative(df)
        df = self._drop_low_correlation(df, threshold=correlation_threshold)

        print(f"\n  Initial : {initial} columns")
        print(f"  Dropped : {initial - df.shape[1]}")
        print(f"  Final   : {df.shape[1]} columns")
        print(f"\n  Remaining columns:")
        for col in df.columns:
            print(f"    • {col}")
        print("=" * 60)
        return df

    # -- Private helpers --------------------------------------------------------

    def _drop_non_informative(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in COLS_TO_DROP if c in df.columns]
        if not to_drop:
            print("  No non-informative columns to drop")
            return df
        self._dropped_non_informative = to_drop
        print(f"  Dropping {len(to_drop)} non-informative columns:")
        for c in to_drop:
            print(f"    • {c}")
        return df.drop(columns=to_drop)

    def _drop_low_correlation(self, df: pd.DataFrame,
                               threshold: float = 0.05) -> pd.DataFrame:
        if "Efficiency" not in df.columns:
            print("  Efficiency column not found - skipping correlation filter")
            return df

        exclude = set(NON_FEATURE_COLS + ["Month", "INV_CAPACITY"])
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        to_check = [c for c in numeric if c not in exclude]

        if not to_check:
            return df

        corr     = df[to_check + ["Efficiency"]].corr()["Efficiency"].abs()
        low_corr = [c for c in to_check if corr.get(c, 1.0) < threshold and c != "Efficiency"]

        if not low_corr:
            print(f"  All features pass |r| ≥ {threshold}")
            return df

        self._dropped_low_corr = low_corr
        print(f"\n  Dropping {len(low_corr)} low-correlation features (|r| < {threshold}):")
        for c in low_corr:
            print(f"    • {c}  r={corr[c]:.4f}")
        return df.drop(columns=low_corr)
