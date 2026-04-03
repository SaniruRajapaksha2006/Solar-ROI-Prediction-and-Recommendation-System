"""
Column lists and feature selector.
"""

import numpy as np
import pandas as pd

from utils.utils_config import get_feature_selection_config, load_config


def get_column_config() -> dict:
    """Return the feature_selection section from config.yaml."""
    return get_feature_selection_config()


def get_non_feature_cols() -> list[str]:
    return get_column_config()["non_feature_cols"]


def get_cols_to_drop() -> list[str]:
    return get_column_config()["cols_to_drop"]


def get_model_features() -> list[str]:
    return get_column_config()["model_features"]


# Module-level convenience — import MODEL_FEATURES from here like before
MODEL_FEATURES     = get_model_features()
NON_FEATURE_COLS   = get_non_feature_cols()
COLS_TO_DROP       = get_cols_to_drop()


class FeatureSelector:

    def __init__(self):
        self._dropped_non_informative: list[str] = []
        self._dropped_low_corr: list[str] = []

    def select_features(self, df: pd.DataFrame,
                        correlation_threshold: float = None) -> pd.DataFrame:
        """
        Two-pass feature selection:
          Pass 1 — Drop cols_to_drop (admin/redundant/collinear columns from config)
          Pass 2 — Drop any remaining columns with |correlation| < threshold

        NON_FEATURE_COLS (ACCOUNT_NO, EXPORT_kWh, Efficiency) are always kept.

        Args:
            df                    : Post-engineering DataFrame
            correlation_threshold : Override threshold (default: from config.yaml)
        """
        cfg = get_column_config()
        if correlation_threshold is None:
            correlation_threshold = cfg["correlation_threshold"]

        cols_to_drop     = cfg["cols_to_drop"]
        non_feature_cols = cfg["non_feature_cols"]

        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)

        initial = df.shape[1]

        # Pass 1: drop non-informative columns defined in config
        df = self._drop_listed(df, cols_to_drop)

        # Pass 2: correlation filter — skip non-feature cols and always-keep cols
        # df = self._drop_low_correlation(
        #     df,
        #     non_feature_cols=non_feature_cols,
        #     threshold=correlation_threshold,
        # )

        print(f"\n  {initial} → {df.shape[1]} columns  "
              f"({initial - df.shape[1]} dropped)")
        print(f"\n  Remaining:")
        for col in df.columns:
            print(f"    • {col}")
        print("=" * 60)
        return df

    # -- Private ---------------------------------------------------------------

    def _drop_listed(self, df: pd.DataFrame,
                     cols_to_drop: list[str]) -> pd.DataFrame:
        """Drop columns that are in the config drop list and exist in df."""
        present = [c for c in cols_to_drop if c in df.columns]
        if not present:
            print("  Pass 1: nothing to drop (already clean)")
            return df
        self._dropped_non_informative = present
        print(f"  Pass 1: dropping {len(present)} non-informative columns")
        for c in present:
            print(f"    • {c}")
        return df.drop(columns=present)

    def _drop_low_correlation(self, df: pd.DataFrame,
                               non_feature_cols: list[str],
                               threshold: float) -> pd.DataFrame:
        """Drop numeric columns with |r| < threshold against Efficiency."""
        if "Efficiency" not in df.columns:
            print("  Pass 2: skipped (Efficiency not present — inference path)")
            return df

        always_keep = set(non_feature_cols + ["Month", "INV_CAPACITY"])
        to_check    = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in always_keep
        ]

        if not to_check:
            return df

        corr     = df[to_check + ["Efficiency"]].corr()["Efficiency"].abs()
        low_corr = [
            c for c in to_check
            if c != "Efficiency" and corr.get(c, 1.0) < threshold
        ]

        if not low_corr:
            print(f"  Pass 2: all features pass |r| ≥ {threshold}")
            return df

        self._dropped_low_corr = low_corr
        print(f"  Pass 2: dropping {len(low_corr)} low-correlation columns "
              f"(|r| < {threshold})")
        for c in low_corr:
            print(f"    • {c}  r={corr[c]:.4f}")
        return df.drop(columns=low_corr)
