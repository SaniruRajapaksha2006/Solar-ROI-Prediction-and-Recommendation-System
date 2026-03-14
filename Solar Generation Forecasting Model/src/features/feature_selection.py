import numpy as np
import pandas as pd


# NON_FEATURE_COLS  -- stay in final.csv but are NOT model inputs
#   ACCOUNT_NO  : used for GroupKFold split, dropped from X in trainer
#   EXPORT_kWh  : used to compute Efficiency, dropped from X in trainer
#   Efficiency  : the training target (y)
#
# COLS_TO_DROP  -- removed entirely during feature selection
#   Everything here is either redundant, non-informative, or
#   highly correlated with a kept column.
#
# MODEL_FEATURES  -- exact columns fed to pipeline.fit() / pipeline.predict()
#   Must match in both model_trainer.py and predict.py.
# ------------------------------------------------------------------------------

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
        self.dropped_cols = {
            'non_informative': [],
            'low_correlation': []
        }

    def drop_non_informative(self, df, non_informative_cols):
        if non_informative_cols is None:
            print(" No non-informative columns to drop")
            return df

        existing = [col for col in non_informative_cols if col in df.columns]

        if not existing:
            print(" No non-informative columns to drop")
            return df

        # Drop
        df_clean = df.drop(columns=existing)
        self.dropped_cols['non_informative'] = existing

        print(f"  Dropped {len(existing)} columns:")
        for col in existing:
            print(f"    • {col}")

        print("-" * 60)

        return df_clean


    def drop_low_correlation(self, df, target='EXPORT_kWh',
                             threshold=0.05, exclude_cols=None):
        print(f"\nDropping features with |correlation| < {threshold}...")
        print("-" * 60)

        if target not in df.columns:
            print(f" Target column '{target}' not found")
            return df

        # Default exclude columns (keep regardless of correlation)
        if exclude_cols is None:
            exclude_cols = ['Month', target]

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        check_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not check_cols:
            print(" No columns to check")
            return df

        # Calculate correlations
        correlations = df[check_cols + [target]].corr()[target].abs()

        # Find low correlation columns
        low_corr = correlations[correlations < threshold].index.tolist()
        low_corr = [col for col in low_corr if col != target]

        if not low_corr:
            print(f" All features have |correlation| >= {threshold}")
            return df

        # Show correlations before dropping
        print(f"  Low correlation features:")
        for col in low_corr:
            corr = correlations[col]
            print(f"    • {col}: {corr:.4f}")

        # Drop
        df_clean = df.drop(columns=low_corr)
        self.dropped_cols['low_correlation'] = low_corr

        print(f"\n  Dropped {len(low_corr)} low-correlation features")
        print("-" * 60)

        return df_clean


    def select_features(self, df: pd.DataFrame,
                        correlation_threshold: float = 0.05):
        """
        Complete feature selection using module-level constants.
        
        Args:
            df: DataFrame (post feature engineering, with Efficiency already created)
            correlation_threshold: Min |correlation| with Efficiency to keep a column

        Returns:
            DataFrame containing MODEL_FEATURES + NON_FEATURE_COLS (that exist)
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)

        initial_cols = df.shape[1]

        # Step 1: Drop non-informative columns
        df_clean = self.drop_non_informative(df, COLS_TO_DROP)

        # Step 2: Drop low correlation -- exclude NON_FEATURE_COLS + Month from check
        exclude_from_corr = NON_FEATURE_COLS + ["Month", "INV_CAPACITY"]
        df_final = self.drop_low_correlation(
            df_clean,
            target="Efficiency",
            threshold=correlation_threshold,
            exclude_cols=exclude_from_corr,
        )

        # Summary
        print("\n" + "=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)
        print(f"\nInitial columns : {initial_cols}")
        print(f"Final columns   : {df_final.shape[1]}")
        print(f"Dropped         : {initial_cols - df_final.shape[1]}")
        print(f"\nRemaining columns:")
        for col in df_final.columns:
            print(f"  • {col}")
        print("=" * 60)

        return df_final

    def get_dropped_summary(self):
        print("\n" + "=" * 60)
        print("DROPPED FEATURES SUMMARY")
        print("=" * 60)

        if self.dropped_cols['non_informative']:
            print(f"\nNon-informative ({len(self.dropped_cols['non_informative'])}):")
            for col in self.dropped_cols['non_informative']:
                print(f"  • {col}")

        if self.dropped_cols['low_correlation']:
            print(f"\nLow correlation ({len(self.dropped_cols['low_correlation'])}):")
            for col in self.dropped_cols['low_correlation']:
                print(f"  • {col}")

        print("=" * 60)