import numpy as np
import pandas as pd

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


    def drop_low_correlation(self, df, target='Total_Generation_kWh',
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


    def select_features(self, df:pd.DataFrame, target:str='Total_Generation_kWh',
                        correlation_threshold:float=0.05,
                        non_informative_cols:list[str]=None,
                        keep_cols:list[str]=None):
        """
        Complete feature selection

        Args:
            df: DataFrame
            target: Target column
            correlation_threshold: Min correlation to keep
            non_informative_cols: Custom non-informative columns
            keep_cols: Columns to keep regardless (eg: Month, target...)
            is_method_corr: whether to use method correlation or only keep keep_cols

        Returns:
            DataFrame with selected features
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)

        initial_cols = df.shape[1]

        # Step 1: Drop non-informative
        df_clean = self.drop_non_informative(df, non_informative_cols)

        # Step 2: Drop low correlation
        if keep_cols is None:
            keep_cols = ['Month', target]

        df_final = self.drop_low_correlation(
            df_clean,
            target=target,
            threshold=correlation_threshold,
            exclude_cols=keep_cols,
        )

        # Summary
        print("\n" + "=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)

        print(f"\nInitial columns: {initial_cols}")
        print(f"Final columns: {df_final.shape[1]}")
        print(f"Dropped: {initial_cols - df_final.shape[1]}")

        print(f"\nRemaining features:")
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