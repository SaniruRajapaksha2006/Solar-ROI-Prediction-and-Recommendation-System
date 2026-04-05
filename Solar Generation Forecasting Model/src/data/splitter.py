"""
Household-safe train/test split using GroupShuffleSplit.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils.utils_config import load_config


class DataSplitter:

    def __init__(self, test_size: float = None, random_state: int = None):
        cfg = load_config()["training"]
        self.test_size    = test_size    if test_size    is not None else cfg["test_size"]
        self.random_state = random_state if random_state is not None else cfg["random_state"]

    def split(self, df: pd.DataFrame,
              group_col: str = "ACCOUNT_NO"):
        """
        Split df by household, returning full train/test DataFrames.

        Returns:
            df_train, df_test — full DataFrames
        """
        print("\n" + "=" * 60)
        print("HOUSEHOLD SPLIT  (GroupShuffleSplit)")
        print("=" * 60)

        groups = df[group_col]

        print(f"  Records  : {len(df):,}")
        print(f"  Accounts : {groups.nunique():,} unique")

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                random_state=self.random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))

        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        if overlap:
            raise ValueError(f"Data leakage! {len(overlap)} accounts in both splits.")

        print(f"  Train : {len(df_train):,} records  |  "
              f"{df_train[group_col].nunique()} accounts")
        print(f"  Test  : {len(df_test):,} records  |  "
              f"{df_test[group_col].nunique()} accounts")
        print("  No account overlap confirmed")

        return df_train, df_test
