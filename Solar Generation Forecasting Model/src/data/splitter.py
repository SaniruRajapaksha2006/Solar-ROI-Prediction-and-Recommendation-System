"""
Household-safe train/test split using GroupShuffleSplit.

Guarantees no account appears in both train and test sets —
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.features.selection import MODEL_FEATURES
from utils.utils_config import load_config


class DataSplitter:

    def __init__(self, test_size: float = None, random_state: int = None):
        cfg = load_config()["training"]
        self.test_size    = test_size    if test_size    is not None else cfg["test_size"]
        self.random_state = random_state if random_state is not None else cfg["random_state"]

    def split(self, df: pd.DataFrame,
              target: str = "Efficiency",
              group_col: str = "ACCOUNT_NO"):
        """
        Split df by household so no account appears in both train and test.

        Args:
            df        : Processed DataFrame (output of data_pipeline.py)
            target    : Target column name
            group_col : Column used to define groups (household ID)

        Returns:
            X_train, X_test, y_train, y_test, groups_train
            (groups_train is the ACCOUNT_NO series aligned to X_train rows
             — needed by GroupKFold inside tuner.py)
        """
        print("\n" + "=" * 60)
        print("HOUSEHOLD SPLIT  (GroupShuffleSplit)")
        print("=" * 60)

        # Only keep MODEL_FEATURES that actually exist in df
        feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
        missing = set(MODEL_FEATURES) - set(feature_cols)
        if missing:
            print(f"Missing features (will be absent from X): {missing}")

        X      = df[feature_cols]
        y      = df[target]
        groups = df[group_col]

        print(f"  Features : {X.shape[1]} columns")
        print(f"  Records  : {len(X):,}")
        print(f"  Accounts : {groups.nunique():,} unique")

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train    = groups.iloc[train_idx].reset_index(drop=True)

        # Verify no account overlap
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        if overlap:
            raise ValueError(f"Data leakage! {len(overlap)} accounts in both splits.")

        print(f"\n  Train : {len(X_train):,} records  |  "
              f"{groups.iloc[train_idx].nunique()} accounts")
        print(f"  Test  : {len(X_test):,} records  |  "
              f"{groups.iloc[test_idx].nunique()} accounts")
        print(f"No account overlap confirmed")

        return X_train, X_test, y_train, y_test, groups_train
