import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """
    Time series-aware data splitter
    Ensures no future data leaks into training
    """

    def __init__(self, config: Dict):
        self.config = config
        self.test_size = config['training'].get('test_size', 0.2)
        self.val_size = config['training'].get('validation_size', 0.1)
        self.random_seed = config['training'].get('random_seed', 42)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Split data into train, validation, test sets
        # Ensure sorted
        df = df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)

        # Create continuous time index
        df['time_idx'] = df['YEAR'] * 12 + df['MONTH']
        unique_times = df['time_idx'].unique()

        n_times = len(unique_times)
        n_test = int(n_times * self.test_size)
        n_val = int(n_times * self.val_size)
        n_train = n_times - n_test - n_val

        # Split indices
        train_times = unique_times[:n_train]
        val_times = unique_times[n_train:n_train + n_val]
        test_times = unique_times[n_train + n_val:]

        # Split data
        train_df = df[df['time_idx'].isin(train_times)].copy()
        val_df = df[df['time_idx'].isin(val_times)].copy()
        test_df = df[df['time_idx'].isin(test_times)].copy()

        logger.info(f"Train period: {train_times[0]} to {train_times[-1]} ({len(train_times)} months)")
        logger.info(f"Val period: {val_times[0]} to {val_times[-1]} ({len(val_times)} months)")
        logger.info(f"Test period: {test_times[0]} to {test_times[-1]} ({len(test_times)} months)")

        return train_df, val_df, test_df

    def split_by_account(self, df: pd.DataFrame) -> Dict[str, Tuple]:
        # Split data by account, maintaining temporal order per account

        result = {}

        for account in df['ACCOUNT_NO'].unique():
            account_data = df[df['ACCOUNT_NO'] == account].sort_values(['YEAR', 'MONTH'])

            if len(account_data) < 24:  # Need at least 2 years
                continue

            n = len(account_data)
            n_test = int(n * self.test_size)
            n_val = int(n * self.val_size)

            train_idx = account_data.index[:n - n_test - n_val].tolist()
            val_idx = account_data.index[n - n_test - n_val:n - n_test].tolist()
            test_idx = account_data.index[n - n_test:].tolist()

            result[account] = {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }

        return result

    def walk_forward_split(self, df: pd.DataFrame,
                          n_windows: int = 5,
                          window_size: int = 12) -> Generator:
        # Generate walk-forward validation splits

        df = df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
        df['time_idx'] = df['YEAR'] * 12 + df['MONTH']
        unique_times = df['time_idx'].unique()

        for i in range(n_windows):
            train_end = (i + 1) * window_size
            test_start = train_end
            test_end = min(test_start + window_size, len(unique_times))

            if test_end <= test_start:
                break

            train_times = unique_times[:train_end]
            test_times = unique_times[test_start:test_end]

            train_df = df[df['time_idx'].isin(train_times)]
            test_df = df[df['time_idx'].isin(test_times)]

            yield train_df, test_df

    def expanding_window_split(self, df: pd.DataFrame,
                              initial_train_size: int = 24,
                              step_size: int = 12) -> Generator:
        # Generate expanding window validation splits
        df = df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
        df['time_idx'] = df['YEAR'] * 12 + df['MONTH']
        unique_times = df['time_idx'].unique()

        current_size = initial_train_size

        while current_size + step_size <= len(unique_times):
            train_times = unique_times[:current_size]
            test_times = unique_times[current_size:current_size + step_size]

            train_df = df[df['time_idx'].isin(train_times)]
            test_df = df[df['time_idx'].isin(test_times)]

            yield train_df, test_df

            current_size += step_size

    def get_time_based_split(self, df: pd.DataFrame,
                            split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Split data based on specific date

        year, month = map(int, split_date.split('-'))
        split_time = year * 12 + month

        df = df.copy()
        df['time_idx'] = df['YEAR'] * 12 + df['MONTH']

        train_df = df[df['time_idx'] < split_time]
        test_df = df[df['time_idx'] >= split_time]

        logger.info(f"Split at {split_date}: Train {len(train_df)} rows, Test {len(test_df)} rows")

        return train_df, test_df


class TimeSeriesCrossValidator(BaseCrossValidator):
    # Custom time series cross-validator for scikit-learn compatibility
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        # Generate indices to split data
        n_samples = len(X)

        # Calculate fold sizes
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.gap
            test_end = min((i + 2) * fold_size, n_samples)

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits