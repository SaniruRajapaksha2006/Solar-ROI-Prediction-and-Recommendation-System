import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ElectricityDataLoader:
    def __init__(self, config: Dict):
        #Initialize the data loader with configuration

        self.config = config
        self.df = None
        self.customer_profiles = {}
        self.transformer_locations = {}

        # Get column names from config
        self.cols = config['data']['columns']

    def load_dataset(self, force_reload: bool = False) -> pd.DataFrame:
        #Load the master dataset from CSV file

        # If already loaded in memory, return cached version
        if self.df is not None and not force_reload:
            logger.info("Using cached dataset in memory")
            return self.df

        # Load from CSV
        dataset_path = self.config['data']['dataset_path']
        logger.info(f"Loading dataset from CSV: {dataset_path}")

        try:
            self.df = pd.read_csv(dataset_path)
            logger.info(f"CSV loaded: {len(self.df)} rows")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

        # Validate and clean
        self._validate_columns()
        self._clean_data()

        # Create in-memory profiles
        self._create_customer_profiles()

        logger.info(f"Dataset loaded: {len(self.df)} rows, {self.df['ACCOUNT_NO'].nunique()} unique accounts")
        return self.df

    def _validate_columns(self) -> None:
        """Validate that all required columns exist"""
        required_cols = [
            self.cols['account_no'],
            self.cols['month'],
            self.cols['net_consumption'],
            self.cols['customer_lat'],
            self.cols['customer_lon'],
            self.cols['tariff'],
            self.cols['phase']
        ]

        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")