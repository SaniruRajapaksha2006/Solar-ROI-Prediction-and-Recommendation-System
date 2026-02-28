"""
data_preprocessor.py
Handles CSV loading, feature engineering, and data preparation for transformer data.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Loads raw transformer CSV and engineers features for ML and rule-based scoring.
    Dataset columns expected:
        TRANSFORMER_CODE, TRANSFORMER_LAT, TRANSFORMER_LON, ACCOUNT_NO,
        CUSTOMER_LAT, CUSTOMER_LON, MONTH, YEAR, IMPORT_kWh, EXPORT_kWh,
        NET_CONSUMPTION_kWh, HAS_SOLAR, INV_CAPACITY, CAL_TARIFF,
        PHASE, DATA_QUALITY, DISTANCE_FROM_TF_M, SOURCE
    """

    # Assumed capacity per transformer (kW) — can be overridden by config
    DEFAULT_CAPACITY_KW = 100

    @staticmethod
    def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
        """
        Load CSV and aggregate by TRANSFORMER_CODE to produce one row per transformer
        with engineered features ready for ML and scoring.

        Returns:
            pd.DataFrame with one row per transformer
        """
        df = pd.read_csv(csv_path)
        print(f"📊 Raw CSV shape: {df.shape}  |  Columns: {df.columns.tolist()}")

        # ── Aggregate numerical columns by transformer ──────────────────────
        agg = df.groupby('TRANSFORMER_CODE').agg(
            TRANSFORMER_LAT=('TRANSFORMER_LAT', 'first'),
            TRANSFORMER_LON=('TRANSFORMER_LON', 'first'),
            avg_consumption=('NET_CONSUMPTION_kWh', 'mean'),
            consumption_std=('NET_CONSUMPTION_kWh', 'std'),
            max_consumption=('NET_CONSUMPTION_kWh', 'max'),
            solar_connections=('HAS_SOLAR', 'sum'),
            total_solar_capacity=('INV_CAPACITY', 'sum'),
            avg_import=('IMPORT_kWh', 'mean'),
            avg_export=('EXPORT_kWh', 'mean'),
            num_customers=('ACCOUNT_NO', 'nunique'),
            avg_distance_from_tf=('DISTANCE_FROM_TF_M', 'mean'),
            data_points=('NET_CONSUMPTION_kWh', 'count'),
        ).reset_index()

        # ── Derived / engineered features ───────────────────────────────────
        cap = DataPreprocessor.DEFAULT_CAPACITY_KW

        agg['ESTIMATED_CAPACITY_kW'] = cap

        # Current load estimate: scale average monthly kWh → kW (30d×24h)
        agg['current_load_kW'] = (agg['avg_consumption'] / (30 * 24)).fillna(0)

        # Solar capacity already in kW (INV_CAPACITY)
        agg['total_solar_capacity'] = agg['total_solar_capacity'].fillna(0)

        # Utilisation rate (0–1+)
        agg['utilization_rate'] = (agg['current_load_kW'] / cap).clip(0, 2)

        # Available headroom before new solar (kW)
        agg['available_headroom'] = (
            cap - agg['current_load_kW'] - agg['total_solar_capacity']
        ).clip(0)

        # Solar penetration ratio (0–1+)
        agg['solar_penetration'] = (agg['total_solar_capacity'] / cap).clip(0)

        # Demand volatility (standard deviation of kW estimate)
        agg['demand_volatility'] = (agg['consumption_std'] / (30 * 24)).fillna(0)

        # Net export ratio — high export = more solar generation than consumption
        agg['export_ratio'] = (agg['avg_export'] / (agg['avg_import'] + 1)).fillna(0)

        agg = agg.fillna(0)
        print(f"✓ Prepared {len(agg)} transformer records")
        return agg