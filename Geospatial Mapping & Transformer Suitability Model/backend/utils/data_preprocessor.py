"""
data_preprocessor.py
Handles CSV loading, feature engineering, and data preparation for transformer data.

Key fix: INV_CAPACITY (solar inverter capacity) repeats across every monthly row
for the same customer. We deduplicate by (TRANSFORMER_CODE, ACCOUNT_NO) before
summing solar capacity — otherwise a 3 kW inverter across 12 months counts as 36 kW.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Loads raw transformer CSV and engineers features for ML and rule-based scoring.
    """

    # Assumed nameplate capacity per transformer (kW).
    DEFAULT_CAPACITY_KW = 100

    @staticmethod
    def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        print(f"📊 Raw CSV shape: {df.shape}")
        print(f"   Transformers: {df['TRANSFORMER_CODE'].nunique()}")
        print(f"   Unique customers: {df['ACCOUNT_NO'].nunique()}")

        # ── 1. Solar capacity — deduplicate per unique customer ─────────────
        # INV_CAPACITY repeats for every month a customer appears in the dataset.
        # Summing across all rows inflates solar by 12x (one per month).
        # Fix: take exactly one row per (transformer, customer) before summing.
        unique_customers = df.drop_duplicates(subset=['TRANSFORMER_CODE', 'ACCOUNT_NO'])

        solar_agg = unique_customers.groupby('TRANSFORMER_CODE').agg(
            total_solar_capacity=('INV_CAPACITY', 'sum'),
            solar_connections=('HAS_SOLAR', 'sum'),
        ).reset_index()

        # ── 2. Consumption — aggregate across monthly rows (correct as-is) ──
        consumption_agg = df.groupby('TRANSFORMER_CODE').agg(
            TRANSFORMER_LAT=('TRANSFORMER_LAT', 'first'),
            TRANSFORMER_LON=('TRANSFORMER_LON', 'first'),
            avg_consumption=('NET_CONSUMPTION_kWh', 'mean'),
            consumption_std=('NET_CONSUMPTION_kWh', 'std'),
            max_consumption=('NET_CONSUMPTION_kWh', 'max'),
            avg_import=('IMPORT_kWh', 'mean'),
            avg_export=('EXPORT_kWh', 'mean'),
            num_customers=('ACCOUNT_NO', 'nunique'),
            avg_distance_from_tf=('DISTANCE_FROM_TF_M', 'mean'),
            data_points=('NET_CONSUMPTION_kWh', 'count'),
        ).reset_index()

        # ── 3. Merge ─────────────────────────────────────────────────────────
        agg = consumption_agg.merge(solar_agg, on='TRANSFORMER_CODE', how='left')
        agg['total_solar_capacity'] = agg['total_solar_capacity'].fillna(0)
        agg['solar_connections']    = agg['solar_connections'].fillna(0)

        # ── 4. Feature engineering ────────────────────────────────────────────
        cap = DataPreprocessor.DEFAULT_CAPACITY_KW
        agg['ESTIMATED_CAPACITY_kW'] = cap

        # avg_consumption = mean kWh/month per customer row
        # divide by 720 hours/month to get average kW
        agg['current_load_kW'] = (agg['avg_consumption'] / 720).fillna(0)

        agg['utilization_rate'] = (agg['current_load_kW'] / cap).clip(0, 1)

        # Headroom: NOT clipped — negative correctly flags overloaded transformers
        agg['available_headroom'] = cap - agg['current_load_kW'] - agg['total_solar_capacity']

        agg['solar_penetration'] = (agg['total_solar_capacity'] / cap).clip(0)
        agg['demand_volatility'] = (agg['consumption_std'] / 720).fillna(0)
        agg['export_ratio'] = (agg['avg_export'] / (agg['avg_import'] + 1)).fillna(0)

        agg = agg.fillna(0)

        print("\n✓ Transformer summary after preprocessing:")
        print(agg[['TRANSFORMER_CODE', 'num_customers', 'current_load_kW',
                    'total_solar_capacity', 'available_headroom']].to_string(index=False))
        return agg