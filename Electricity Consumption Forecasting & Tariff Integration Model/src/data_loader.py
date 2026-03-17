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

    def _clean_data(self) -> None:
        # Clean and preprocess the dataset
        # Convert column names if needed
        self.df.columns = [str(col).strip() for col in self.df.columns]

        # Convert data types
        numeric_cols = [
            self.cols['customer_lat'],
            self.cols['customer_lon'],
            self.cols['net_consumption'],
            'INV_CAPACITY'
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Clean string columns
        string_cols = [
            self.cols['account_no'],
            self.cols['tariff'],
            self.cols['phase']
        ]

        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        # Filter out invalid locations (outside Sri Lanka bounds)
        bounds = self.config['similarity']['sri_lanka_bounds']
        lat_mask = (self.df[self.cols['customer_lat']] >= bounds['lat_min']) & \
                   (self.df[self.cols['customer_lat']] <= bounds['lat_max'])
        lon_mask = (self.df[self.cols['customer_lon']] >= bounds['lon_min']) & \
                   (self.df[self.cols['customer_lon']] <= bounds['lon_max'])

        self.df = self.df[lat_mask & lon_mask].copy()

        # Remove rows with missing consumption
        self.df = self.df[self.df[self.cols['net_consumption']].notna()]

        # Remove obviously erroneous values
        self.df = self.df[
            (self.df[self.cols['net_consumption']] >= 10) &
            (self.df[self.cols['net_consumption']] <= 2000)
        ]

        logger.info(f"After cleaning: {len(self.df)} rows remaining")

    def _create_customer_profiles(self):
        # Create in-memory customer profiles from the dataframe
        logger.info("Creating customer profiles...")

        # Group by account number
        for account in self.df[self.cols['account_no']].unique():
            account_data = self.df[self.df[self.cols['account_no']] == account]
            self.customer_profiles[account] = self._create_profile(account_data)

        logger.info(f"Created profiles for {len(self.customer_profiles)} customers")

    def _create_profile(self, account_data: pd.DataFrame) -> Dict:
        # Create a profile for a single household
        account_data = account_data.sort_values(self.cols['month'])
        first_row = account_data.iloc[0]

        # Create monthly pattern
        monthly_pattern = {}
        for _, row in account_data.iterrows():
            month = int(row[self.cols['month']])
            monthly_pattern[month] = {
                'net_consumption': float(row[self.cols['net_consumption']]),
                'import': float(row[self.cols['import_kwh']]) if self.cols['import_kwh'] in row else None,
                'export': float(row[self.cols['export_kwh']]) if self.cols['export_kwh'] in row else 0
            }

        # Calculate annual statistics
        net_values = [monthly_pattern[m]['net_consumption'] for m in monthly_pattern]

        return {
            'account_no': first_row[self.cols['account_no']],
            'transformer_code': first_row.get(self.cols['transformer_code'], 'UNKNOWN'),
            'has_solar': int(first_row.get(self.cols['has_solar'], 0)),
            'inv_capacity': float(first_row.get('INV_CAPACITY', 0)),
            'tariff': first_row.get(self.cols['tariff'], 'D1'),
            'phase': first_row.get(self.cols['phase'], 'SP'),
            'latitude': float(first_row[self.cols['customer_lat']]),
            'longitude': float(first_row[self.cols['customer_lon']]),
            'distance_from_tf': float(first_row.get(self.cols['distance_from_tf'], 0)),
            'monthly_pattern': monthly_pattern,
            'annual_stats': {
                'total': sum(net_values),
                'average': np.mean(net_values),
                'std': np.std(net_values),
                'min': min(net_values),
                'max': max(net_values),
                'peak_month': max(monthly_pattern.keys(), key=lambda m: monthly_pattern[m]['net_consumption']) if monthly_pattern else None
            }
        }

    def get_customer_profile(self, account_no: str) -> Optional[Dict]:
        #Get profile for a specific customer from in-memory cache
        return self.customer_profiles.get(account_no)

    def get_profiles_by_location(self, latitude: float, longitude: float,
                                 radius_km: float = 2.0) -> List[Tuple[str, float]]:

        #Get profiles within a geographic radius
        nearby_profiles = []

        for account, profile in self.customer_profiles.items():
            distance = self._calculate_distance(
                latitude, longitude,
                profile['latitude'], profile['longitude']
            )

            if distance <= radius_km * 1000:
                nearby_profiles.append((account, distance))

        # Sort by distance (closest first)
        return sorted(nearby_profiles, key=lambda x: x[1])

    def get_profiles_by_tariff(self, tariff_code: str) -> List[str]:
        # Get profiles with specific tariff
        return [
            account for account, profile in self.customer_profiles.items()
            if profile['tariff'] == tariff_code
        ]

    def get_dataset_statistics(self) -> Dict:
        #Get dataset statistics

        if self.df is None:
            return {}

        stats = {
            'total_records': len(self.df),
            'unique_accounts': self.df[self.cols['account_no']].nunique(),
            'unique_transformers': self.df[self.cols['transformer_code']].nunique() if self.cols[
                                                                                           'transformer_code'] in self.df.columns else 0,
            'date_range': {
                'min_month': int(self.df[self.cols['month']].min()),
                'max_month': int(self.df[self.cols['month']].max())
            },
            'consumption_stats': {
                'mean': float(self.df[self.cols['net_consumption']].mean()),
                'median': float(self.df[self.cols['net_consumption']].median()),
                'std': float(self.df[self.cols['net_consumption']].std()),
                'min': float(self.df[self.cols['net_consumption']].min()),
                'max': float(self.df[self.cols['net_consumption']].max())
            },
            'tariff_distribution': self.df[self.cols['tariff']].value_counts().to_dict() if self.cols[
                                                                                                'tariff'] in self.df.columns else {},
            'solar_customers': int((self.df[self.cols['has_solar']] > 0).sum()) if self.cols[
                                                                                       'has_solar'] in self.df.columns else 0
        }

        return stats

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:

        #Calculate Haversine distance between two coordinates in meters

        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def validate_user_location(self, latitude: float, longitude: float) -> bool:
        # Validate if user location is within Sri Lanka bounds

        bounds = self.config['similarity']['sri_lanka_bounds']

        return (bounds['lat_min'] <= latitude <= bounds['lat_max']) and \
            (bounds['lon_min'] <= longitude <= bounds['lon_max'])