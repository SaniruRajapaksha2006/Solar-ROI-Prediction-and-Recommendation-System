import pandas as pd
import numpy as np


class FeatureEngineer:
    """Engineer features for solar forecasting"""
    
    def __init__(self):
        self.created_features = []
    
    def create_temperature_features(self, df):
        """
        Temperature-based features
        
        Solar panels lose efficiency at higher temps (~0.5% per °C above 25°C)
        
        Args:
            df: DataFrame with Temperature, Max_Temperature, Min_Temperature
            
        Returns:
            DataFrame with temperature features
        """
        print("\nCreating temperature features...")
        print("-" * 60)
        
        df_feat = df.copy()
        
        # 1. Temperature efficiency factor
        # Panel efficiency = 1 + (T - 25) * -0.005
        TEMP_COEFF = -0.005
        df_feat['Temp_Efficiency'] = (1 + (df_feat['Temperature'] - 25) * TEMP_COEFF).round(3)
        print(" Temp_Efficiency (panel efficiency loss)")
        self.created_features.append('Temp_Efficiency')
        
        # 2. Temperature range (stability indicator)
        df_feat['Temp_Range'] = (df_feat['Max_Temperature'] - df_feat['Min_Temperature']).round(2)
        print(" Temp_Range (daily variation)")
        self.created_features.append('Temp_Range')
        
        # 3. GHI adjusted for temperature
        df_feat['GHI_Adjusted'] = (df_feat['Solar_Irradiance_GHI'] * df_feat['Temp_Efficiency']).round(2)
        print(" GHI_Adjusted (GHI × temp efficiency)")
        self.created_features.append('GHI_Adjusted')
        
        print("-" * 60)
        return df_feat
    
    def create_cloud_features(self, df):
        """
        Cloud/weather features
        
        Args:
            df: DataFrame with Solar_Irradiance_GHI, Clear_Sky_GHI
            
        Returns:
            DataFrame with cloud features
        """
        print("\nCreating cloud features...")
        print("-" * 60)
        
        df_feat = df.copy()
        
        # Cloud factor (how much clouds reduce irradiance)
        # Avoid division by zero
        df_feat['Cloud_Factor'] = df_feat['Solar_Irradiance_GHI'] / df_feat['Clear_Sky_GHI'].replace(0, np.nan)

        # Fill missing values (NaN from zero or missing Clear_Sky_GHI)
        df_feat['Cloud_Factor'] = df_feat['Cloud_Factor'].fillna(1.0)

        # Round to 2 decimals
        df_feat['Cloud_Factor'] = df_feat['Cloud_Factor'].round(2)
        print(" Cloud_Factor (actual GHI / clear sky GHI)")
        self.created_features.append('Cloud_Factor')
        
        print("-" * 60)
        return df_feat
    
    def create_temporal_features(self, df):
        """
        Time-based features for seasonality
        
        Args:
            df: DataFrame with Month
            
        Returns:
            DataFrame with temporal features
        """
        print("\nCreating temporal features...")
        print("-" * 60)
        
        df_feat = df.copy()
        
        # 1. Cyclical encoding (captures month similarity: Dec ≈ Jan)
        df_feat['Month_Sin'] = (np.sin(2 * np.pi * df_feat['Month'] / 12)).round(2)
        df_feat['Month_Cos'] = (np.cos(2 * np.pi * df_feat['Month'] / 12)).round(2)
        print(" Month_Sin, Month_Cos (cyclical encoding)")
        self.created_features.extend(['Month_Sin', 'Month_Cos'])
        
        # 4. Days in month
        days_map = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                    7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        df_feat['Days_In_Month'] = df_feat['Month'].map(days_map)
        print(" Days_In_Month")
        self.created_features.append('Days_In_Month')
        
        print("-" * 60)
        return df_feat
    
    def create_system_features(self, df):
        """
        System-based features
        
        Args:
            df: DataFrame with INV_CAPACITY, Solar_Irradiance_GHI
            
        Returns:
            DataFrame with system features
        """
        print("\nCreating system features...")
        print("-" * 60)
        
        df_feat = df.copy()

        # Feature: Theoretical Yield (DC-to-AC conversion)
        # This acts as a 'Physics-Informed' feature to guide the model's scale.
        # It represents the ideal energy harvest considering standard system losses (PR=0.80).
        PERFORMANCE_RATIO = 0.80  # System losses (inverter, wiring, soiling)
        df_feat['Expected_Generation'] = (
            df_feat['Solar_Irradiance_GHI'] *
            df_feat['INV_CAPACITY'] *
            PERFORMANCE_RATIO *
            df_feat['Days_In_Month']
        ).round(2)
        print(" Expected_Generation (physics formula)")
        self.created_features.append('Expected_Generation')

        # Benchmark: Specific Yield (Deterministic Baseline)
        # Used as a Zero-Intelligence (ZI) baseline to validate ML performance gain.
        # Calculated in kWh/kW to provide a direct performance ratio comparison
        # against the model's predicted Efficiency target.
        df_feat['Physics_Pred'] = (
            df_feat['Solar_Irradiance_GHI'] *
            PERFORMANCE_RATIO *
            df_feat['Days_In_Month']
        ).round(2)
        print(" Physics_Pred (GHI × 0.80 × Days — naive baseline, kWh/kW)")
        self.created_features.append('Physics_Pred')

        print("-" * 60)
        return df_feat
    
    def create_all_features(self, df):
        """
        Create all engineered features
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with all features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df_feat = df.copy()
        
        # Create features
        df_feat = self.create_temperature_features(df_feat)
        df_feat = self.create_cloud_features(df_feat)
        df_feat = self.create_temporal_features(df_feat)
        df_feat = self.create_system_features(df_feat)
        
        print(f"\n✓ Created {len(self.created_features)} features")
        print("="*60)
        
        return df_feat
    
    def get_feature_list(self):
        """Return list of created features"""
        return self.created_features
