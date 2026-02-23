from datetime import datetime
from pathlib import Path

from src.data_load import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.handle_missing import HandleMissing
from src.outlier_detection import OutlierDetector

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"

def main():
    print("\n" + "=" * 80)
    print("SOLAR FORECASTING - DATA PREPARATION PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    loader = DataLoader()

    CEB_FILE_PATH = PROJECT_ROOT / "processed" / "MASTER_DATASET_ALL_10TRANSFORMERS.csv"
    WEATHER_FILE = "data_2025.csv"
    NASA_API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    MERGED_FILE = '00_merged_data.csv'
    MISSING_HANDLED_FILE = '01_imputed.csv'
    OUTLIERS_REMOVED_FILE = "02_outliers_removed.csv"
    FEATURE_ENGINEERING_FILE = "04_features_engineered.csv"

    # Maharagama coordinates
    LATITUDE = 6.8514
    LONGITUDE = 79.9211

    PARAMS = {
        'ALLSKY_SFC_SW_DWN': 'Solar_Irradiance_GHI',  # kWh/m²/day
        'T2M': 'Temperature',  # °C
        'T2M_MAX': 'Max_Temperature',  # °C
        'T2M_MIN': 'Min_Temperature',  # °C
        'RH2M': 'Humidity',  # %
        'PRECTOTCORR': 'Precipitation',  # mm/day
        'WS10M': 'Wind_Speed',  # m/s
        'CLRSKY_SFC_SW_DWN': 'Clear_Sky_GHI'  # kWh/m²/day
    }


    # 1. Data loading

    print("\nSTEP 1: DATA LOADING")
    print("=" * 80)
    ceb_data = loader.load_ceb_data(ceb_file_path=CEB_FILE_PATH)

    print("\n1.1 Loading Weather Data")
    print("-" * 40)

    WEATHER_FILE_PATH = RAW_DATA_DIR / WEATHER_FILE

    if WEATHER_FILE_PATH.exists():
        weather_data = loader.load_local_weather_data(weather_path=WEATHER_FILE_PATH)
    else:
        print("Local dataset not found")
        weather_data = loader.fetch_weather_data(NASA_API=NASA_API,
                                                 latitude=LATITUDE,
                                                 longitude=LONGITUDE,
                                                 start_yr=2025,
                                                 end_yr=2025,
                                                 save=True,
                                                 savePath=RAW_DATA_DIR,
                                                 params=PARAMS
                                                 )

    print("\n1.2 Merging CEB + Weather Data")
    print("-" * 40)
    merged_data = loader.merge_ceb_weather(ceb_df=ceb_data, weather_df=weather_data)

    loader.save_data(df=merged_data, filepath=PROCESSED_DIR / MERGED_FILE)

    print("\nData Loading completed")



    # 2. missing value handling

    print("\n\nSTEP 2: MISSING VALUES HANDLING")
    print("=" * 40)

    handler = HandleMissing()

    print("2.1 Analysing missing values")
    print("-" * 40)
    stats_df = handler.analysis_missing(dataframe=merged_data)

    print("\n2.2 Imputing missing values")
    print("-" * 40)
    handled_data = handler.impute_missing(data=merged_data,
                           missing_stats=stats_df,
                           params=PARAMS,
                           NASA_API=NASA_API,
                           latitude=LATITUDE,
                           start_yr=2014,
                           end_yr=2024,
                           longitude=LONGITUDE,
                           save=False)

    print("\n2.3 Reanalyse missing values")
    print("-" * 40)
    handler.analysis_missing(dataframe=handled_data)

    loader.save_data(df=handled_data, filepath=PROCESSED_DIR / MISSING_HANDLED_FILE)

    print("\nMissing values handling completed")



    print("\n\nSTEP 3: OUTLIER DETECTION & REMOVAL")
    print("=" * 40)

    detector = OutlierDetector(threshold=1.5)

    # 3.1 remove high export accounts
    df_no_high_export = detector.remove_high_export_accounts(handled_data,700)

    # 3.2 remove outliers from
    df_outliers_cleaned = detector.detect_monthly_outliers(df_no_high_export, 'EXPORT_kWh', 1.5)


    featureEngineer = FeatureEngineer()

    df_with_new_features = featureEngineer.create_all_features(df_outliers_cleaned)

    new_features = featureEngineer.get_feature_list()

    print(df_with_new_features.columns)



    non_informative_cols = [
        # IDs - just identifiers, no predictive value
        'TRANSFORMER_CODE',
        'ACCOUNT_NO',

        # Locations - derived weather data
        'TRANSFORMER_LAT',
        'TRANSFORMER_LON',
        'CUSTOMER_LAT',
        'CUSTOMER_LON',
        'DISTANCE_FROM_TF_M',

        # Metadata - only for analysis
        'DATA_QUALITY',
        'SOURCE',
        'CAL_TARIFF',
        'PHASE',

        # only available one year data
        'YEAR',

        # Redundant
        'HAS_SOLAR',  # All records are solar (filtered)

        # Not target
        'NET_CONSUMPTION_kWh',
        'IMPORT_kWh',  # only care about EXPORT

        # =========================================================
        # correlated with temperature
        'Max_Temperature',
        'Min_Temperature',
        # correlated with solar irradiance
        'Clear_Sky_GHI'
    ]

    selector = FeatureSelector()

    # =========================================================
    keep_cols = ['EXPORT_kWh', 'Month', 'INV_CAPACITY'] + list(PARAMS.values()) + new_features

    df_feature_cleaned = selector.select_features(df=df_with_new_features,
                                                  target='EXPORT_kWh',
                                                  correlation_threshold=0.05,
                                                  non_informative_cols=non_informative_cols,
                                                  keep_cols=keep_cols)

    loader.save_data(df=df_feature_cleaned,filepath=PROCESSED_DIR / FEATURE_ENGINEERING_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")