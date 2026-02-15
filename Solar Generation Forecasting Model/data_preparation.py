import os.path
from datetime import datetime
from src.handle_missing import HandleMissing
from src.data_load import DataLoader

def main():
    print("\n" + "=" * 80)
    print("SOLAR FORECASTING - DATA PREPARATION PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    loader = DataLoader()

    DATA_DIR="data/raw"
    CEB_FILE_PATH = "../processed/MASTER_DATASET_ALL_10TRANSFORMERS.csv"
    WEATHER_FILE = "data_2025.csv"
    NASA_API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    MERGED_FILE = 'merged_data.csv'
    MISSING_HANDLED_FILE = 'missing_val_handled_data.csv'

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

    WEATHER_FILE_PATH = f"{DATA_DIR}/{WEATHER_FILE}"

    if os.path.exists(WEATHER_FILE_PATH):
        weather_data = loader.load_local_weather_data(weather_path=WEATHER_FILE_PATH)
    else:
        print("Local dataset not found")
        weather_data = loader.fetch_weather_data(NASA_API=NASA_API,
                                                 latitude=LATITUDE,
                                                 longitude=LONGITUDE,
                                                 start_yr=2025,
                                                 end_yr=2025,
                                                 save=True,
                                                 savePath=DATA_DIR,
                                                 params=PARAMS
                                                 )

    print("\n1.2 Merging CEB + Weather Data")
    print("-" * 40)
    merged_data = loader.merge_ceb_weather(ceb_df=ceb_data, weather_df=weather_data)

    loader.save_data(df=merged_data, filename=MERGED_FILE)

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

    loader.save_data(df=handled_data, filename=MISSING_HANDLED_FILE)

    print("\nMissing values handling completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")