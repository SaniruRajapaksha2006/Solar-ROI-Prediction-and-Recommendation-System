import os.path
from datetime import datetime

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

    # Maharagama coordinates
    LATITUDE = 6.8514
    LONGITUDE = 79.9211

    print("STEP 1: DATA LOADING")
    print("-" * 80)
    ceb_data = loader.load_ceb_data(ceb_file_path=CEB_FILE_PATH)

    print("\n1.2 Loading Weather Data")
    print("-" * 40)

    WEATHER_FILE_PATH = f"{DATA_DIR}/{WEATHER_FILE}"

    if os.path.exists(WEATHER_FILE_PATH):
        weather_data = loader.load_local_weather_data(weather_path=WEATHER_FILE_PATH)
    else:
        print("\nLocal dataset not found")
        weather_data = loader.fetch_weather_data(NASA_API=NASA_API,
                                                 latitude=LATITUDE,
                                                 longitude=LONGITUDE,
                                                 start_yr=2025,
                                                 end_yr=2025,
                                                 savePath=DATA_DIR
                                                 )

    print("\n1.3 Merging CEB + Weather Data")
    print("-" * 40)
    merged_data = loader.merge_ceb_weather(ceb_df=ceb_data, weather_df=weather_data)

    loader.save_data(df=merged_data, filename=MERGED_FILE)

    print("\nData Loading completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")