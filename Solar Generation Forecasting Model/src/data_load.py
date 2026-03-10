from pathlib import Path

import pandas as pd

from utils.nasa_power import fetch_monthly, label_monthly, NASA_PARAMS


class DataLoader:

    def __init__(self):
        pass

    def load_ceb_data(self, ceb_file_path: str | Path):
        path = Path(ceb_file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        print("Loading CEB data...")
        ceb_data = pd.read_csv(path)

        print("Filtering solar records...")
        solar_data = ceb_data[ceb_data["HAS_SOLAR"] == 1].copy()
        solar_data = solar_data.drop(columns=["HAS_SOLAR"])

        print(f"Loaded {len(solar_data):,} solar records")
        return solar_data

    def load_local_weather_data(self, weather_path: str | Path = "data/raw/data_2025.csv"):
        path = Path(weather_path)
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")

        print(f"Loading weather data: {path}")
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} months")
        return df

    def fetch_weather_data(self, latitude: float, longitude: float,
                           start_yr: int = 2025, end_yr: int = 2025,
                           params: dict = None, save: bool = True,
                           save_path: str | Path = "data/raw/data_2025.csv") -> pd.DataFrame:
        """
        Fetch NASA POWER monthly weather and return as labeled DataFrame.
        """
        if params is None:
            params = NASA_PARAMS

        raw = fetch_monthly(lat=latitude, lon=longitude,
                            start_yr=start_yr, end_yr=end_yr, params=params)

        df = label_monthly(raw, params=params, year=start_yr)

        if save:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Saved: {path}")

        return df

    def merge_ceb_weather(self, ceb_df: pd.DataFrame, weather_df: pd.DataFrame):
        print("Merging CEB + Weather data...")
        ceb_df = ceb_df.rename(columns={"MONTH": "Month"})
        merged = ceb_df.merge(weather_df, on="Month", how="left")
        print(f"Merged: {len(merged):,} records  |  Columns: {merged.shape[1]}")
        merged['Total_Generation_kWh'] = merged['EXPORT_kWh'] + merged['IMPORT_kWh']
        return merged

    def save_data(self, df: pd.DataFrame, filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
