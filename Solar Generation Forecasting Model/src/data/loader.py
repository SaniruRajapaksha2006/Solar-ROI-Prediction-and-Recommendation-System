"""
Loads CEB export data and NASA POWER weather, merges and saves.
"""

from pathlib import Path

import pandas as pd

from utils.nasa_power import fetch_monthly, label_monthly, NASA_PARAMS


class DataLoader:

    def load_ceb_data(self, ceb_file_path: str | Path) -> pd.DataFrame:
        path = Path(ceb_file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        print("Loading CEB data...")
        df = pd.read_csv(path)

        print("Filtering solar records (HAS_SOLAR == 1)...")
        df = df[df["HAS_SOLAR"] == 1].drop(columns=["HAS_SOLAR"]).copy()

        print(f"  Loaded {len(df):,} solar records")
        return df

    def load_local_weather(self, weather_path: str | Path = "data/raw/data_2025.csv") -> pd.DataFrame:
        path = Path(weather_path)
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")

        print(f"Loading local weather: {path}")
        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} months")
        return df

    def fetch_nasa_weather(self, latitude: float, longitude: float,
                           start_yr: int = 2025, end_yr: int = 2025,
                           params: dict = None,
                           save_path: str | Path = "data/raw/data_2025.csv") -> pd.DataFrame:
        """Fetch NASA POWER monthly weather and cache to CSV."""
        if params is None:
            params = NASA_PARAMS

        raw = fetch_monthly(lat=latitude, lon=longitude,
                            start_yr=start_yr, end_yr=end_yr, params=params)
        df = label_monthly(raw, params=params, year=start_yr)

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  Saved weather: {path}")
        return df

    def merge_ceb_weather(self, ceb_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        print("Merging CEB + weather on Month...")
        ceb_df = ceb_df.rename(columns={"MONTH": "Month"})
        merged = ceb_df.merge(weather_df, on="Month", how="left")
        print(f"  {len(merged):,} records  |  {merged.shape[1]} columns")
        return merged

    def save(self, df: pd.DataFrame, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
