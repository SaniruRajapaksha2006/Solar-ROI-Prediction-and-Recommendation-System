import pandas as pd
import numpy as np
import requests
import os

class DataLoader:
    
    def __init__(self, filepath: str = "data"):
        self.filepath = filepath

    def load_ceb_data(self, ceb_file_path):
        if not os.path.exists(ceb_file_path):
            raise FileNotFoundError(f"File not found: {ceb_file_path}")
        try:
            print("Loading CEB data...")
            ceb_data = pd.read_csv(ceb_file_path)
            print(f"Loaded {len(ceb_data):,} records")
            return ceb_data
        except FileNotFoundError as e:
            print(e)

    def load_local_weather_data(self, weather_path: str= "data/raw/data_2025.csv"):
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f"Weather file not found: {weather_path}")
        try:
            print(f"Loading weather data: {weather_path}")
            df = pd.read_csv(weather_path)
            print(f"Loaded {len(df)} months")
            return df
        except FileNotFoundError as e:
            print(e)

    def fetch_weather_data(self, NASA_API:str, latitude:float, longitude:float,
                           start_yr:int=2025, end_yr:int=2025, savePath:str="data/raw/" ):
        print(f"Fetching NASA POWER weather data...")
        print(f"  Location: {latitude}°N, {longitude}°E")
        print(f"  Year: {start_yr}-{end_yr}")

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

        params_str = ",".join(PARAMS.keys())
        url = (
            f"{NASA_API}?"
            f"parameters={params_str}&"
            f"community=SB&"
            f"longitude={longitude}&"
            f"latitude={latitude}&"
            f"start={start_yr}&"
            f"end={end_yr}&"
            f"format=JSON"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data_2025 = response.json()

            if data_2025:
                print("Data received")
                labeled_data =  self.labeling(data_2025, PARAMS)
                path = os.path.join(savePath, "data_2025.csv")
                labeled_data.to_csv(path, index=False)
                return labeled_data
            return None

        except Exception as e:
            print(e)
            return None

    def labeling(self, data:str, params:dict={}):
        print("Labeling data...")
        weather_2025 = []

        if data:
            param_data = data['properties']['parameter']

            for month in range(1, 13):
                date_key = f"2025{month:02d}"
                row = {'Month': month}

                for orig_name, readable_name in params.items():
                    if date_key in param_data[orig_name]:
                        row[readable_name] = param_data[orig_name][date_key]
                    else:
                        row[readable_name] = np.nan  # Mark as missing

                weather_2025.append(row)
        weather_df = pd.DataFrame(weather_2025)
        print("Finished labeling data")
        return weather_df

    def merge_ceb_weather(self, ceb_df: pd.DataFrame, weather_df: pd.DataFrame):
        print("Merging CEB + Weather data...")

        ceb_df.rename(columns={"MONTH": "Month"}, inplace=True)

        print(f"Merged: {len(ceb_df):,} records")
        print(f"Columns: {ceb_df.shape[1]}")
        return ceb_df.merge(weather_df, on='Month', how='left')

    def save_data(self, df, filename):
        print("Saving data...")
        filepath = f"data/processed/{filename}"
        df.to_csv(filepath, index=False)
        print(f"\nSaved: {filepath}")
