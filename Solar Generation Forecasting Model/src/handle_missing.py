import os.path

import numpy as np
import pandas as pd
from src.data_load import DataLoader

class HandleMissing:

    def __init__(self):
        pass

    def analysis_missing(self, dataframe:pd.DataFrame):
        missing_stat = []

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].replace(-999, np.nan)
            missingCount = dataframe[col].isna().sum()
            if missingCount:
                missing_stat.append({
                    'Column': col,
                    'Missing': missingCount,
                    'Percent': f"{missingCount / len(dataframe) * 100:.1f}%"
                })

        if not missing_stat:
            print("No missing values found")
            print(missing_stat)
            return pd.DataFrame()

        stats_df = pd.DataFrame(missing_stat)
        print(stats_df.to_string(index=False))
        print("-" * 40)

        return stats_df


    def impute_missing(self, data:pd.DataFrame, missing_stats:pd.DataFrame, params:dict,
                       NASA_API:str, latitude:float, longitude:float, start_yr:int=2014, end_yr:int=2024,
                       save:bool=False, savePath:str="data/raw/"):

        loader = DataLoader()

        new_params = {}
        for org_name, read_name in params.items():
            if read_name in missing_stats.columns:
                new_params[org_name] = read_name

        try:
            data_hist = loader.fetch_weather_data(NASA_API=NASA_API,
                                                   latitude=latitude,
                                                   longitude=longitude,
                                                   start_yr=start_yr,
                                                   end_yr=end_yr,
                                                   save=save,
                                                   savePath=savePath,
                                                   params=params
                                                   )

            param_allsky = data_hist['properties']['parameter']['ALLSKY_SFC_SW_DWN']
            param_csky   = data_hist['properties']['parameter']['CLRSKY_SFC_SW_DWN']

            averages = {}
            for month in [11, 12]:
                vals_allsky = [param_allsky[f"{year}{month:02d}"] for year in range(2018, 2023)]
                vals_csky   = [param_csky[f"{year}{month:02d}"] for year in range(2018, 2023)]

                averages[month] = {
                    'Solar_Irradiance_GHI': round(sum(vals_allsky)/len(vals_allsky),2),
                    'Clear_Sky_GHI': round(sum(vals_csky)/len(vals_csky),2)
                }

            for month in [11, 12]:
                data.loc[data['Month'].astype(int)==month, 'Solar_Irradiance_GHI'] = averages[month]['Solar_Irradiance_GHI']
                data.loc[data['Month'].astype(int)==month, 'Clear_Sky_GHI'] = averages[month]['Clear_Sky_GHI']

            print(f"\nFilled missing months")
            return data

        except Exception as e:
            print(f"✗ Could not fetch 10-year historical data: {e}")