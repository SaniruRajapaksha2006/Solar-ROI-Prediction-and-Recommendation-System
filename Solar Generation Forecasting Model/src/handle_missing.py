import numpy as np
import pandas as pd

from utils.nasa_power import fetch_monthly, NASA_PARAMS


class HandleMissing:

    def __init__(self):
        pass

    def analysis_missing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing_stat = []

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].replace(-999, np.nan)
            count = dataframe[col].isna().sum()
            if count:
                missing_stat.append({
                    "Column":  col,
                    "Missing": count,
                    "Percent": f"{count / len(dataframe) * 100:.1f}%"
                })

        if not missing_stat:
            print("No missing values found")
            return pd.DataFrame()

        stats_df = pd.DataFrame(missing_stat)
        print(stats_df.to_string(index=False))
        print("-" * 40)
        return stats_df

    def impute_missing(self, data: pd.DataFrame, missing_stats: pd.DataFrame,
                       latitude: float, longitude: float,
                       impute_months: list = None,
                       start_yr: int = 2018, end_yr: int = 2022,
                       params: dict = None) -> pd.DataFrame:
        """
        Fill missing Solar_Irradiance_GHI and Clear_Sky_GHI for specific months
        using a 5-year average from NASA POWER monthly data.

        Args:
            data           : DataFrame with Month column and solar columns
            missing_stats  : output of analysis_missing() - used to check which
                             columns actually need imputation
            latitude       : location latitude
            longitude      : location longitude
            impute_months  : list of month numbers to fill (default: [11, 12])
            start_yr       : historical average start year (default: 2018)
            end_yr         : historical average end year   (default: 2022)
            params         : NASA param map (default: NASA_PARAMS)

        Returns:
            DataFrame with missing values filled
        """
        if params is None:
            params = NASA_PARAMS

        if impute_months is None:
            impute_months = [11, 12]

        # Only fetch if these columns are actually missing
        cols_to_fill = ["Solar_Irradiance_GHI", "Clear_Sky_GHI"]
        if not any(col in missing_stats["Column"].values for col in cols_to_fill):
            print("No solar columns missing - skipping imputation")
            return data

        try:
            raw = fetch_monthly(lat=latitude, lon=longitude,
                                start_yr=start_yr, end_yr=end_yr, params=params)

            param_data = raw["properties"]["parameter"]
            allsky = param_data["ALLSKY_SFC_SW_DWN"]
            clrsky = param_data["CLRSKY_SFC_SW_DWN"]

            for month in impute_months:
                # Average across years for this month
                allsky_vals = [allsky[f"{yr}{month:02d}"] for yr in range(start_yr, end_yr + 1)
                               if f"{yr}{month:02d}" in allsky]
                clrsky_vals = [clrsky[f"{yr}{month:02d}"] for yr in range(start_yr, end_yr + 1)
                               if f"{yr}{month:02d}" in clrsky]

                if allsky_vals:
                    avg_allsky = round(sum(allsky_vals) / len(allsky_vals), 2)
                    data.loc[data["Month"].astype(int) == month, "Solar_Irradiance_GHI"] = avg_allsky

                if clrsky_vals:
                    avg_clrsky = round(sum(clrsky_vals) / len(clrsky_vals), 2)
                    data.loc[data["Month"].astype(int) == month, "Clear_Sky_GHI"] = avg_clrsky

                print(f"  Filled month {month}: GHI={avg_allsky:.2f}, ClearSky={avg_clrsky:.2f}")

            print(f"Imputed {len(impute_months)} months using {start_yr}–{end_yr} average")
            return data

        except Exception as e:
            print(f"✗ Imputation failed: {e}")
            return data
