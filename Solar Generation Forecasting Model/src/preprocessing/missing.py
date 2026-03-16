"""
src/preprocessing/missing.py
Fills missing Solar_Irradiance_GHI and Clear_Sky_GHI for specific months
using a 5-year historical average from NASA POWER.
"""

import numpy as np
import pandas as pd

from utils.nasa_power import fetch_monthly, NASA_PARAMS


class MissingValueHandler:

    def analyse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print and return a summary of missing / -999 values."""
        rows = []
        for col in df.columns:
            df[col] = df[col].replace(-999, np.nan)
            n = df[col].isna().sum()
            if n:
                rows.append({
                    "Column":  col,
                    "Missing": n,
                    "Pct":     f"{n / len(df) * 100:.1f}%",
                })

        if not rows:
            print("  No missing values found")
            return pd.DataFrame()

        stats = pd.DataFrame(rows)
        print(stats.to_string(index=False))
        return stats

    def impute_ghi(self, df: pd.DataFrame,
                   missing_stats: pd.DataFrame,
                   latitude: float,
                   longitude: float,
                   impute_months: list[int] = None,
                   start_yr: int = 2018,
                   end_yr: int = 2022,
                   params: dict = None) -> pd.DataFrame:
        """
        Fill Solar_Irradiance_GHI and Clear_Sky_GHI for specific months
        using the mean of NASA POWER monthly data over start_yr–end_yr.

        Args:
            df             : Merged CEB + weather DataFrame
            missing_stats  : Output of analyse() — used to skip fetch when
                             neither GHI column is actually missing
            latitude       : Site latitude
            longitude      : Site longitude
            impute_months  : Month numbers to fill (default: [11, 12])
            start_yr       : Historical average start year (default: 2018)
            end_yr         : Historical average end year   (default: 2022)
            params         : NASA param map (default: NASA_PARAMS)

        Returns:
            DataFrame with missing GHI values filled
        """
        if params is None:
            params = NASA_PARAMS
        if impute_months is None:
            impute_months = [11, 12]

        ghi_cols = ["Solar_Irradiance_GHI", "Clear_Sky_GHI"]
        if missing_stats.empty or not any(
            c in missing_stats["Column"].values for c in ghi_cols
        ):
            print("  No GHI columns missing — skipping imputation")
            return df

        print(f"  Fetching NASA historical averages ({start_yr}–{end_yr})...")
        try:
            raw        = fetch_monthly(lat=latitude, lon=longitude,
                                       start_yr=start_yr, end_yr=end_yr,
                                       params=params)
            param_data = raw["properties"]["parameter"]
            allsky     = param_data["ALLSKY_SFC_SW_DWN"]
            clrsky     = param_data["CLRSKY_SFC_SW_DWN"]

            df = df.copy()
            for month in impute_months:
                allsky_vals = [allsky[f"{yr}{month:02d}"]
                               for yr in range(start_yr, end_yr + 1)
                               if f"{yr}{month:02d}" in allsky]
                clrsky_vals = [clrsky[f"{yr}{month:02d}"]
                               for yr in range(start_yr, end_yr + 1)
                               if f"{yr}{month:02d}" in clrsky]

                avg   = None
                avg_c = None

                if allsky_vals:
                    avg = round(sum(allsky_vals) / len(allsky_vals), 2)
                    df.loc[df["Month"].astype(int) == month,
                           "Solar_Irradiance_GHI"] = avg

                if clrsky_vals:
                    avg_c = round(sum(clrsky_vals) / len(clrsky_vals), 2)
                    df.loc[df["Month"].astype(int) == month,
                           "Clear_Sky_GHI"] = avg_c

                ghi_str  = f"{avg:.2f}"   if avg   is not None else "n/a"
                clr_str  = f"{avg_c:.2f}" if avg_c is not None else "n/a"
                print(f"  Month {month:02d}: GHI={ghi_str}  ClearSky={clr_str}")

            print(f"  ✓ Imputed {len(impute_months)} months")
            return df

        except Exception as e:
            print(f"  ✗ Imputation failed: {e} — returning original df")
            return df
