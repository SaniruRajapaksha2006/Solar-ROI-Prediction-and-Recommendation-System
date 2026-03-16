"""
fetch_future_weather.py
===========================================================================
Fetch a 12-month future weather DataFrame for Maharagama / Colombo.

Primary  : Copernicus C3S SEAS5 (GRIB, monthly_mean)
             • Area-filtered to Sri Lanka
             • Cached as CSV after first parse
Fallback : NASA POWER TMY via utils/nasa_power.py
             • Fills any NaN cell from C3S, or used entirely if C3S fails

OUTPUT — raw weather only, no features, no INV_CAPACITY
---------------------------------------------------------
Columns: Month, Solar_Irradiance_GHI, Temperature, Max_Temperature,
         Min_Temperature, Humidity, Precipitation, Wind_Speed,
         Clear_Sky_GHI, data_source
"""

import math
import os
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from utils.nasa_power import fetch_tmy
from utils.utils_config import load_config

DAYS_IN_MONTH = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}


def _locations():
    cfg = load_config()["location"]
    return [cfg["primary"], cfg["fallback"]]

def _c3s_cfg():
    return load_config()["c3s"]


# ===============================================================================
# C3S HELPERS  (private)
# ===============================================================================

def _check_cdsapi() -> bool:
    try:
        import cdsapi  # noqa: F401
        if not (Path.home() / ".cdsapirc").exists():
            print("~/.cdsapirc not found. Create at: https://cds.climate.copernicus.eu/profile")
            return False
        return True
    except ImportError:
        print("cdsapi not installed → pip install cdsapi")
        return False


def _months_to_fetch(start_year: int, start_month: int, n: int = 12) -> list:
    result, y, m = [], start_year, start_month
    for _ in range(n):
        result.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return result


def _open_grib_var(grib_path: str, short_name: str):
    """Open one variable from a SEAS5 GRIB, scanning all cfgrib sub-datasets."""
    try:
        import cfgrib
    except ImportError:
        return None
    try:
        ds = cfgrib.open_dataset(grib_path, filter_by_keys={"shortName": short_name},
                                 indexpath=None, errors="raise")
        return ds[list(ds.data_vars)[0]]
    except Exception:
        pass
    try:
        for ds in cfgrib.open_datasets(grib_path, indexpath=None, errors="ignore"):
            for var in ds.data_vars:
                if var == short_name or ds[var].attrs.get("GRIB_shortName") == short_name:
                    return ds[var]
    except Exception:
        pass
    return None


def _extract_leadtime_values(da, leadtimes: list, lat: float, lon: float) -> dict:
    """Ensemble-mean scalar per leadtime. Handles all SEAS5 dim layouts."""
    result = {}
    try:
        if "number" in da.dims:
            da = da.mean(dim="number")
        sel = {d: (lat if d == "latitude" else lon)
               for d in ["latitude", "longitude"] if d in da.dims}
        if sel:
            da = da.sel(sel, method="nearest")
        if "step" in da.dims:
            for idx, lt in enumerate(leadtimes):
                try:
                    val = float(da.isel(step=idx).values) if idx < da.sizes["step"] else np.nan
                    result[lt] = np.nan if math.isnan(val) else val
                except Exception:
                    result[lt] = np.nan
        else:
            raw = da.values
            val = float(raw) if raw.ndim == 0 else float(raw.mean())
            for lt in leadtimes:
                result[lt] = np.nan if math.isnan(val) else val
    except Exception:
        for lt in leadtimes:
            result[lt] = np.nan
    return result


def _cache_to_dict(cache_df: pd.DataFrame, months_needed: list,
                   leadtimes: list, n_months: int) -> dict:
    cols = ["Solar_Irradiance_GHI", "Temperature", "Max_Temperature",
            "Min_Temperature", "Humidity", "Precipitation", "Wind_Speed"]
    result = {col: [np.nan] * n_months for col in cols}
    for i, (y, m) in enumerate(months_needed):
        row = cache_df[(cache_df["year"] == y) & (cache_df["month"] == m)]
        if not row.empty:
            for col in cols:
                v = row[col].iloc[0] if col in row.columns else np.nan
                result[col][i] = float(v) if not pd.isna(v) else np.nan
    print(f"{sum(1 for v in result['Temperature'] if not np.isnan(v))} months from cache")
    return result


# ===============================================================================
# C3S FETCH
# ===============================================================================

def _fetch_c3s(lat: float, lon: float, start_year: int, start_month: int,
               n_months: int, tmp_dir: str) -> Optional[dict]:
    """
    Fetch ECMWF SEAS5 forecast (monthly_mean, all variables, Sri Lanka area).
    Returns {col_name: [val, ...]} or None on failure.
    Caches parsed scalars as CSV — re-downloads only when cache is missing.
    """
    if not _check_cdsapi():
        return None

    import cdsapi
    c3s = _c3s_cfg()

    os.makedirs(tmp_dir, exist_ok=True)
    cache_dir = os.path.join(tmp_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    months_needed = _months_to_fetch(start_year, start_month, n_months)

    # Init month (SEAS5 ~1-month lag)
    today = date.today()
    iy, im = today.year, today.month - 1
    if im < 1:
        im, iy = 12, iy - 1

    max_lt = _c3s_cfg()["max_leadtime_months"]
    leadtimes = [(y - iy)*12 + (m - im) for y, m in months_needed
                 if 1 <= (y - iy)*12 + (m - im) <= max_lt]
    if not leadtimes:
        print("Outside SEAS5 6-month window → NASA TMY")
        return None

    print(f"  C3S init: {iy}-{im:02d}  lead times: {leadtimes}")

    # Cache check
    cache_csv = os.path.join(cache_dir, f"seas5_{iy}{im:02d}_extracted.csv")
    if os.path.exists(cache_csv):
        print(f"  [CACHE] {cache_csv}")
        try:
            return _cache_to_dict(pd.read_csv(cache_csv), months_needed, leadtimes, n_months)
        except Exception as e:
            print(f"Cache read failed ({e}) — re-downloading...")

    # Download
    grib_path = os.path.join(tmp_dir, f"seas5_{iy}{im:02d}.grib")
    if not os.path.exists(grib_path):
        print(f"  Downloading → {grib_path}  (Sri Lanka area only)")
        try:
            cdsapi.Client(quiet=True).retrieve(c3s["dataset"], {
                "originating_centre": c3s["originating_centre"], "system": c3s["system"],
                "product_type": c3s["product_type"],
                "variable": [
                    "2m_temperature", "maximum_2m_temperature_in_the_last_24_hours",
                    "minimum_2m_temperature_in_the_last_24_hours", "2m_dewpoint_temperature",
                    "10m_u_component_of_wind", "10m_v_component_of_wind",
                    "surface_solar_radiation_downwards", "total_precipitation",
                ],
                "year": str(iy), "month": f"{im:02d}",
                "leadtime_month": [str(lt) for lt in leadtimes],
                "area": c3s["sri_lanka_area"], "format": "grib",
            }, grib_path)
            print(f"{os.path.getsize(grib_path)/1e6:.2f} MB")
        except Exception as e:
            print(f"  [✗] Download failed: {e}")
            return None
    else:
        print(f"  [GRIB EXISTS] {os.path.getsize(grib_path)/1e6:.2f} MB")

    try:
        import cfgrib    # noqa: F401
        import xarray    # noqa: F401
    except ImportError:
        print("  [✗] pip install cfgrib xarray eccodes")
        return None

    # Parse
    print("  Parsing...")
    extracted = {}
    for sn in ["2t", "mx2t24", "mn2t24", "2d", "u10", "v10", "ssrd", "tp"]:
        da = _open_grib_var(grib_path, sn)
        if da is not None:
            extracted[sn] = _extract_leadtime_values(da, leadtimes, lat, lon)
            n_ok = sum(1 for v in extracted[sn].values() if not np.isnan(v))
            print(f"  {sn:8s}: {n_ok}/{len(leadtimes)}")
        else:
            print(f"  {sn}: not found")

    if not extracted or all(all(np.isnan(v) for v in d.values()) for d in extracted.values()):
        print("  [✗] All NaN — falling back to NASA TMY")
        return None

    # Unit conversion
    result = {col: [np.nan] * n_months for col in
              ["Solar_Irradiance_GHI", "Temperature", "Max_Temperature",
               "Min_Temperature", "Humidity", "Precipitation", "Wind_Speed"]}

    # Build month-position map: (year,month) -> index in months_needed
    month_pos = {(y, m): idx for idx, (y, m) in enumerate(months_needed)}

    for i, lt in enumerate(leadtimes):
        y, m = months_needed[i]
        pos  = month_pos[(y, m)]   # correct position in n_months-sized result array
        days = DAYS_IN_MONTH[m]

        for src, col in [("2t","Temperature"), ("mx2t24","Max_Temperature"), ("mn2t24","Min_Temperature")]:
            v = extracted.get(src, {}).get(lt, np.nan)
            if not np.isnan(v): result[col][pos] = round(v - 273.15, 2)

        td_k, t_c = extracted.get("2d", {}).get(lt, np.nan), result["Temperature"][pos]
        if not np.isnan(td_k) and not np.isnan(t_c):
            td_c = td_k - 273.15
            result["Humidity"][pos] = round(min(
                100 * math.exp(17.625*td_c/(243.04+td_c)) / math.exp(17.625*t_c/(243.04+t_c)), 100.0), 1)

        ssrd = extracted.get("ssrd", {}).get(lt, np.nan)
        if not np.isnan(ssrd): result["Solar_Irradiance_GHI"][pos] = round(ssrd * 86400 / 3_600_000, 3)

        tp = extracted.get("tp", {}).get(lt, np.nan)
        if not np.isnan(tp): result["Precipitation"][pos] = round(tp * 1000 / days, 3)

        u, v = extracted.get("u10", {}).get(lt, np.nan), extracted.get("v10", {}).get(lt, np.nan)
        if not np.isnan(u) and not np.isnan(v): result["Wind_Speed"][pos] = round(math.sqrt(u**2 + v**2), 2)

    # Save cache
    rows = [{"leadtime": lt, "year": months_needed[i][0], "month": months_needed[i][1],
             **{col: result[col][month_pos[(months_needed[i][0], months_needed[i][1])]] for col in result}}
            for i, lt in enumerate(leadtimes)]
    pd.DataFrame(rows).to_csv(cache_csv, index=False)
    print(f"Cached: {cache_csv}")
    return result


# ===============================================================================
# MERGE  (C3S primary + NASA TMY fallback per cell)
# ===============================================================================

def _merge(c3s: Optional[dict], tmy: dict, months: list) -> pd.DataFrame:
    """Build weather DataFrame, filling NaN C3S cells with NASA TMY values."""
    raw_cols = ["Solar_Irradiance_GHI", "Temperature", "Max_Temperature",
                "Min_Temperature", "Humidity", "Precipitation", "Wind_Speed"]
    rows = []
    for i, (year, month) in enumerate(months):
        row, sources = {"Month": month}, set()
        for col in raw_cols:
            c3s_val  = (c3s or {}).get(col, [None]*12)[i]
            nasa_val = tmy.get(col, [np.nan]*12)[month - 1]
            if c3s_val is not None and not np.isnan(c3s_val):
                row[col] = c3s_val; sources.add("C3S")
            else:
                row[col] = nasa_val; sources.add("NASA_TMY")
        row["Clear_Sky_GHI"] = tmy.get("Clear_Sky_GHI", [np.nan]*12)[month - 1]
        row["data_source"] = ("C3S" if sources == {"C3S"} else
                              "NASA_TMY" if sources == {"NASA_TMY"} else "HYBRID")
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Max_Temperature"] = df["Max_Temperature"].fillna(df["Temperature"] + 3.0)
    df["Min_Temperature"] = df["Min_Temperature"].fillna(df["Temperature"] - 3.0)
    df["Humidity"]        = df["Humidity"].fillna(load_config()["physics"]["humidity_default"])
    return df


# ===============================================================================
# PUBLIC API
# ===============================================================================

def fetch_weather_forecast(
    start_year:     int  = None,
    start_month:    int  = None,
    n_months:       int  = 12,
    nasa_only:      bool = False,
    force_download: bool = False,
    tmp_dir:        str  = None,
) -> pd.DataFrame:
    """
    Fetch 12-month future weather for Maharagama (fallback: Colombo).

    Args:
        start_year     : forecast start year  (default: current year)
        start_month    : forecast start month (default: current month)
        n_months       : how many months to fetch (default: 12)
        nasa_only      : skip C3S, use NASA TMY only
        force_download : delete GRIB + cache CSV and re-download
        tmp_dir        : directory for GRIB files and CSV cache

    Returns:
        DataFrame — Month, Solar_Irradiance_GHI, Temperature, Max_Temperature,
                    Min_Temperature, Humidity, Precipitation, Wind_Speed,
                    Clear_Sky_GHI, data_source
    """
    today       = date.today()
    start_year  = start_year  or today.year
    start_month = start_month or today.month
    months      = _months_to_fetch(start_year, start_month, n_months)
    if tmp_dir is None:
        tmp_dir = _c3s_cfg()["tmp_dir"]

    print("\n" + "="*60)
    print("  FUTURE WEATHER FORECAST")
    print("="*60)
    print(f"  {n_months} months from {start_year}-{start_month:02d}")
    print(f"  Mode: {'NASA TMY only' if nasa_only else 'C3S + NASA fallback'}")

    # Step 1 — NASA TMY (always needed as C3S fallback)
    tmy = None
    for loc in _locations():
        try:
            tmy = fetch_tmy(loc["lat"], loc["lon"])
            print(f"NASA TMY from {loc['name']}")
            break
        except Exception as e:
            print(f"NASA {loc['name']}: {e}")
    if tmy is None:
        raise RuntimeError("NASA POWER failed for both locations. Check internet connection.")

    # Step 2 — C3S (optional)
    if force_download and not nasa_only:
        iy, im = today.year, today.month - 1
        if im < 1: im, iy = 12, iy - 1
        for p in [os.path.join(tmp_dir, "cache", f"seas5_{iy}{im:02d}_extracted.csv"),
                  os.path.join(tmp_dir, f"seas5_{iy}{im:02d}.grib")]:
            if os.path.exists(p):
                os.remove(p)
                print(f"  [force] Deleted: {p}")

    c3s = None
    if not nasa_only:
        for loc in _locations():
            try:
                c3s = _fetch_c3s(loc["lat"], loc["lon"],
                                  start_year, start_month, n_months, tmp_dir)
                if c3s:
                    print(f"C3S from {loc['name']}")
                    break
            except Exception as e:
                print(f"C3S {loc['name']}: {e}")

    # Step 3 — Merge
    df = _merge(c3s, tmy, months)
    print(f"  Sources: {df['data_source'].value_counts().to_dict()}")
    print(f"Weather ready — {len(df)} months\n")
    return df
