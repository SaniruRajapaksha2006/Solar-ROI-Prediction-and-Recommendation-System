import numpy as np
import pandas as pd
import requests

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

NASA_MONTHLY_URL     = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_CLIMATOLOGY_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# ── PARAMETER MAP ──────────────────────────

NASA_PARAMS = {
    "ALLSKY_SFC_SW_DWN": "Solar_Irradiance_GHI",   # kWh/m²/day
    "T2M":               "Temperature",             # °C  (monthly mean)
    "T2M_MAX":           "Max_Temperature",         # °C
    "T2M_MIN":           "Min_Temperature",         # °C
    "RH2M":              "Humidity",                # %
    "PRECTOTCORR":       "Precipitation",           # mm/day
    "WS10M":             "Wind_Speed",              # m/s
    "CLRSKY_SFC_SW_DWN": "Clear_Sky_GHI",          # kWh/m²/day
}

# Fill value NASA uses for missing data
NASA_FILL = -999.0


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _clean(value: float) -> float:
    """Replace NASA fill value with NaN."""
    return np.nan if value <= -990 else value


def _param_str(params: dict = None) -> str:
    """Build comma-separated parameter string. Defaults to all NASA_PARAMS."""
    keys = params.keys() if params else NASA_PARAMS.keys()
    return ",".join(keys)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def fetch_monthly(
    lat: float,
    lon: float,
    start_yr: int,
    end_yr: int,
    params: dict = None,
    community: str = "RE",
    timeout: int = 60,
) -> dict:
    """
    Fetch NASA POWER monthly data for a specific year range.

    Args:
        lat, lon   : coordinates
        start_yr   : first year (e.g. 2014)
        end_yr     : last year  (e.g. 2024)
        params     : {nasa_key: col_name} — defaults to NASA_PARAMS
        community  : NASA community string (default "RE" for renewable energy)
        timeout    : request timeout in seconds

    Returns:
        dict  (raw NASA JSON)

    Raises:
        ConnectionError on network failure
    """
    if params is None:
        params = NASA_PARAMS

    url = (
        f"{NASA_MONTHLY_URL}?"
        f"parameters={_param_str(params)}&"
        f"community={community}&"
        f"longitude={lon}&"
        f"latitude={lat}&"
        f"start={start_yr}&"
        f"end={end_yr}&"
        f"format=JSON"
    )

    print(f"  Fetching NASA POWER monthly  (lat={lat}, lon={lon}, {start_yr}–{end_yr})...")

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        print(f"NASA monthly data received")
        return data
    except requests.RequestException as e:
        raise ConnectionError(f"NASA POWER monthly request failed: {e}")


def label_monthly(data: dict, params: dict = None, year: int = 2025) -> pd.DataFrame:
    """
    Convert raw NASA monthly JSON -> labeled DataFrame (one row per month).

    Args:
        data   : raw JSON from fetch_monthly()
        params : {nasa_key: col_name} — defaults to NASA_PARAMS
        year   : year to label (default 2025)

    Returns:
        DataFrame with columns: Month, Solar_Irradiance_GHI, Temperature, ...
    """
    if params is None:
        params = NASA_PARAMS

    param_data = data["properties"]["parameter"]
    rows = []

    for month in range(1, 13):
        date_key = f"{year}{month:02d}"
        row = {"Month": month}
        for nasa_key, col_name in params.items():
            raw_val = param_data.get(nasa_key, {}).get(date_key, np.nan)
            row[col_name] = _clean(raw_val) if isinstance(raw_val, (int, float)) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_tmy(
    lat: float,
    lon: float,
    params: dict = None,
    community: str = "RE",
    timeout: int = 60,
) -> dict:
    """
    Fetch NASA POWER Typical Meteorological Year (TMY) climatology.

    Args:
        lat, lon   : coordinates
        params     : {nasa_key: col_name} — defaults to NASA_PARAMS
        community  : NASA community string (default "RE")
        timeout    : request timeout in seconds

    Returns:
        dict  {col_name: [jan_val, feb_val, ..., dec_val]}   (12 values each)

    Raises:
        ConnectionError on network failure
    """
    if params is None:
        params = NASA_PARAMS

    api_params = {
        "parameters": _param_str(params),
        "community":  community,
        "longitude":  lon,
        "latitude":   lat,
        "format":     "JSON",
    }

    print(f"  Fetching NASA POWER TMY  (lat={lat}, lon={lon})...")

    try:
        resp = requests.get(NASA_CLIMATOLOGY_URL, params=api_params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"NASA POWER TMY request failed: {e}")

    raw = data["properties"]["parameter"]
    month_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    tmy = {}

    for nasa_key, col_name in params.items():
        monthly = raw.get(nasa_key, {})
        tmy[col_name] = [_clean(monthly.get(k, np.nan)) for k in month_keys]

    print(f"NASA TMY fetched — 12 monthly means for {len(tmy)} variables")
    return tmy
