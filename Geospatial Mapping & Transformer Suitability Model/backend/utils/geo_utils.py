"""
geo_utils.py
Geospatial helpers used throughout the assessment pipeline.
"""

from __future__ import annotations
import math
import pandas as pd
from geopy.distance import geodesic


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Fast haversine implementation (metres).
    Use geodesic() from geopy for production accuracy; haversine is ~50× faster.
    """
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geodesic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """High-accuracy geodesic distance (metres) via geopy."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def filter_nearby_transformers(
    transformer_df: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    radius_m: float,
    use_geodesic: bool = True,
) -> pd.DataFrame:
    """
    Add a DISTANCE_M column and return only rows within radius_m.

    Args:
        transformer_df: One row per transformer with TRANSFORMER_LAT / TRANSFORMER_LON.
        user_lat / user_lon: User's coordinates.
        radius_m: Search radius in metres.
        use_geodesic: True → use geopy (accurate); False → haversine (fast).

    Returns:
        Filtered DataFrame sorted by DISTANCE_M ascending, with DISTANCE_M column.
    """
    dist_fn = geodesic_distance_m if use_geodesic else haversine_distance_m

    transformer_df = transformer_df.copy()
    transformer_df['DISTANCE_M'] = transformer_df.apply(
        lambda row: dist_fn(user_lat, user_lon,
                            float(row['TRANSFORMER_LAT']),
                            float(row['TRANSFORMER_LON'])),
        axis=1,
    )

    nearby = transformer_df[transformer_df['DISTANCE_M'] <= radius_m].copy()
    nearby.sort_values('DISTANCE_M', inplace=True)
    return nearby


def bounding_box(lat: float, lon: float, radius_m: float):
    """
    Quick bounding-box pre-filter to avoid computing distance for every row.
    Returns (min_lat, max_lat, min_lon, max_lon).
    """
    lat_delta = math.degrees(radius_m / 6_371_000)
    lon_delta = math.degrees(radius_m / (6_371_000 * math.cos(math.radians(lat))))
    return (lat - lat_delta, lat + lat_delta,
            lon - lon_delta, lon + lon_delta)