"""
Prepares raw CEB + NASA data into final.csv ready for model_trainer.py.

Steps
-----
1. Load      CEB solar records + NASA POWER weather
2. Impute    Fill missing GHI months from NASA 2018–2022 historical avg
3. Outliers  IQR-based removal on EXPORT_kWh per month
4. Engineer  Create all features + Efficiency target (= EXPORT_kWh / INV_CAPACITY)
5. Select    Drop non-informative and low-correlation columns
6. Filter    Keep residential accounts only (INV_CAPACITY ≤ 20 kW)
7. Save      Write data/processed/final.csv

"""

from datetime import datetime
from pathlib import Path

from utils.utils_config import load_config
from src.data.loader import DataLoader
from src.preprocessing.missing import MissingValueHandler
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector


SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_RAW_DIR  = SCRIPT_DIR / "data" / "raw"
DATA_PROC_DIR = SCRIPT_DIR / "data" / "processed"


def run_pipeline(save_intermediates: bool = False) -> None:
    cfg   = load_config()
    loc  = cfg["location"]["primary"]
    paths = cfg["paths"]
    imp   = cfg["imputation"]
    res   = cfg["residential"]
    fs    = cfg["feature_selection"]

    print("=" * 60)
    print("  DATA PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    loader = DataLoader()

    # -- STEP 1: Load ----------------------------------------------
    print("\nSTEP 1 - LOAD")
    print("-" * 40)

    ceb_df       = loader.load_ceb_data(PROJECT_ROOT / paths["ceb_dataset"])
    weather_path = SCRIPT_DIR / paths["weather_cache"]

    if weather_path.exists():
        weather_df = loader.load_local_weather_data(weather_path)
    else:
        print("  Weather CSV not found — fetching from NASA POWER...")
        weather_df = loader.fetch_weather_data(
            latitude=loc["lat"],
            longitude=loc["lon"],
            start_yr=imp["start_year"],
            end_yr=imp["end_year"],
            save_path=weather_path,
        )

    df = loader.merge_ceb_weather(ceb_df, weather_df)

    # -- STEP 2: Impute missing GHI --------------------------------
    print("\nSTEP 2 - MISSING VALUE IMPUTATION")
    print("-" * 40)

    handler       = MissingValueHandler()
    missing_stats = handler.analyse(df)
    if not missing_stats.empty:
        df = handler.impute_ghi(
            df, missing_stats,
            latitude=loc["lat"],
            longitude=loc["lon"],
            impute_months=imp["months"],
            start_yr=imp["start_year"],
            end_yr=imp["end_year"],
        )

    if save_intermediates:
        loader.save(df, DATA_PROC_DIR / "01_imputed.csv")

    # -- STEP 3: Feature engineering + target ---------------------
    print("\nSTEP 3 - FEATURE ENGINEERING + TARGET")
    print("-" * 40)

    df = FeatureEngineer().create_all_features(df)

    # -- STEP 4: Feature selection ---------------------------------
    print("\nSTEP 4 - FEATURE SELECTION")
    print("-" * 40)

    df = FeatureSelector().select_features(
        df, correlation_threshold=fs["correlation_threshold"]
    )

    # -- STEP 5: Residential filter --------------------------------
    print("\nSTEP 5 - RESIDENTIAL FILTER")
    print("-" * 40)

    before = len(df)
    df     = df[df["INV_CAPACITY"] <= res["max_capacity_kw"]].copy()
    print(f"  Removed {before - len(df):,} records > {res['max_capacity_kw']} kW")
    print(f"  Remaining: {len(df):,}")

    # -- STEP 6: Save ----------------------------------------------
    print("\nSTEP 6 - SAVE")
    print("-" * 40)

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    loader.save(df, DATA_PROC_DIR / paths["final_dataset"])

    print("\n" + "=" * 60)
    print(f"  Done — {len(df):,} records · {df.shape[1]} columns")
    print(f"  Output : data/processed/{paths['final_dataset']}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[✗] Pipeline failed: {e}")
        raise
