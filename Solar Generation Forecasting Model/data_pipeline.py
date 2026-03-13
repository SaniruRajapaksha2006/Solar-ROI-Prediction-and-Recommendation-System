"""
data_pipeline.py
===========================================================================
Solar Forecasting - Data Preparation Pipeline

Steps:
  1. Load     - CEB export data + NASA POWER weather
  2. Merge    - Join on Month
  3. Impute   - Fill missing GHI months with historical averages
  4. Outliers - IQR-based removal on EXPORT_kWh
  5. Target   - Create Efficiency = EXPORT_kWh / INV_CAPACITY
  6. Features - Engineer Temp_Efficiency, Cloud_Factor, Physics_Pred, etc.
  7. Select   - Drop non-informative / redundant columns (EXPORT_kWh kept for trainer)
  8. Filter   - Remove commercial accounts (>20 kW)
  9. Save     - Write final.csv ready for model_trainer.py
"""

from datetime import datetime
from pathlib import Path

from src.data_load import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.handle_missing import HandleMissing
from src.outlier_detection import OutlierDetector
from utils.utils_config import load_config

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def main():
    print("\n" + "=" * 70)
    print("  SOLAR FORECASTING - DATA PREPARATION PIPELINE")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg  = load_config()
    nasa = cfg["nasa"]
    p    = cfg["paths"]
    loc  = cfg["location"]["primary"]

    PROCESSED_DIR = SCRIPT_DIR / p["processed_dir"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_intermediates = p.get("save_intermediates", False)

    loader  = DataLoader()
    handler = HandleMissing()

    # -- STEP 1: Load -----------------------------------------------------------
    print("STEP 1 - DATA LOADING")
    print("-" * 40)

    ceb_data     = loader.load_ceb_data(PROJECT_ROOT / p["ceb_dataset"])
    weather_path = SCRIPT_DIR / p["weather_cache"]

    if weather_path.exists():
        weather_data = loader.load_local_weather_data(weather_path)
    else:
        print("  Cache not found - fetching from NASA POWER...")
        weather_data = loader.fetch_weather_data(
            latitude=loc["lat"], longitude=loc["lon"],
            start_yr=2025, end_yr=2025,
            save=True, save_path=weather_path,
        )

    df = loader.merge_ceb_weather(ceb_df=ceb_data, weather_df=weather_data)

    # -- STEP 2: Impute Missing --------------------------------------------------
    print("\nSTEP 2 - MISSING VALUE IMPUTATION")
    print("-" * 40)

    missing_stats = handler.analysis_missing(df)
    if not missing_stats.empty:
        df = handler.impute_missing(
            data=df,
            missing_stats=missing_stats,
            latitude=loc["lat"], longitude=loc["lon"],
            impute_months=nasa["impute_months"],
            start_yr=nasa["impute_start_year"],
            end_yr=nasa["impute_end_year"],
        )
        print("  Post-imputation check:")
        handler.analysis_missing(df)

    if save_intermediates:
        loader.save_data(df, PROCESSED_DIR / "01_imputed.csv")

    # -- STEP 3: Outlier Removal ----------------------------------------------
    print("\nSTEP 3 - OUTLIER REMOVAL")
    print("-" * 40)

    oc  = cfg["outlier_detection"]
    df  = OutlierDetector(oc["iqr_threshold"]).detect_monthly_outliers(
              df, oc["target_column"], oc["iqr_threshold"])

    # -- STEP 4: Efficiency Target
    print("\nSTEP 4 - EFFICIENCY TARGET")
    print("-" * 40)

    df["Efficiency"] = (df["EXPORT_kWh"] / df["INV_CAPACITY"]).round(4)
    print(f"  Efficiency (kWh/kW)  mean={df['Efficiency'].mean():.2f}  "
          f"min={df['Efficiency'].min():.2f}  max={df['Efficiency'].max():.2f}")

    # -- STEP 5: Feature Engineering ------------------------------------------
    print("\nSTEP 5 - FEATURE ENGINEERING")
    print("-" * 40)

    fe           = FeatureEngineer()
    df           = fe.create_all_features(df)
    new_features = fe.get_feature_list()

    # -- STEP 6: Feature Selection -------------------------------------------
    print("\nSTEP 6 - FEATURE SELECTION")
    print("-" * 40)

    fs        = cfg["feature_selection"]
    keep_cols = (
        ["Efficiency", "EXPORT_kWh", "ACCOUNT_NO", "Month", "INV_CAPACITY"]
        + list(nasa["params"].values())
        + new_features
    )

    df = FeatureSelector().select_features(
        df=df,
        target="Efficiency",
        correlation_threshold=fs["correlation_threshold"],
        non_informative_cols=fs["drop_columns"],
        keep_cols=keep_cols,
    )

    # -- STEP 7: Residential Filter -------------------------------------------
    print("\nSTEP 7 - RESIDENTIAL FILTER")
    print("-" * 40)

    max_kw = cfg["residential"]["max_capacity_kw"]
    before = len(df)
    df     = df[df["INV_CAPACITY"] <= max_kw].copy()
    print(f"  Removed {before - len(df):,} accounts > {max_kw} kW  "
          f"| Remaining: {len(df):,}")

    # -- STEP 8: Save --------------------------------------------------------
    print("\nSTEP 8 - SAVE")
    print("-" * 40)

    loader.save_data(df, PROCESSED_DIR / p["final_dataset"])

    print("\n" + "=" * 70)
    print(f"  Done - {len(df):,} records · {df.shape[1]} features")
    print(f"  Output → {PROCESSED_DIR / p['final_dataset']}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[✗] Pipeline failed: {e}")
        raise
