"""
Predicts monthly solar export and income for a given inverter capacity.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.utils_config import load_config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.features.selection import MODEL_FEATURES
from src.training.saver import ModelSaver


SCRIPT_DIR   = Path(__file__).resolve().parent


def _model_path() -> Path:
    return SCRIPT_DIR / load_config()["paths"]["model"]

def _weather_path() -> Path:
    return SCRIPT_DIR / load_config()["paths"]["weather_cache"]

MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


# -- Core prediction ------------------------------------------------------------

def predict_monthly(inv_capacity_kw: float,
                    custom_tariff: float = None) -> pd.DataFrame:
    """
    Predict monthly solar export and income for a given inverter capacity.

    Args:
        inv_capacity_kw : System size in kW
        custom_tariff   : LKR/kWh tariff override. If None, uses config default
                          (net_plus_tariff_lkr)

    Returns:
        DataFrame — Month, Predicted_Export_kWh, Lower_kWh, Upper_kWh,
                    Confidence_Pct, Estimated_Income_LKR
    """
    cfg        = load_config()
    max_kw     = cfg["residential"]["max_capacity_kw"]
    if not (0 < inv_capacity_kw <= max_kw):
        raise ValueError(
            f"inv_capacity_kw must be between 0 and {max_kw} kW. "
            f"Got: {inv_capacity_kw}"
        )

    tariff     = custom_tariff if custom_tariff is not None \
                 else cfg["roi"]["net_plus_tariff_lkr"]
    mae_per_kw = cfg["roi"]["mae_kwh_per_kw"]

    pipeline   = ModelSaver().load(_model_path())
    weather_df = DataLoader().load_local_weather_data(_weather_path())

    # Build feature matrix — no EXPORT_kWh at inference (inference path)
    weather_df["INV_CAPACITY"] = inv_capacity_kw
    weather_df = FeatureEngineer().create_all_features(weather_df)

    # Predict Efficiency (kWh/kW) → scale to Export kWh
    efficiency = pipeline.predict(weather_df[MODEL_FEATURES]).clip(min=0)
    export_kwh = efficiency * inv_capacity_kw
    mae_kwh    = mae_per_kw * inv_capacity_kw

    confidence = np.where(
        export_kwh > 0,
        (1 - mae_kwh / export_kwh).clip(0, 1) * 100,
        0.0,
    ).round(1)

    return pd.DataFrame({
        "Month":                weather_df["Month"].values,
        "Predicted_Export_kWh": export_kwh.round(2),
        "Lower_kWh":            (export_kwh - mae_kwh).clip(min=0).round(2),
        "Upper_kWh":            (export_kwh + mae_kwh).round(2),
        "Confidence_Pct":       confidence,
        "Estimated_Income_LKR": (export_kwh * tariff).round(2),
    })


# -- Display --------------------------------------------------------------------

def _print_results(df: pd.DataFrame, inv_capacity_kw: float,
                   tariff_used: float) -> None:
    W = 72
    cfg            = load_config()
    default_tariff = cfg["roi"]["net_plus_tariff_lkr"]
    is_custom      = (tariff_used != default_tariff)

    print("\n" + "═" * W)
    print(f"  SOLAR FORECAST  —  {inv_capacity_kw} kW  |  Tariff: LKR {tariff_used:.2f}/kWh"
          + (" (custom override)" if is_custom else " (Net Plus, Jul 2024 — PUCSL)"))
    print("═" * W)
    print(f"  {'Month':<6} {'Export kWh':>11} {'Lower':>8} {'Upper':>8}"
          f" {'Confidence':>11} {'Income LKR':>12}")
    print("  " + "-" * (W - 2))

    for _, r in df.iterrows():
        m = MONTH_NAMES[int(r["Month"]) - 1]
        print(f"  {m:<6} {r['Predicted_Export_kWh']:>11.1f}"
              f" {r['Lower_kWh']:>8.1f} {r['Upper_kWh']:>8.1f}"
              f" {r['Confidence_Pct']:>10.1f}%"
              f" {r['Estimated_Income_LKR']:>12,.2f}")

    print("  " + "-" * (W - 2))
    print(f"  {'Annual':<6} {df['Predicted_Export_kWh'].sum():>11.1f}"
          f" {'':>8} {'':>8}"
          f" {df['Confidence_Pct'].mean():>10.1f}%"
          f" {df['Estimated_Income_LKR'].sum():>12,.2f}")
    print("═" * W + "\n")


# -- CLI ------------------------------------------------------------------------

def main() -> None:
    if not _model_path().exists():
        print(f"Model not found: {_model_path()}\n    Run model_trainer.py first.")
        sys.exit(1)
    if not _weather_path().exists():
        print(f"Weather data not found: {_weather_path()}\n    Run data_pipeline.py first.")
        sys.exit(1)

    cfg    = load_config()
    max_kw = cfg["residential"]["max_capacity_kw"]
    default_tariff = cfg["roi"]["net_plus_tariff_lkr"]

    # -- Accept args from command line or prompt interactively -----
    # Usage: python predict.py <capacity_kw> [tariff_lkr]
    # Example: python predict.py 5.0 40.00
    if len(sys.argv) >= 2:
        try:
            kw = float(sys.argv[1])
        except ValueError:
            print("  Error: capacity must be a number.  e.g. python predict.py 5.0")
            sys.exit(1)
        custom_tariff = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    else:
        while True:
            try:
                kw = float(input(f"\n  Enter inverter capacity in kW (1–{max_kw}): ").strip())
                if 0 < kw <= max_kw:
                    break
                print(f"  Must be between 0 and {max_kw} kW.")
            except ValueError:
                print("  Please enter a valid number.")
        custom_tariff = None

    if not (0 < kw <= max_kw):
        print(f"  Capacity {kw} kW is outside the valid range (0–{max_kw} kW).")
        sys.exit(1)

    tariff_used = custom_tariff if custom_tariff is not None else default_tariff
    if custom_tariff is not None:
        print(f"\n  Using dynamic tariff  : LKR {custom_tariff:.2f}/kWh")
    else:
        print(f"\n  Using config tariff   : LKR {default_tariff:.2f}/kWh  (pass as 2nd arg to override)")

    results = predict_monthly(kw, custom_tariff=custom_tariff)
    _print_results(results, kw, tariff_used)


if __name__ == "__main__":
    main()
