"""
model_trainer.py
================
Trains, evaluates, and compares all models. Saves the best ML pipeline.

Steps
-----
1. Split      Household-safe train/test (src/data/splitter.py)
2. Baseline   Physics formula vs untuned ML    (src/training/baseline.py)
3. Similarity KNN statistical model            (src/models/similarity_engine.py)
4. Tune       GridSearchCV + GroupKFold        (src/training/tuner.py)
5. Evaluate   MAE / RMSE / R² / overfit check  (src/training/evaluator.py)
6. Compare    Unified thesis comparison table
7. Save       Best ML pipeline → models/       (src/training/saver.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import json
from src.data.splitter import DataSplitter
from src.features.selection import MODEL_FEATURES
from src.preprocessing.outliers import OutlierRemover
from utils.utils_config import load_config
from src.models.similarity_engine import SimilarityEngine
from src.training.baseline import BaselineEvaluator
from src.training.tuner import ModelTuner
from src.training.evaluator import ModelEvaluator
from src.training.saver import ModelSaver
from utils.utils_config import load_config


SCRIPT_DIR = Path(__file__).resolve().parent


def _data_path() -> Path:
    cfg = load_config()
    return SCRIPT_DIR / cfg["paths"]["processed_dir"] / cfg["paths"]["final_dataset"]


# -- Unified comparison table ---------------------------------------------------

def _comparison_table(
    physics_mae:    float,
    physics_rmse:    float,
    physics_r2:     float,
    physics_mape:     float,
    similarity_metrics: dict,
    ml_results:     pd.DataFrame,
) -> None:
    """
    Print a single formatted table covering all three approaches.
    Intended for direct inclusion in the thesis results chapter.

    Approach order: Physics (deterministic) → Similarity (statistical) → ML (learned)
    """
    W = 78
    print("\n" + "=" * W)
    print("  MODEL COMPARISON TABLE")
    print("  Target: Efficiency (kWh/kW)  |  Lower MAE = Better")
    print("=" * W)
    print(f"  {'Model':<22} {'Approach':<16} {'MAE':>8} {'RMSE':>8} "
          f"{'R²':>7} {'MAPE%':>7}")
    print("  " + "-" * (W - 2))

    # Row 1 — Physics (deterministic)
    print(f"  {'Physics Formula':<22} {'Deterministic':<16} "
          f"{physics_mae:>8.4f} {physics_rmse:>8.4f} "
          f"{physics_r2:>7.4f} {physics_mape:>7.2f}")

    # Row 2 — Similarity matching (statistical)
    sm = similarity_metrics
    print(f"  {'SimilarityMatch':<22} {'Statistical':<16} "
          f"{sm['MAE']:>8.4f} {sm['RMSE']:>8.4f} "
          f"{sm['R²']:>7.4f} {sm['MAPE (%)']:>7.2f}")

    # Rows 3+ — ML models (tuned)
    print("  " + "-" * (W - 2))
    for _, row in ml_results.iterrows():
        flag = " ★" if row.name == ml_results.index[0] else ""
        print(f"  {row['Model'] + flag:<22} {'ML (tuned)':<16} "
              f"{row['Test MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['R²']:>7.4f} {row['MAPE (%)']:>7.2f}")

    print("  " + "-" * (W - 2))

    # Improvement summary
    best_ml_mae = ml_results.iloc[0]["Test MAE"]
    ml_vs_physics    = (physics_mae    - best_ml_mae) / physics_mae    * 100
    ml_vs_similarity = (sm["MAE"]      - best_ml_mae) / sm["MAE"]      * 100

    print(f"\n  Best ML vs Physics   : {ml_vs_physics:+.1f}% MAE improvement")
    print(f"  Best ML vs Similarity: {ml_vs_similarity:+.1f}% MAE improvement")
    print("=" * W)


# -- Main training flow ---------------------------------------------------------

def train() -> None:
    DATA_PATH = _data_path()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}\n"
            "Run data_pipeline.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {len(df):,} records  |  {df.shape[1]} columns")

    # -- 1. Split by household ------------------------------------
    splitter = DataSplitter()
    df_train, df_test = splitter.split(df)

    # -- 2. Outlier removal on train only --------------------------
    # Bounds fitted on train, saved for reproducibility. Test untouched.
    oc            = load_config()["outlier_detection"]
    artifacts_dir = SCRIPT_DIR / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    remover  = OutlierRemover()
    df_train = remover.fit_transform(df_train, column=oc["target_column"])
    remover.save(artifacts_dir / "outlier_remover.pkl")

    # Save split record for reproducibility
    split_record = {"train_accounts": sorted(df_train["ACCOUNT_NO"].unique().tolist()),
                    "test_accounts":  sorted(df_test["ACCOUNT_NO"].unique().tolist())}
    with open(artifacts_dir / "split_record.json", "w") as f:
        json.dump(split_record, f, indent=2)

    # Extract X / y / groups
    feature_cols = [col for col in MODEL_FEATURES if col in df_train.columns]
    X_train      = df_train[feature_cols]
    y_train      = df_train["Efficiency"]
    groups_train = df_train["ACCOUNT_NO"].reset_index(drop=True)
    X_test       = df_test[[col for col in MODEL_FEATURES if col in df_test.columns]]
    y_test       = df_test["Efficiency"]
    df_test_raw  = df_test

    # -- 2. Physics + untuned baseline ----------------------------
    print("\n" + "=" * 78)
    print("STEP 2 - PHYSICS + BASELINE COMPARISON")
    print("=" * 78)
    baseline = BaselineEvaluator()
    baseline.compare(X_train, X_test, y_train, y_test, df_test_raw=df_test_raw)

    # Extract physics MAE/R² for the comparison table
    physics_preds = df_test_raw["Physics_Pred"].values
    physics_mae   = mean_absolute_error(y_test, physics_preds)
    physics_r2    = r2_score(y_test, physics_preds)

    physics_rmse = np.sqrt(np.mean((y_test - physics_preds) ** 2))
    physics_mape = np.mean(
        np.abs((y_test.values - physics_preds) /
               np.where(y_test.values == 0, np.nan, y_test.values))
    ) * 100

    # -- 3. Similarity matching ------------------------------------
    print("\n" + "=" * 78)
    print("STEP 3 - SIMILARITY MATCHING  (Statistical baseline)")
    print("=" * 78)

    sim_engine = SimilarityEngine()   # reads n_neighbors, metric, features from config
    sim_engine.fit(X_train, y_train)
    sim_metrics = sim_engine.evaluate(X_test, y_test)

    print(f"\n  SimilarityMatch results:")
    print(f"    MAE      : {sim_metrics['MAE']:.4f} kWh/kW")
    print(f"    RMSE     : {sim_metrics['RMSE']:.4f}")
    print(f"    R²       : {sim_metrics['R²']:.4f}")
    print(f"    MAPE     : {sim_metrics['MAPE (%)']:.2f}%")
    print(f"\n  Interpretation: For each test month, the model found the")
    print(f"  {sim_engine.n_neighbors} most similar historical months by GHI,")
    print(f"  Temperature, and Cloud_Factor — and averaged their Efficiency.")

    # -- 4. Tune ML models -----------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 - ML HYPERPARAMETER TUNING  (GroupKFold)")
    print("=" * 78)

    tuner        = ModelTuner()
    tuned_models = tuner.tune_all(X_train, y_train, groups_train)

    # -- 5. Evaluate ML models -------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5 - ML EVALUATION + OVERFITTING CHECK")
    print("=" * 78)

    ml_results = ModelEvaluator().evaluate(
        tuned_models, X_test, y_test,
        cv_mae_scores=tuner.cv_mae_scores,
    )

    # -- 6. comparison table --------------------------------
    _comparison_table(
        physics_mae=physics_mae,
        physics_rmse=physics_rmse,
        physics_r2=physics_r2,
        physics_mape=physics_mape,
        similarity_metrics=sim_metrics,
        ml_results=ml_results,
    )

    # -- 7. Save best ML model -------------------------------------
    best_name = ml_results.iloc[0]["Model"]
    ModelSaver().save(tuned_models, best_name)

    print("\n" + "=" * 78)
    print("  TRAINING COMPLETE")
    print(f"  Best ML model : {best_name}")
    print(f"  Physics MAE   : {physics_mae:.4f} kWh/kW")
    print(f"  Similarity MAE: {sim_metrics['MAE']:.4f} kWh/kW")
    print(f"  Best ML MAE   : {ml_results.iloc[0]['Test MAE']:.4f} kWh/kW")
    print("=" * 78)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n Training failed: {e}")
        raise
