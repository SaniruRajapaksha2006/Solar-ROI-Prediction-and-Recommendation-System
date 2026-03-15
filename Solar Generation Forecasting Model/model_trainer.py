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

from src.data.splitter import DataSplitter
from src.models.similarity_engine import SimilarityEngine
from src.training.baseline import BaselineEvaluator
from src.training.tuner import ModelTuner
from src.training.evaluator import ModelEvaluator
from src.training.saver import ModelSaver


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / "data" / "processed" / "final.csv"

def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"final.csv not found at {DATA_PATH}\n"
            "Run data_pipeline.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {len(df):,} records  |  {df.shape[1]} columns")

    # -- 1. Split --------------------------------------------------
    X_train, X_test, y_train, y_test, groups_train = DataSplitter().split(df)

    # Keep df rows aligned to test indices — needed by baseline + similarity
    df_test_raw = df.loc[X_test.index]

    # -- 2. Physics + untuned baseline ----------------------------
    print("\n" + "=" * 60)
    print("STEP 2 - PHYSICS + BASELINE COMPARISON")
    print("=" * 60)
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
    print("\n" + "=" * 60)
    print("STEP 3 - SIMILARITY MATCHING  (Statistical baseline)")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("STEP 4 - ML HYPERPARAMETER TUNING  (GroupKFold)")
    print("=" * 60)

    tuner        = ModelTuner(n_folds=5)
    tuned_models = tuner.tune_all(X_train, y_train, groups_train)

    # -- 5. Evaluate ML models -------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 - ML EVALUATION + OVERFITTING CHECK")
    print("=" * 60)

    ml_results = ModelEvaluator().evaluate(
        tuned_models, X_test, y_test,
        cv_mae_scores=tuner.cv_mae_scores,
    )


    # -- 7. Save best ML model -------------------------------------
    best_name = ml_results.iloc[0]["Model"]
    ModelSaver().save(tuned_models, best_name)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print(f"  Best ML model : {best_name}")
    print(f"  Physics MAE   : {physics_mae:.4f} kWh/kW")
    print(f"  Similarity MAE: {sim_metrics['MAE']:.4f} kWh/kW")
    print(f"  Best ML MAE   : {ml_results.iloc[0]['Test MAE']:.4f} kWh/kW")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n[✗] Training failed: {e}")
        raise
