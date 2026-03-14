"""
Trains and evaluates all models, saves the best pipeline.

Steps
-----
1. Split     Household-safe train/test via GroupShuffleSplit
2. Baseline  Untuned models vs physics benchmark
3. Tune      GridSearchCV with GroupKFold
4. Evaluate  MAE, RMSE, R², overfitting check
5. Save      Best pipeline → models/best_solar_pipeline.pkl
"""

from pathlib import Path

import pandas as pd

from src.data.splitter import DataSplitter
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

    test_mask   = df.index.isin(X_test.index)
    df_test_raw = df[test_mask]

    # -- 2. Baseline -----------------------------------------------
    BaselineEvaluator().compare(
        X_train, X_test, y_train, y_test,
        df_test_raw=df_test_raw,
    )

    # -- 3. Tune ---------------------------------------------------
    tuner        = ModelTuner(n_folds=5)
    tuned_models = tuner.tune_all(X_train, y_train, groups_train)

    # -- 4. Evaluate -----------------------------------------------
    results = ModelEvaluator().evaluate(
        tuned_models, X_test, y_test,
        cv_mae_scores=tuner.cv_mae_scores,
    )

    # -- 5. Save ---------------------------------------------------
    best_name = results.iloc[0]["Model"]
    ModelSaver().save(tuned_models, best_name)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print(f"  Best model : {best_name}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n[✗] Training failed: {e}")
        raise
