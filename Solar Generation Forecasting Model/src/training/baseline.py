"""
Physics-based baseline comparison before tuning.

Compares naive Physics_Pred (GHI × 0.80 × Days) against quick untuned
sklearn models. Any ML model that cannot beat the physics baseline on
MAE does not justify its complexity.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Conservative default params
_BASE_MODELS = {
    "Ridge":    Ridge(alpha=1.0),
    "Lasso":    Lasso(alpha=0.1, max_iter=10_000),
    "SVR":      SVR(kernel="rbf", C=10, max_iter=5_000),
    "RandomForest": RandomForestRegressor(
        n_estimators=50, max_depth=8,
        min_samples_split=5, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=50, max_depth=4,
        learning_rate=0.1, random_state=42
    ),
}


class BaselineEvaluator:

    def compare(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                df_test_raw: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fit base models (untuned) and compare against physics baseline.

        Args:
            X_train, X_test  : Feature matrices
            y_train, y_test  : Efficiency targets (kWh/kW)
            df_test_raw      : Full df rows for the test split — needed to
                               extract Physics_Pred for the baseline MAE.
                               Pass None to skip the physics comparison.

        Returns:
            DataFrame sorted by MAE (best first)
        """
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON  (untuned models vs physics)")
        print("=" * 60)

        physics_mae = self._physics_baseline(y_test, df_test_raw)

        results = []
        for name, model in _BASE_MODELS.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) /
                                  y_test.replace(0, np.nan))) * 100

            vs_physics = ""
            if physics_mae is not None:
                vs_physics = "beats" if mae < physics_mae else "✗ loses"

            results.append({
                "Model":        name,
                "MAE (kWh/kW)": round(mae, 4),
                "RMSE":         round(rmse, 4),
                "R²":           round(r2, 4),
                "MAPE (%)":     round(mape, 2),
                "vs Physics":   vs_physics,
            })
            print(f"  {name:<20} MAE={mae:.4f}  R²={r2:.4f}  {vs_physics}")

        df_results = pd.DataFrame(results).sort_values("MAE (kWh/kW)")
        print("\n" + df_results.to_string(index=False))
        return df_results


    def _physics_baseline(self, y_test: pd.Series,
                          df_test_raw: pd.DataFrame) -> float | None:
        if df_test_raw is None or "Physics_Pred" not in df_test_raw.columns:
            print("  (Physics baseline skipped — pass df_test_raw to enable)\n")
            return None

        mae = mean_absolute_error(y_test, df_test_raw["Physics_Pred"])
        r2  = r2_score(y_test, df_test_raw["Physics_Pred"])
        print(f"\n  Physics baseline  MAE={mae:.4f} kWh/kW  R²={r2:.4f}")
        print(f"  ML models must beat MAE={mae:.4f} to justify complexity\n")
        return mae
