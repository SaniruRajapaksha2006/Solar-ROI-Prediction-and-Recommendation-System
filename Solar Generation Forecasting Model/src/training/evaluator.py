"""
Evaluates tuned models on the test set and checks for overfitting.

Overfitting check
-----------------
CV MAE   = average MAE across GroupKFold folds
Test MAE = MAE on the held-out households
Gap      = Test MAE − CV MAE

  Gap > 15 kWh/kW  ->  OVERFIT   (model memorised training households)
  Gap < −5 kWh/kW  ->  CV leak?  (CV was suspiciously optimistic)
  Otherwise        ->  stable
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


OVERFIT_THRESHOLD  =  15.0   # kWh/kW gap that flags overfitting
CV_LEAK_THRESHOLD  =  -5.0   # negative gap that suggests CV leakage


class ModelEvaluator:

    def evaluate(self, tuned_models: dict[str, Pipeline],
                 X_test: pd.DataFrame, y_test: pd.Series,
                 cv_mae_scores: dict[str, float] = None) -> pd.DataFrame:
        """
        Score every tuned model on the test split and report overfitting.

        Args:
            tuned_models   : {name: pipeline} from ModelTuner.tune_all()
            X_test         : Test feature matrix
            y_test         : Test target (Efficiency kWh/kW)
            cv_mae_scores  : {name: cv_mae} from ModelTuner.cv_mae_scores
                             Pass None to skip overfitting check.

        Returns:
            DataFrame sorted by Test MAE (best model first)
        """
        print("\n" + "=" * 60)
        print("EVALUATION + OVERFITTING CHECK")
        print("=" * 60)

        cv_scores = cv_mae_scores or {}
        rows      = []

        for name, pipeline in tuned_models.items():
            preds = pipeline.predict(X_test)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs(
                (y_test - preds) / y_test.replace(0, np.nan)
            )) * 100

            cv_mae = cv_scores.get(name)
            gap    = round(mae - cv_mae, 4) if cv_mae is not None else None

            if gap is None:
                overfit = "—"
            elif gap > OVERFIT_THRESHOLD:
                overfit = "OVERFIT"
            elif gap < CV_LEAK_THRESHOLD:
                overfit = "CV leak?"
            else:
                overfit = "stable"

            rows.append({
                "Model":    name,
                "CV MAE":   round(cv_mae, 4) if cv_mae is not None else "—",
                "Test MAE": round(mae, 4),
                "Gap":      round(gap, 4) if gap is not None else "—",
                "Overfit?": overfit,
                "RMSE":     round(rmse, 4),
                "R²":       round(r2, 4),
                "MAPE (%)": round(mape, 2),
            })

        df = pd.DataFrame(rows).sort_values("Test MAE")

        print("\n" + df.to_string(index=False))

        best = df.iloc[0]
        print(f"\n  Best model : {best['Model']}")
        print(f"  Test MAE   : {best['Test MAE']:.4f} kWh/kW")
        print(f"  R²         : {best['R²']:.4f}")
        print(f"  Overfit    : {best['Overfit?']}")

        return df
