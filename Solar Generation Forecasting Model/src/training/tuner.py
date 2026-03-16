"""
src/training/tuner.py
Hyperparameter tuning using GridSearchCV with GroupKFold.

GroupKFold is mandatory here — the same household must not appear in
both the CV train and validation fold, otherwise CV MAE is optimistic
and the overfitting check in evaluator.py becomes meaningless.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils.utils_config import get_training_config


# Each entry: (estimator_step_name, estimator, param_grid)
_TUNING_CONFIGS = [
    (
        "ridge",
        Ridge(),
        {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    (
        "lasso",
        Lasso(max_iter=10_000),   # max_iter suppresses ConvergenceWarning
        {"lasso__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    ),
    (
        "svr",
        SVR(max_iter=5_000),      # max_iter suppresses ConvergenceWarning
        {
            "svr__C":       [1, 10, 100],
            "svr__epsilon": [0.01, 0.05, 0.1],
            "svr__kernel":  ["rbf"],
        },
    ),
    (
        "rf",
        RandomForestRegressor(random_state=42),
        {
            # n_estimators: 200 added — captures more trees without overfitting
            "rf__n_estimators":      [50, 100, 200],
            # max_depth: 3 added — shallow trees generalise better on noisy data;
            # None allows full growth for comparison
            "rf__max_depth":         [3, 5, 10, None],
            "rf__min_samples_split": [2, 5, 10],
        },
    ),
    (
        "gb",
        GradientBoostingRegressor(random_state=42),
        {
            "gb__n_estimators":      [50, 100, 200],
            "gb__max_depth":         [3, 4, 5],
            "gb__learning_rate":     [0.05, 0.1, 0.2],
            "gb__min_samples_split": [5, 10],
        },
    ),
]

_MODEL_NAMES = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "svr":   "SVR",
    "rf":    "RandomForest",
    "gb":    "GradientBoosting",
}


class ModelTuner:

    def __init__(self, n_folds: int = None):
        """
        Args:
            n_folds: GroupKFold folds. Default: from config training.n_folds (5).
        """
        self.n_folds = n_folds if n_folds is not None \
                       else get_training_config()["n_folds"]
        self.cv_mae_scores: dict[str, float] = {}

    def tune_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                 groups_train: pd.Series) -> dict[str, Pipeline]:
        """
        Tune all models with GroupKFold CV.

        Args:
            X_train       : Training feature matrix
            y_train       : Training target (Efficiency kWh/kW)
            groups_train  : ACCOUNT_NO Series aligned to X_train rows
                            (returned by HouseholdSplitter.split)

        Returns:
            {model_name: best_fitted_pipeline}  — ready for predict() calls
        """
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER TUNING  (GroupKFold, {self.n_folds} folds)")
        print("=" * 60)

        group_kfold  = GroupKFold(n_splits=self.n_folds)
        tuned_models = {}

        for i, (step_name, estimator, param_grid) in enumerate(_TUNING_CONFIGS, 1):
            display_name = _MODEL_NAMES[step_name]
            print(f"\n{i}. {display_name}...")

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                (step_name, estimator),
            ])

            grid = GridSearchCV(
                pipe, param_grid,
                cv=group_kfold,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train, groups=groups_train)

            best_cv_mae = -grid.best_score_
            tuned_models[display_name]        = grid.best_estimator_
            self.cv_mae_scores[display_name]  = best_cv_mae

            print(f"   Best params : {grid.best_params_}")
            print(f"   CV MAE      : {best_cv_mae:.4f} kWh/kW")

        print(f"\n {len(tuned_models)} models tuned")
        return tuned_models
