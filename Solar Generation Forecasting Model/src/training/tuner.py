"""
Hyperparameter tuning using GridSearchCV with GroupKFold.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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
            "rf__n_estimators":      [50, 100],
            "rf__max_depth":         [6, 8, 10],
            "rf__min_samples_split": [5, 10],
            "rf__min_samples_leaf":  [2, 4],
        },
    ),
    (
        "gb",
        GradientBoostingRegressor(random_state=42),
        {
            "gb__n_estimators":      [50, 100],
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