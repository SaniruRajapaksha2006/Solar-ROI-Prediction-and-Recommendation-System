"""
Similarity Matching Model — a statistical (non-ML) approach that predicts
solar Efficiency by finding the most historically similar weather months.

How it works
-------------
1. FIT   : Stores the training set (historical months with known Efficiency).
2. PREDICT: For each query row (future weather month), computes Euclidean
            distance against all historical rows using the 3 key weather
            signals: GHI, Temperature, Cloud_Factor.
3. RESULT : Predicted Efficiency = mean Efficiency of the k nearest neighbours.

Why these 3 features?
  GHI          — primary driver of solar generation
  Temperature  — degrades panel efficiency via temperature coefficient
  Cloud_Factor — captures cloud cover independently of GHI magnitude

If it performs well, it shows the historical data has
strong self-similarity. If ML beats it, it proves ML captures interactions
that simple distance cannot.

Comparison table position:
  Method            | Approach      | Expected MAE | Key claim
  ------------------┼---------------┼--------------┼--------------------------
  Physics Formula   | Deterministic | Highest      | GHI × 0.80 × days
  Similarity Match  | Statistical   | Medium       | Historical analogue search
  ML (best model)   | Learned       | Lowest       | Non-linear feature weights
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.utils_config import get_similarity_config


def _sim_cfg() -> dict:
    return get_similarity_config()


class SimilarityEngine:
    """
    K-Nearest Neighbours similarity model for solar Efficiency prediction.
    """

    def __init__(self, n_neighbors: int = None, metric: str = None):
        """
        Args:
            n_neighbors : k neighbours to average. Default: from config similarity.n_neighbors
            metric      : Distance metric. Default: from config similarity.metric
        """
        cfg = _sim_cfg()
        self.n_neighbors      = n_neighbors or cfg["n_neighbors"]
        self.metric           = metric      or cfg["metric"]
        self._similarity_features = cfg["features"]   # read from config
        self._scaler          = StandardScaler()
        self._nn              = NearestNeighbors(n_neighbors=self.n_neighbors,
                                                 metric=self.metric)
        self._train_efficiency: np.ndarray | None = None
        self._is_fitted = False

    # -- Fit -------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SimilarityEngine":
        """
        Memorise the training set.

        Args:
            X_train : Training features (must contain self._similarity_features)
            y_train : Training target (Efficiency kWh/kW)

        Returns:
            self (for chaining)
        """
        self._validate_features(X_train)

        X_scaled = self._scaler.fit_transform(X_train[self._similarity_features])
        self._nn.fit(X_scaled)
        self._train_efficiency = y_train.values
        self._is_fitted = True

        print(f"  SimilarityEngine fitted on {len(y_train):,} historical months")
        print(f"  k={self.n_neighbors}  metric={self.metric}  "
              f"features={self._similarity_features}")
        return self

    # -- Predict ---------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict Efficiency for each row in X.

        For each query row:
          1. Scale features using the training scaler
          2. Find k nearest neighbours in training set
          3. Return mean Efficiency of those k neighbours

        Args:
            X : Feature DataFrame

        Returns:
            np.ndarray of predicted Efficiency values (kWh/kW)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        self._validate_features(X)

        X_scaled             = self._scaler.transform(X[self._similarity_features])
        distances, indices   = self._nn.kneighbors(X_scaled)

        # Mean Efficiency of k nearest historical months per query row
        predictions = np.array([
            self._train_efficiency[idx].mean()
            for idx in indices
        ])
        return predictions

    def predict_with_details(self, X: pd.DataFrame,
                             X_train_ref: pd.DataFrame = None) -> pd.DataFrame:
        """
        Predict and return a detailed DataFrame showing which historical
        months were matched for each query

        Args:
            X           : Query rows
            X_train_ref : Original training DataFrame to show matched rows.
                          Pass None to skip details.

        Returns:
            DataFrame with columns:
                query_idx, predicted_Efficiency,
                neighbour_1_idx, neighbour_1_dist, neighbour_1_Efficiency,
                ... (repeated for each neighbour)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_with_details().")

        self._validate_features(X)
        X_scaled           = self._scaler.transform(X[self._similarity_features])
        distances, indices = self._nn.kneighbors(X_scaled)

        rows = []
        for q_i, (dists, idxs) in enumerate(zip(distances, indices)):
            row = {
                "query_idx":            q_i,
                "predicted_Efficiency": self._train_efficiency[idxs].mean().round(4),
            }
            for rank, (d, idx) in enumerate(zip(dists, idxs), 1):
                row[f"match_{rank}_train_idx"]  = idx
                row[f"match_{rank}_distance"]   = round(d, 4)
                row[f"match_{rank}_Efficiency"] = round(self._train_efficiency[idx], 4)
                if X_train_ref is not None:
                    for feat in self._similarity_features:
                        row[f"match_{rank}_{feat}"] = round(
                            X_train_ref.iloc[idx][feat], 4
                        )
            rows.append(row)

        return pd.DataFrame(rows)

    # -- Evaluate --------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame,
                 y_test: pd.Series) -> dict:
        """
        Score the model on a test set.

        Args:
            X_test : Test features
            y_test : True Efficiency values

        Returns:
            dict with MAE, RMSE, R², MAPE
        """
        preds = self.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        mape = np.mean(
            np.abs((y_test.values - preds) /
                   np.where(y_test.values == 0, np.nan, y_test.values))
        ) * 100

        return {
            "Model":    "SimilarityMatch",
            "Approach": "Statistical",
            "MAE":      round(mae, 4),
            "RMSE":     round(rmse, 4),
            "R²":       round(r2, 4),
            "MAPE (%)": round(mape, 2),
        }

    # -- Private ---------------------------------------------------------------

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(self._similarity_features) - set(X.columns)
        if missing:
            raise ValueError(
                f"SimilarityEngine requires these features: {self._similarity_features}\n"
                f"Missing from input: {missing}"
            )
