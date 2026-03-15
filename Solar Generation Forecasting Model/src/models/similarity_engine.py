import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.utils_config import load_config


def _sim_cfg() -> dict:
    return load_config()["similarity"]


class SimilarityEngine:
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


        X_scaled             = self._scaler.transform(X[self._similarity_features])
        distances, indices   = self._nn.kneighbors(X_scaled)

        # Mean Efficiency of k nearest historical months per query row
        predictions = np.array([
            self._train_efficiency[idx].mean()
            for idx in indices
        ])
        return predictions