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