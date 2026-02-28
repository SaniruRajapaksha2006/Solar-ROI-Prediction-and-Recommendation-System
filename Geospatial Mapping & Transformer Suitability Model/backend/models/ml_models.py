"""
ml_models.py
Machine-learning components:
  - SolarSuitabilityMLModel  (Random Forest — overload / suitability prediction)
  - TransformerClusterer     (K-Means — unsupervised risk segmentation)
  - LoadForecastModel        (Linear-Regression trend — 12-month load projection)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# ─── 1. Suitability / overload classifier ───────────────────────────────────

class SolarSuitabilityMLModel:
    """
    Random-Forest classifier that predicts whether a transformer can safely
    host additional solar capacity.

    Label convention (used during training):
        1 → suitable  (utilisation_rate < 0.75)
        0 → at-risk
    """

    FEATURE_COLUMNS = [
        'current_load_kW',
        'total_solar_capacity',
        'utilization_rate',
        'solar_penetration',
        'demand_volatility',
        'available_headroom',
        'export_ratio',
    ]

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        )
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[self.FEATURE_COLUMNS], y)
        self.is_trained = True
        print("✓ SolarSuitabilityMLModel trained")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of class=1 (suitable) for each row."""
        if not self.is_trained:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X[self.FEATURE_COLUMNS])[:, 1]

    def feature_importance(self) -> dict:
        if not self.is_trained:
            return {}
        pairs = zip(self.FEATURE_COLUMNS, self.model.feature_importances_)
        return dict(sorted(pairs, key=lambda x: x[1], reverse=True))


# ─── 2. Unsupervised transformer risk clustering ─────────────────────────────

CLUSTER_PROFILE_NAMES = {
    0: 'Underutilised — Low Risk',
    1: 'Balanced Load — Medium Risk',
    2: 'High Utilisation — High Risk',
}


class TransformerClusterer:
    """
    K-Means clustering that segments transformers into three risk tiers.
    Cluster IDs are re-mapped after fitting so that cluster 0 always corresponds
    to the lowest average utilisation (ascending risk order).
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self._label_map: dict = {}

    def fit_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        raw_labels = self.kmeans.fit_predict(X_scaled)

        # Re-map so cluster 0 = lowest utilisation centroid (index 2 of feature list)
        centroid_util = self.kmeans.cluster_centers_[:, 2]  # utilization_rate column
        rank_order = np.argsort(centroid_util)
        self._label_map = {orig: new for new, orig in enumerate(rank_order)}
        return np.vectorize(self._label_map.get)(raw_labels)

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        raw = self.kmeans.predict(X_scaled)
        return np.vectorize(self._label_map.get)(raw)

    @staticmethod
    def cluster_names() -> dict:
        return CLUSTER_PROFILE_NAMES


# ─── 3. Load trend forecaster ────────────────────────────────────────────────

class LoadForecastModel:
    """
    Simple linear-regression trend fitted on per-transformer time-series averages.
    Used to project current_load_kW N months into the future.
    """

    ANNUAL_GROWTH_FALLBACK = 0.03  # 3 % per year if model not trained

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, time_index: np.ndarray, load_values: np.ndarray):
        X = time_index.reshape(-1, 1)
        self.model.fit(X, load_values)
        self.is_trained = True

    def predict_future_load(self, current_load: float, months_ahead: int = 12) -> float:
        if not self.is_trained:
            factor = (1 + self.ANNUAL_GROWTH_FALLBACK) ** (months_ahead / 12)
            return current_load * factor
        X_future = np.array([[months_ahead]])
        prediction = float(self.model.predict(X_future)[0])
        # Clamp to realistic range
        return max(prediction, current_load * 0.9)