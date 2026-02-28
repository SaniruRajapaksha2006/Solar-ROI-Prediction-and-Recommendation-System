"""
app.py
Flask application factory and system initialisation.

Usage:
    python app.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from flask import Flask, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

from backend.utils.data_preprocessor import DataPreprocessor
from backend.models.ml_models import (
    SolarSuitabilityMLModel,
    TransformerClusterer,
    LoadForecastModel,
)
from backend.api.routes import api_bp

# ── CSV path — override via env var for deployment ───────────────────────────
CSV_PATH = os.environ.get(
    'TRANSFORMER_CSV',
    r'MASTER_DATASET_ALL_10TRANSFORMERS.csv'
)


def create_app() -> Flask:
    app = Flask(__name__, static_folder='frontend', static_url_path='')
    CORS(app)

    # ── Static file serving (serve index.html at root) ────────────────────
    @app.route('/')
    def index():
        return send_from_directory('frontend', 'index.html')

    # ── Register API blueprint ─────────────────────────────────────────────
    app.register_blueprint(api_bp)

    # ── Initialise ML system and store in app config ───────────────────────
    app.config['APP_STATE'] = _initialise_system(CSV_PATH)

    return app


def _initialise_system(csv_path: str) -> dict:
    """Load data, train models, return shared state dict."""
    print("🚀  Initialising Solar Transformer Intelligence System …")

    # 1. Load + engineer features
    print(f"📊  Loading data from: {csv_path}")
    transformer_data = DataPreprocessor.load_and_prepare_data(csv_path)

    # 2. Build ML feature matrix
    feat_cols = SolarSuitabilityMLModel.FEATURE_COLUMNS
    X_raw = transformer_data[feat_cols].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Train suitability classifier
    ml_model = SolarSuitabilityMLModel()
    # Ensure both classes exist — add tiny noise to guarantee mixed labels
    util = transformer_data['utilization_rate']
    y = (util < util.median()).astype(int)  # split at median, always gives both classes
    ml_model.train(pd.DataFrame(X_scaled, columns=feat_cols), y)

    # 4. Train clusterer
    clusterer = TransformerClusterer(n_clusters=3)
    clusterer.fit_predict(X_scaled)

    # 5. Train forecaster on time series proxy (transformer index as time)
    forecaster = LoadForecastModel()
    forecaster.train(
        np.arange(len(transformer_data)),
        transformer_data['current_load_kW'].values,
    )

    print(f"✓  System ready  |  {len(transformer_data)} transformers loaded")
    print(f"   Feature importance: {list(ml_model.feature_importance().keys())[:3]} …")

    return {
        'transformer_data': transformer_data,
        'ml_model'        : ml_model,
        'clusterer'       : clusterer,
        'forecaster'      : forecaster,
        'scaler'          : scaler,
    }


if __name__ == '__main__':
    flask_app = create_app()
    print("\n📡  Server:  http://localhost:5000")
    print("   Endpoints: POST /api/assess  |  GET /api/stats  |  GET /api/health\n")
    flask_app.run(debug=True, host='0.0.0.0', port=5000)