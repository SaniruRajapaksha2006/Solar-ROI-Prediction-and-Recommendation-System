# ============================================================================
# FILE: comprehensive_app.py - Full Featured Backend API with Map Support
# ============================================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_home():
    return send_file('index.html', mimetype='text/html')

TRANSFORMER_CSV_PATH = r"C:\Users\dewmi\OneDrive\Documents\IIT\2nd Year\DSGP\MASTER_DATASET_ALL_10TRANSFORMERS.csv"

# Global variables
transformer_data = None
ml_models = {}
scaler = StandardScaler()


# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handle data loading and feature engineering"""

    @staticmethod
    def load_and_prepare_data(csv_path):
        """Load CSV and create features - FLEXIBLE FOR DIFFERENT COLUMN NAMES"""
        df = pd.read_csv(csv_path)

        print(f"📊 Loaded CSV with columns: {df.columns.tolist()}")

        # Aggregate by transformer
        agg_dict = {
            'TRANSFORMER_LAT': 'first',
            'TRANSFORMER_LON': 'first',
            'NET_CONSUMPTION_kWh': ['mean', 'std', 'max'],
            'HAS_SOLAR': 'sum',
            'INV_CAPACITY': 'sum',
            'IMPORT_kWh': 'mean',
            'EXPORT_kWh': 'mean'
        }

        # Build aggregation dict based on available columns
        available_agg = {}
        for key, value in agg_dict.items():
            if key in df.columns:
                available_agg[key] = value

        transformer_stats = df.groupby('TRANSFORMER_CODE').agg(available_agg).reset_index()

        # Flatten column names
        transformer_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                     for col in transformer_stats.columns.values]

        # Standardize column names
        rename_map = {
            'TRANSFORMER_LAT_first': 'TRANSFORMER_LAT',  # ADD THIS
            'TRANSFORMER_LON_first': 'TRANSFORMER_LON',  # ADD THIS
            'TRANSFORMER_CODE_': 'TRANSFORMER_CODE',  # ADD THIS (safety)
            'NET_CONSUMPTION_kWh_mean': 'avg_consumption',
            'NET_CONSUMPTION_kWh_std': 'consumption_std',
            'NET_CONSUMPTION_kWh_max': 'max_consumption',
            'HAS_SOLAR_sum': 'solar_connections',
            'INV_CAPACITY_sum': 'total_solar_capacity',
            'IMPORT_kWh_mean': 'avg_import',
            'EXPORT_kWh_mean': 'avg_export'
        }

        transformer_stats.rename(columns=rename_map, inplace=True)

        # Feature engineering
        transformer_stats['ESTIMATED_CAPACITY_kW'] = 40
        transformer_stats['current_load_kW'] = transformer_stats['avg_consumption'].fillna(0) * 0.85
        transformer_stats['utilization_rate'] = (transformer_stats['current_load_kW'] /
                                                 transformer_stats['ESTIMATED_CAPACITY_kW'])
        transformer_stats['available_headroom'] = (transformer_stats['ESTIMATED_CAPACITY_kW'] -
                                                   transformer_stats['current_load_kW'] -
                                                   transformer_stats['total_solar_capacity'].fillna(0))
        transformer_stats['solar_penetration'] = (transformer_stats['total_solar_capacity'].fillna(0) /
                                                  transformer_stats['ESTIMATED_CAPACITY_kW'])
        transformer_stats['demand_volatility'] = transformer_stats['consumption_std'].fillna(0)

        # Fill NaN values
        transformer_stats = transformer_stats.fillna(0)

        return transformer_stats


# ============================================================================
# 2. MACHINE LEARNING MODELS
# ============================================================================

class SolarSuitabilityMLModel:
    """AI-powered suitability prediction"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = None
        self.is_trained = False

    def train(self, X, y):
        """Train the model"""
        self.feature_columns = X.columns.tolist()
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        """Predict suitability"""
        if not self.is_trained:
            return np.random.rand(len(X))
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """Return feature importance for explainability"""
        if not self.is_trained:
            return {}

        importance_dict = {}
        for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
            importance_dict[feature] = float(importance)

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class TransformerClusterer:
    """Unsupervised clustering of transformers"""

    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = None

    def fit_predict(self, X):
        """Cluster transformers"""
        self.labels = self.kmeans.fit_predict(X)
        return self.labels

    def get_cluster_names(self):
        """Generate cluster interpretations"""
        return {
            0: 'Underutilized - Low Risk',
            1: 'Balanced Load - Medium Risk',
            2: 'High Utilization - High Risk'
        }


class LoadForecastModel:
    """Predict future transformer load"""

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, X_time, y_load):
        """Train forecasting model"""
        X_numeric = np.arange(len(X_time)).reshape(-1, 1)
        self.model.fit(X_numeric, y_load)
        self.is_trained = True

    def predict_future_load(self, current_load, months_ahead=12):
        """Predict load X months ahead"""
        if not self.is_trained:
            return current_load * (1.03 ** (months_ahead / 12))

        future_periods = np.array([[i] for i in range(1, months_ahead + 1)])
        predictions = self.model.predict(future_periods)
        return predictions[-1] if len(predictions) > 0 else current_load


# ============================================================================
# 3. TRANSFORMER SUITABILITY ASSESSMENT (RULE-BASED)
# ============================================================================

class TransformerSuitability:
    """Rule-based suitability scoring"""

    SAFETY_MARGIN = 0.8
    CURTAILMENT_THRESHOLD = 0.75

    def __init__(self, solar_forecast_kW):
        self.solar_forecast_kW = solar_forecast_kW

    def calculate_headroom(self, transformer_row):
        """Calculate available capacity"""
        capacity = transformer_row['ESTIMATED_CAPACITY_kW']
        current_load = transformer_row['current_load_kW']
        existing_solar = transformer_row['total_solar_capacity']

        available_capacity = capacity - (current_load + existing_solar)
        safe_headroom = capacity * self.SAFETY_MARGIN - (current_load + existing_solar)

        can_support = available_capacity >= self.solar_forecast_kW
        curtailment_risk = ((current_load + existing_solar + self.solar_forecast_kW) /
                            capacity > self.CURTAILMENT_THRESHOLD)

        return {
            'transformer_capacity_kW': capacity,
            'current_load_kW': current_load,
            'existing_solar_kW': existing_solar,
            'total_before_new_kW': current_load + existing_solar,
            'available_headroom_kW': available_capacity,
            'safe_headroom_kW': safe_headroom,
            'new_solar_request_kW': self.solar_forecast_kW,
            'total_after_new_kW': current_load + existing_solar + self.solar_forecast_kW,
            'can_support': can_support,
            'curtailment_risk': curtailment_risk,
            'utilization_before': (current_load + existing_solar) / capacity,
            'utilization_after': (current_load + existing_solar + self.solar_forecast_kW) / capacity
        }

    def score_suitability(self, transformer_row, distance_m):
        """Calculate overall suitability score"""
        headroom = self.calculate_headroom(transformer_row)

        # Headroom score (40% weight)
        if headroom['safe_headroom_kW'] >= self.solar_forecast_kW * 1.5:
            headroom_score = 100
        elif headroom['safe_headroom_kW'] >= self.solar_forecast_kW:
            headroom_score = 80
        elif headroom['available_headroom_kW'] >= self.solar_forecast_kW:
            headroom_score = 50
        else:
            headroom_score = 0

        # Distance score (30% weight)
        distance_score = max(0, 100 - (distance_m / 10))

        # Grid stability score (30% weight)
        if headroom['utilization_after'] <= 0.7:
            stability_score = 100
        elif headroom['utilization_after'] <= 0.85:
            stability_score = 75
        elif headroom['utilization_after'] <= 0.95:
            stability_score = 40
        else:
            stability_score = 0

        overall_score = (headroom_score * 0.40 + distance_score * 0.30 + stability_score * 0.30)

        return {
            'overall_score': overall_score,
            'headroom_score': headroom_score,
            'distance_score': distance_score,
            'stability_score': stability_score,
            'headroom_analysis': headroom
        }


# ============================================================================
# 4. INITIALIZE SYSTEM
# ============================================================================

def initialize_system():
    """Initialize all models and data"""
    global transformer_data, ml_models, scaler

    print("🚀 Initializing Comprehensive Solar Assessment System...")

    try:
        # Load data
        print("📊 Loading transformer data...")
        transformer_data = DataPreprocessor.load_and_prepare_data(TRANSFORMER_CSV_PATH)
        print(f"✓ Loaded {len(transformer_data)} transformers")

        # Initialize ML models
        print("🤖 Initializing ML models...")
        ml_models['suitability'] = SolarSuitabilityMLModel()
        ml_models['clusterer'] = TransformerClusterer(n_clusters=3)
        ml_models['forecaster'] = LoadForecastModel()

        # Prepare features for training
        feature_cols = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
                        'solar_penetration', 'demand_volatility', 'available_headroom']
        X = transformer_data[feature_cols].fillna(0)

        # Mock labels: high utilization = higher risk
        y = (transformer_data['utilization_rate'] < 0.75).astype(int)

        # Scale and train
        X_scaled = scaler.fit_transform(X)
        ml_models['suitability'].train(pd.DataFrame(X_scaled, columns=feature_cols), y)

        # Train clustering
        ml_models['clusterer'].fit_predict(X_scaled)

        # Train forecaster
        ml_models['forecaster'].train(np.arange(len(transformer_data)),
                                      transformer_data['current_load_kW'].values)

        print("✓ ML Models trained successfully")

    except Exception as e:
        print(f"✗ Initialization error: {str(e)}")
        raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/assess', methods=['POST'])
def assess_transformer():
    """Comprehensive transformer assessment"""
    try:
        data = request.json
        user_lat = float(data.get('latitude', 0))
        user_lon = float(data.get('longitude', 0))
        solar_capacity = float(data.get('solarCapacity', 5.0))
        search_radius = float(data.get('searchRadius', 500))

        user_coords_tuple = (user_lat, user_lon)

        # Find nearby transformers
        transformer_data['DISTANCE_M'] = transformer_data.apply(
            lambda row: geodesic(user_coords_tuple,
                                 (row['TRANSFORMER_LAT'], row['TRANSFORMER_LON'])).meters,
            axis=1
        )

        nearby = transformer_data[transformer_data['DISTANCE_M'] <= search_radius].copy()

        if len(nearby) == 0:
            return jsonify({'error': f'No transformers found within {search_radius}m'}), 200

        # Calculate suitability scores
        suitability_assessor = TransformerSuitability(solar_capacity)
        transformer_results = []

        # Get ML predictions
        feature_cols = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
                        'solar_penetration', 'demand_volatility', 'available_headroom']
        X_ml = nearby[feature_cols].fillna(0)
        X_scaled = scaler.transform(X_ml)
        ml_scores = ml_models['suitability'].predict(pd.DataFrame(X_scaled, columns=feature_cols))

        # Get cluster assignments
        cluster_labels = ml_models['clusterer'].fit_predict(X_scaled)
        cluster_names = ml_models['clusterer'].get_cluster_names()

        for idx, (_, tf_row) in enumerate(nearby.iterrows()):
            score_info = suitability_assessor.score_suitability(tf_row, tf_row['DISTANCE_M'])
            headroom = score_info['headroom_analysis']

            # Blend rule-based and ML scores
            rule_score = score_info['overall_score']
            ml_score = ml_scores[idx] * 100
            blended_score = (rule_score * 0.5 + ml_score * 0.5)

            # Forecast future load
            future_load = ml_models['forecaster'].predict_future_load(
                headroom['current_load_kW'],
                months_ahead=12
            )

            transformer_results.append({
                'rank': None,
                'code': tf_row['TRANSFORMER_CODE'],
                'latitude': float(tf_row['TRANSFORMER_LAT']),
                'longitude': float(tf_row['TRANSFORMER_LON']),
                'distance': float(tf_row['DISTANCE_M']),
                'score': float(blended_score),
                'ruleBasedScore': float(rule_score),
                'mlScore': float(ml_score),
                'capacity': float(headroom['transformer_capacity_kW']),
                'currentLoad': float(headroom['current_load_kW']),
                'existingSolar': float(headroom['existing_solar_kW']),
                'newSolar': float(headroom['new_solar_request_kW']),
                'utilizationBefore': float(headroom['utilization_before'] * 100),
                'utilizationAfter': float(headroom['utilization_after'] * 100),
                'canSupport': bool(headroom['can_support']),
                'curtailmentRisk': bool(headroom['curtailment_risk']),
                'cluster': str(cluster_names[cluster_labels[idx]]),
                'futureLoad12m': float(future_load),
                'recommendation': get_recommendation(blended_score)
            })

        # Sort by score
        transformer_results.sort(key=lambda x: x['score'], reverse=True)
        for idx, tf in enumerate(transformer_results):
            tf['rank'] = idx + 1

        return jsonify({
            'userLocation': {'latitude': user_lat, 'longitude': user_lon},
            'solarForecast': solar_capacity,
            'searchRadius': search_radius,
            'transformersFound': len(transformer_results),
            'transformers': transformer_results,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Assessment failed: {str(e)}'}), 500


@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get ML model feature importance"""
    try:
        importance = ml_models['suitability'].get_feature_importance()
        return jsonify({
            'features': importance,
            'explanation': 'Feature importance shows which factors most influence transformer suitability'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clustering', methods=['GET'])
def get_clustering_analysis():
    """Get transformer clustering analysis"""
    try:
        feature_cols = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
                        'solar_penetration', 'demand_volatility', 'available_headroom']
        X = transformer_data[feature_cols].fillna(0)
        X_scaled = scaler.transform(X)

        labels = ml_models['clusterer'].fit_predict(X_scaled)
        cluster_names = ml_models['clusterer'].get_cluster_names()

        clusters = {}
        for cluster_id in range(3):
            mask = labels == cluster_id
            cluster_transformers = transformer_data[mask]['TRANSFORMER_CODE'].tolist()
            clusters[cluster_names[cluster_id]] = {
                'count': int(mask.sum()),
                'transformers': cluster_transformers,
                'avgUtilization': float(transformer_data[mask]['utilization_rate'].mean() * 100),
                'avgSolarPenetration': float(transformer_data[mask]['solar_penetration'].mean() * 100)
            }

        return jsonify(clusters), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get network statistics"""
    try:
        return jsonify({
            'totalTransformers': len(transformer_data),
            'avgCapacity': float(transformer_data['ESTIMATED_CAPACITY_kW'].mean()),
            'totalCapacity': float(transformer_data['ESTIMATED_CAPACITY_kW'].sum()),
            'transformersWithSolar': int(transformer_data['solar_connections'].sum()),
            'totalInstalledSolar': float(transformer_data['total_solar_capacity'].sum()),
            'avgUtilization': float(transformer_data['utilization_rate'].mean() * 100),
            'avgSolarPenetration': float(transformer_data['solar_penetration'].mean() * 100),
            'highRiskTransformers': int((transformer_data['utilization_rate'] > 0.85).sum())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'transformersLoaded': len(transformer_data) if transformer_data is not None else 0,
        'mlModelsReady': all(m is not None for m in ml_models.values())
    }), 200

def get_recommendation(score):
    """Generate recommendation"""
    if score >= 80:
        return "HIGHLY SUITABLE - Proceed with connection"
    elif score >= 50:
        return "CONDITIONALLY SUITABLE - Review with utility"
    else:
        return "NOT SUITABLE - Consider alternative transformers"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    initialize_system()

    print("\n📡 Starting Flask API Server...")
    print("✓ API running at http://localhost:5000")
    print("\nAvailable Endpoints:")
    print("  POST /api/assess - Comprehensive assessment")
    print("  GET  /api/feature-importance - ML feature importance (XAI)")
    print("  GET  /api/clustering - Transformer clustering analysis")
    print("  GET  /api/stats - Network statistics")
    print("  GET  /api/health - Health check")

    app.run(debug=True, host='0.0.0.0', port=5000)