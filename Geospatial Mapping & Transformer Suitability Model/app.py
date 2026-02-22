from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

TRANSFORMER_CSV_PATH = r"C:\Users\dewmi\OneDrive\Documents\IIT\2nd Year\DSGP\MASTER_DATASET_ALL_10TRANSFORMERS.csv"

# Global variables
transformer_mapper = None
ml_models = {}
scaler = StandardScaler()


# 1. DATA LOADING & PREPROCESSING

class DataPreprocessor:
    """Handle data loading and feature engineering"""

    @staticmethod
    def load_and_prepare_data(csv_path):
        """Load CSV and create features"""
        df = pd.read_csv(csv_path)

        # Aggregate by transformer
        transformer_stats = df.groupby('TRANSFORMER_CODE').agg({
            'TRANSFORMER_LAT': 'first',
            'TRANSFORMER_LON': 'first',
            'NET_CONSUMPTION_kWh': ['mean', 'std', 'max'],
            'HAS_SOLAR': 'sum',
            'INV_CAPACITY': 'sum',
            'IMPORT_kWh': 'mean',
            'EXPORT_kWh': 'mean',
            'PHASE': 'first'
        }).reset_index()

        transformer_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                     for col in transformer_stats.columns.values]

        # Rename for clarity
        transformer_stats.rename(columns={
            'NET_CONSUMPTION_kWh_mean': 'avg_consumption',
            'NET_CONSUMPTION_kWh_std': 'consumption_std',
            'NET_CONSUMPTION_kWh_max': 'max_consumption',
            'HAS_SOLAR_sum': 'solar_connections',
            'INV_CAPACITY_sum': 'total_solar_capacity'
        }, inplace=True)

        # Feature engineering
        transformer_stats['ESTIMATED_CAPACITY_kW'] = 40  # Standard transformer
        transformer_stats['current_load_kW'] = transformer_stats['avg_consumption'] * 0.85
        transformer_stats['utilization_rate'] = (transformer_stats['current_load_kW'] /
                                                 transformer_stats['ESTIMATED_CAPACITY_kW'])
        transformer_stats['available_headroom'] = (transformer_stats['ESTIMATED_CAPACITY_kW'] -
                                                   transformer_stats['current_load_kW'] -
                                                   transformer_stats['total_solar_capacity'])
        transformer_stats['solar_penetration'] = (transformer_stats['total_solar_capacity'] /
                                                  transformer_stats['ESTIMATED_CAPACITY_kW'])
        transformer_stats['demand_volatility'] = transformer_stats['consumption_std'].fillna(0)

        return transformer_stats


# 2. MACHINE LEARNING MODELS

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
            return np.random.rand(len(X))  # Mock if not trained
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
            # Mock forecast: 3% annual growth
            return current_load * (1.03 ** (months_ahead / 12))

        future_periods = np.array([[i] for i in range(1, months_ahead + 1)])
        predictions = self.model.predict(future_periods)
        return predictions[-1] if len(predictions) > 0 else current_load


# 3. INITIALIZE SYSTEM

def initialize_system():
    """Initialize all models and data"""
    global transformer_mapper, ml_models, scaler

    print("🚀 Initializing Comprehensive Solar Assessment System...")

    # Load data
    print("📊 Loading transformer data...")
    df = DataPreprocessor.load_and_prepare_data(TRANSFORMER_CSV_PATH)
    transformer_mapper = df

    # Initialize ML models
    print("🤖 Initializing ML models...")
    ml_models['suitability'] = SolarSuitabilityMLModel()
    ml_models['clusterer'] = TransformerClusterer(n_clusters=3)
    ml_models['forecaster'] = LoadForecastModel()

    # Prepare features for training (mock training data)
    feature_cols = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
                    'solar_penetration', 'demand_volatility', 'available_headroom']
    X = df[feature_cols].fillna(0)

    # Mock labels: high utilization = higher risk (0 = unsuitable, 1 = suitable)
    y = (df['utilization_rate'] < 0.75).astype(int)

    # Scale and train
    X_scaled = scaler.fit_transform(X)
    ml_models['suitability'].train(pd.DataFrame(X_scaled, columns=feature_cols), y)

    # Train clustering
    ml_models['clusterer'].fit_predict(X_scaled)

    # Train forecaster
    ml_models['forecaster'].train(np.arange(len(df)), df['current_load_kW'].values)

    print("✓ System initialized successfully!")
    print(f"✓ Loaded {len(df)} transformers")
    print("✓ ML Models trained")

    return df


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/assess', methods=['POST'])
def assess_transformer():
    """Comprehensive transformer assessment"""
    try:
        data = request.json
        location = data.get('location', '')
        solar_capacity = float(data.get('solarCapacity', 5.0))
        search_radius = float(data.get('searchRadius', 500))

        from transformer_assessment import CoordinateExtractor, TransformerSuitability

        # Extract coordinates
        extractor = CoordinateExtractor()
        user_coords = extractor.get_coordinates(location)
        user_address = extractor.reverse_geocode(user_coords[0], user_coords[1])

        # Find nearby transformers
        user_coords_tuple = (user_coords[0], user_coords[1])
        transformer_mapper['DISTANCE_M'] = transformer_mapper.apply(
            lambda row: geodesic(user_coords_tuple,
                                 (row['TRANSFORMER_LAT'], row['TRANSFORMER_LON'])).meters,
            axis=1
        )

        nearby = transformer_mapper[transformer_mapper['DISTANCE_M'] <= search_radius].copy()

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
            'userLocation': user_address,
            'userCoords': list(user_coords),
            'solarForecast': solar_capacity,
            'searchRadius': search_radius,
            'transformersFound': len(transformer_results),
            'transformers': transformer_results,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': f'Assessment failed: {str(e)}'}), 500


@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get ML model feature importance for explainability"""
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
        X = transformer_mapper[feature_cols].fillna(0)
        X_scaled = scaler.transform(X)

        labels = ml_models['clusterer'].fit_predict(X_scaled)
        cluster_names = ml_models['clusterer'].get_cluster_names()

        clusters = {}
        for cluster_id in range(3):
            mask = labels == cluster_id
            cluster_transformers = transformer_mapper[mask]['TRANSFORMER_CODE'].tolist()
            clusters[cluster_names[cluster_id]] = {
                'count': int(mask.sum()),
                'transformers': cluster_transformers,
                'avgUtilization': float(transformer_mapper[mask]['utilization_rate'].mean() * 100),
                'avgSolarPenetration': float(transformer_mapper[mask]['solar_penetration'].mean() * 100)
            }

        return jsonify(clusters), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<transformer_code>', methods=['GET'])
def get_load_forecast(transformer_code):
    """Get future load forecast for a transformer"""
    try:
        tf = transformer_mapper[transformer_mapper['TRANSFORMER_CODE'] == transformer_code]

        if len(tf) == 0:
            return jsonify({'error': 'Transformer not found'}), 404

        current_load = tf.iloc[0]['current_load_kW']

        # Generate 12-month forecast
        forecasts = []
        for month in range(1, 13):
            future_load = ml_models['forecaster'].predict_future_load(current_load, month)
            forecasts.append({
                'month': month,
                'load_kW': float(future_load),
                'growth_percent': float((future_load / current_load - 1) * 100)
            })

        return jsonify({
            'transformerCode': transformer_code,
            'currentLoad': float(current_load),
            'forecast12m': forecasts
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get network statistics"""
    try:
        return jsonify({
            'totalTransformers': len(transformer_mapper),
            'avgCapacity': float(transformer_mapper['ESTIMATED_CAPACITY_kW'].mean()),
            'totalCapacity': float(transformer_mapper['ESTIMATED_CAPACITY_kW'].sum()),
            'transformersWithSolar': int(transformer_mapper['solar_connections'].sum()),
            'totalInstalledSolar': float(transformer_mapper['total_solar_capacity'].sum()),
            'avgUtilization': float(transformer_mapper['utilization_rate'].mean() * 100),
            'avgSolarPenetration': float(transformer_mapper['solar_penetration'].mean() * 100),
            'highRiskTransformers': int((transformer_mapper['utilization_rate'] > 0.85).sum())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'transformersLoaded': len(transformer_mapper) if transformer_mapper is not None else 0,
        'mlModelsReady': all(m is not None for m in ml_models.values())
    }), 200


def get_recommendation(score):
    """Generate recommendation based on score"""
    if score >= 80:
        return "HIGHLY SUITABLE - Proceed with connection"
    elif score >= 50:
        return "CONDITIONALLY SUITABLE - Review with utility"
    else:
        return "NOT SUITABLE - Consider alternative transformers"


# MAIN

if __name__ == '__main__':
    initialize_system()

    print("\n📡 Starting Flask API Server...")
    print("✓ API running at http://localhost:5000")
    print("\nAvailable Endpoints:")
    print("  POST /api/assess - Comprehensive assessment")
    print("  GET  /api/feature-importance - ML feature importance (XAI)")
    print("  GET  /api/clustering - Transformer clustering analysis")
    print("  GET  /api/forecast/<code> - Load forecast (12 months)")
    print("  GET  /api/stats - Network statistics")
    print("  GET  /api/health - Health check")

    app.run(debug=True, host='0.0.0.0', port=5000)