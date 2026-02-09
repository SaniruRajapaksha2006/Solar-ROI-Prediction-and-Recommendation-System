
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings

warnings.filterwarnings('ignore')

# Import your existing transformer assessment classes
from transformer_assessment import (
    CoordinateExtractor,
    TransformerMapper,
    TransformerSuitability
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize with your CSV path
TRANSFORMER_CSV_PATH = r"C:\Users\dewmi\OneDrive\Documents\IIT\2nd Year\DSGP\MASTER_DATASET_ALL_10TRANSFORMERS.csv"

# Global variable to store mapper (load once)
transformer_mapper = None


def init_mapper():
    """Initialize transformer mapper on startup"""
    global transformer_mapper
    try:
        transformer_mapper = TransformerMapper(TRANSFORMER_CSV_PATH)
        print("✓ Transformer data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading transformer data: {str(e)}")
        raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/assess', methods=['POST'])
def assess_transformer():
    """
    Main assessment endpoint
    Expected JSON:
    {
        "location": "6.849,79.925" or "address",
        "solarCapacity": 5.0,
        "searchRadius": 500
    }
    """
    try:
        data = request.json
        location = data.get('location', '')
        solar_capacity = float(data.get('solarCapacity', 5.0))
        search_radius = float(data.get('searchRadius', 500))

        # Validate inputs
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        if solar_capacity <= 0 or solar_capacity > 100:
            return jsonify({'error': 'Solar capacity must be between 0 and 100 kW'}), 400

        # Step 1: Extract coordinates
        extractor = CoordinateExtractor()
        user_coords = extractor.get_coordinates(location)
        user_address = extractor.reverse_geocode(user_coords[0], user_coords[1])

        # Step 2: Find nearby transformers
        nearby = transformer_mapper.find_nearby_transformers(
            user_coords[0],
            user_coords[1],
            search_radius
        )

        if len(nearby) == 0:
            return jsonify({
                'error': f'No transformers found within {search_radius}m',
                'userLocation': user_address,
                'userCoords': user_coords,
                'solarForecast': solar_capacity,
                'transformers': []
            }), 200

        # Step 3: Assess suitability
        suitability_assessor = TransformerSuitability(solar_capacity)
        transformer_results = []

        for idx, (_, tf_row) in enumerate(nearby.iterrows()):
            score_info = suitability_assessor.score_suitability(
                tf_row,
                tf_row['DISTANCE_M']
            )
            headroom = score_info['headroom_analysis']

            transformer_results.append({
                'rank': None,  # Will set after sorting
                'code': tf_row['TRANSFORMER_CODE'],
                'distance': tf_row['DISTANCE_M'],
                'score': score_info['overall_score'],
                'capacity': headroom['transformer_capacity_kW'],
                'currentLoad': headroom['current_load_kW'],
                'existingSolar': headroom['existing_solar_kW'],
                'newSolar': headroom['new_solar_request_kW'],
                'utilizationBefore': headroom['utilization_before'] * 100,
                'utilizationAfter': headroom['utilization_after'] * 100,
                'canSupport': bool(headroom['can_support']),
                'curtailmentRisk': bool(headroom['curtailment_risk']),
                'headroomScore': score_info['headroom_score'],
                'distanceScore': score_info['distance_score'],
                'stabilityScore': score_info['stability_score'],
                'recommendation': get_recommendation(score_info['overall_score'])
            })

        # Sort by score descending and assign ranks
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
            'timestamp': pd.Timestamp.now().isoformat()
        }), 200

    except ValueError as e:
        return jsonify({'error': f'Input error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Assessment failed: {str(e)}'}), 500


@app.route('/api/transformer/<code>', methods=['GET'])
def get_transformer_details(code):
    """Get detailed information about a specific transformer"""
    try:
        tf_data = transformer_mapper.transformers[
            transformer_mapper.transformers['TRANSFORMER_CODE'] == code
            ]

        if len(tf_data) == 0:
            return jsonify({'error': 'Transformer not found'}), 404

        tf = tf_data.iloc[0]

        return jsonify({
            'code': tf['TRANSFORMER_CODE'],
            'latitude': tf['TRANSFORMER_LAT'],
            'longitude': tf['TRANSFORMER_LON'],
            'estimatedCapacity': tf['ESTIMATED_CAPACITY_kW'],
            'avgNetConsumption': tf['NET_CONSUMPTION_kWh'],
            'solarsConnected': tf['HAS_SOLAR'],
            'totalInstalledCapacity': tf['INV_CAPACITY'],
            'avgImport': tf['IMPORT_kWh'],
            'avgExport': tf['EXPORT_kWh']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API health"""
    return jsonify({
        'status': 'healthy',
        'transformersLoaded': len(transformer_mapper.transformers) if transformer_mapper else 0
    }), 200


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall statistics about the transformer network"""
    try:
        if transformer_mapper is None:
            return jsonify({'error': 'Data not loaded'}), 500

        df = transformer_mapper.transformers

        return jsonify({
            'totalTransformers': len(df),
            'avgCapacity': float(df['ESTIMATED_CAPACITY_kW'].mean()),
            'totalCapacity': float(df['ESTIMATED_CAPACITY_kW'].sum()),
            'transformersWithSolar': int(df['HAS_SOLAR'].sum()),
            'totalInstalledSolar': float(df['INV_CAPACITY'].sum()),
            'avgConsumption': float(df['NET_CONSUMPTION_kWh'].mean()),
            'totalExport': float(df['EXPORT_kWh'].sum())
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_recommendation(score):
    """Generate recommendation based on score"""
    if score >= 80:
        return "HIGHLY SUITABLE - Proceed with connection"
    elif score >= 50:
        return "CONDITIONALLY SUITABLE - Review with utility"
    else:
        return "NOT SUITABLE - Consider alternative transformers"


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("🚀 Initializing Flask API...")
    print(f"📁 CSV Path: {TRANSFORMER_CSV_PATH}")

    # Initialize mapper
    init_mapper()

    print("\n📡 Starting server...")
    print("✓ API running at http://localhost:5000")
    print("✓ CORS enabled for frontend")
    print("\nEndpoints:")
    print("  POST /api/assess - Run transformer assessment")
    print("  GET  /api/transformer/<code> - Get transformer details")
    print("  GET  /api/stats - Get network statistics")
    print("  GET  /api/health - Check API health")

    app.run(debug=True, host='0.0.0.0', port=5000)