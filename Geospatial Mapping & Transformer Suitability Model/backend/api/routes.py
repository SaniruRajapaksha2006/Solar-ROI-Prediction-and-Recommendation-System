"""
routes.py
Flask blueprint defining all REST API endpoints.

Endpoints:
  POST /api/assess              – Full geospatial + ML + rule-based assessment
  GET  /api/feature-importance  – XAI: model feature weights
  GET  /api/clustering          – Transformer risk cluster summary
  GET  /api/stats               – Network-wide statistics
  GET  /api/health              – Liveness probe
"""

from __future__ import annotations
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, current_app

from backend.utils.geo_utils import filter_nearby_transformers
from backend.models.suitability_engine import (
    TransformerSuitabilityEngine,
    generate_recommendation,
)
from backend.models.ml_models import CLUSTER_PROFILE_NAMES

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ─── Helper: pull shared state from app context ───────────────────────────────

def _ctx():
    return current_app.config['APP_STATE']


# ─── /assess ──────────────────────────────────────────────────────────────────

@api_bp.route('/assess', methods=['POST'])
def assess():
    """
    Body JSON:
      latitude, longitude  – user's location (decimal degrees)
      solarCapacity        – proposed solar installation size (kW)
      searchRadius         – max distance to search (metres, default 500)
    """
    try:
        body          = request.get_json(force=True)
        user_lat      = float(body['latitude'])
        user_lon      = float(body['longitude'])
        solar_kw      = float(body.get('solarCapacity', 5.0))
        search_radius = float(body.get('searchRadius', 500))

        state = _ctx()
        transformer_data: pd.DataFrame = state['transformer_data']
        ml_model                        = state['ml_model']
        clusterer                       = state['clusterer']
        forecaster                      = state['forecaster']
        scaler                          = state['scaler']

        # 1. Geospatial filter
        nearby = filter_nearby_transformers(
            transformer_data, user_lat, user_lon, search_radius
        )

        if nearby.empty:
            return jsonify({
                'error': f'No transformers found within {search_radius:.0f} m. '
                         'Try increasing the search radius.'
            }), 404

        # 2. ML feature matrix
        feat_cols = ml_model.FEATURE_COLUMNS
        X_raw    = nearby[feat_cols].fillna(0)
        X_scaled = scaler.transform(X_raw)

        ml_probs      = ml_model.predict_proba(pd.DataFrame(X_scaled, columns=feat_cols))
        cluster_ids   = clusterer.predict(X_scaled)
        cluster_names = clusterer.cluster_names()

        # 3. Rule-based scoring + blend
        engine  = TransformerSuitabilityEngine(solar_kw)
        results = []

        for i, (_, row) in enumerate(nearby.iterrows()):
            rule_result   = engine.evaluate(row, float(row['DISTANCE_M']))
            headroom      = rule_result.headroom_analysis

            rule_score    = rule_result.overall_score
            ml_score      = float(ml_probs[i]) * 100
            blended_score = rule_score * 0.55 + ml_score * 0.45  # rule-based dominates

            future_load   = forecaster.predict_future_load(
                headroom.current_load_kW, months_ahead=12
            )

            recommendation = generate_recommendation(blended_score, headroom)

            results.append({
                'rank'            : None,  # assigned after sort
                'code'            : str(row['TRANSFORMER_CODE']),
                'latitude'        : float(row['TRANSFORMER_LAT']),
                'longitude'       : float(row['TRANSFORMER_LON']),
                'distance'        : float(row['DISTANCE_M']),
                # Scores
                'score'           : round(blended_score, 2),
                'ruleBasedScore'  : round(rule_score, 2),
                'mlScore'         : round(ml_score, 2),
                'headroomScore'   : rule_result.headroom_score,
                'distanceScore'   : rule_result.distance_score,
                'stabilityScore'  : rule_result.stability_score,
                'suitabilityLabel': rule_result.suitability_label,
                # Capacity
                'capacity'        : headroom.transformer_capacity_kW,
                'currentLoad'     : round(headroom.current_load_kW, 2),
                'existingSolar'   : round(headroom.existing_solar_kW, 2),
                'newSolar'        : headroom.new_solar_request_kW,
                'availableHeadroom': round(headroom.available_headroom_kW, 2),
                'safeHeadroom'    : round(headroom.safe_headroom_kW, 2),
                'headroomMarginPct': round(headroom.headroom_margin_pct, 1),
                # Utilisation
                'utilizationBefore': round(headroom.utilization_before * 100, 1),
                'utilizationAfter' : round(headroom.utilization_after * 100, 1),
                # Risk flags
                'canSupport'      : bool(headroom.can_support),
                'curtailmentRisk' : bool(headroom.curtailment_risk),
                # Cluster
                'cluster'         : cluster_names.get(int(cluster_ids[i]),
                                                       'Unknown Cluster'),
                'clusterId'       : int(cluster_ids[i]),
                # Forecast
                'futureLoad12m'   : round(future_load, 2),
                # Text
                'recommendation'  : recommendation,
            })

        # 4. Rank by blended score
        results.sort(key=lambda x: x['score'], reverse=True)
        for idx, r in enumerate(results):
            r['rank'] = idx + 1

        return jsonify({
            'userLocation'     : {'latitude': user_lat, 'longitude': user_lon},
            'solarForecast'    : solar_kw,
            'searchRadius'     : search_radius,
            'transformersFound': len(results),
            'transformers'     : results,
            'timestamp'        : datetime.utcnow().isoformat() + 'Z',
        }), 200

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Assessment failed: {e}'}), 500


# ─── /feature-importance ──────────────────────────────────────────────────────

@api_bp.route('/feature-importance', methods=['GET'])
def feature_importance():
    try:
        importance = _ctx()['ml_model'].feature_importance()
        return jsonify({
            'features'   : importance,
            'explanation': (
                'Feature importance (Random Forest) indicates which transformer '
                'attributes most influence suitability predictions.'
            ),
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── /clustering ──────────────────────────────────────────────────────────────

@api_bp.route('/clustering', methods=['GET'])
def clustering():
    try:
        state            = _ctx()
        transformer_data = state['transformer_data']
        ml_model         = state['ml_model']
        clusterer        = state['clusterer']
        scaler           = state['scaler']

        feat_cols = ml_model.FEATURE_COLUMNS
        X_scaled  = scaler.transform(transformer_data[feat_cols].fillna(0))
        labels    = clusterer.predict(X_scaled)
        names     = clusterer.cluster_names()

        clusters = {}
        for cid in range(3):
            mask = labels == cid
            subset = transformer_data[mask]
            clusters[names[cid]] = {
                'count'              : int(mask.sum()),
                'transformers'       : subset['TRANSFORMER_CODE'].tolist(),
                'avgUtilizationPct'  : round(float(subset['utilization_rate'].mean() * 100), 1),
                'avgSolarPenetrationPct': round(float(subset['solar_penetration'].mean() * 100), 1),
                'avgHeadroom_kW'     : round(float(subset['available_headroom'].mean()), 2),
            }

        return jsonify(clusters), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── /stats ───────────────────────────────────────────────────────────────────

@api_bp.route('/stats', methods=['GET'])
def stats():
    try:
        df = _ctx()['transformer_data']
        return jsonify({
            'totalTransformers'       : len(df),
            'avgCapacity_kW'          : round(float(df['ESTIMATED_CAPACITY_kW'].mean()), 1),
            'totalCapacity_kW'        : round(float(df['ESTIMATED_CAPACITY_kW'].sum()), 1),
            'transformersWithSolar'   : int((df['solar_connections'] > 0).sum()),
            'totalInstalledSolar_kW'  : round(float(df['total_solar_capacity'].sum()), 1),
            'avgUtilizationPct'       : round(float(df['utilization_rate'].mean() * 100), 1),
            'avgSolarPenetrationPct'  : round(float(df['solar_penetration'].mean() * 100), 1),
            'highRiskCount'           : int((df['utilization_rate'] > 0.85).sum()),
            'avgHeadroom_kW'          : round(float(df['available_headroom'].mean()), 2),
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── /health ──────────────────────────────────────────────────────────────────

@api_bp.route('/health', methods=['GET'])
def health():
    state = _ctx()
    return jsonify({
        'status'          : 'healthy',
        'transformers'    : len(state.get('transformer_data', [])),
        'mlModelReady'    : state['ml_model'].is_trained,
        'clustererReady'  : True,
        'forecasterReady' : state['forecaster'].is_trained,
        'timestamp'       : datetime.utcnow().isoformat() + 'Z',
    }), 200