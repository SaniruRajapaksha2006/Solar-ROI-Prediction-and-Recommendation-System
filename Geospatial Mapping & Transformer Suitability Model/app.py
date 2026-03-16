import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic

st.set_page_config(
    page_title="SolarGrid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .metric-card {
        background: #1c2539;
        border: 1px solid #1f2d45;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        font-family: monospace;
        color: #00d4ff;
    }
    .metric-val-green { color: #10b981 !important; }
    .metric-val-amber { color: #f59e0b !important; }
    .metric-val-red   { color: #ef4444 !important; }
    .metric-lbl {
        font-size: 11px;
        color: #64748b;
        font-family: monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .tf-card {
        background: #1c2539;
        border: 1px solid #1f2d45;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .tf-card-selected { border-color: #00d4ff !important; }

    .score-green { color: #10b981; font-weight: 700; font-family: monospace; }
    .score-blue  { color: #00d4ff; font-weight: 700; font-family: monospace; }
    .score-amber { color: #f59e0b; font-weight: 700; font-family: monospace; }
    .score-red   { color: #ef4444; font-weight: 700; font-family: monospace; }

    .rec-box {
        background: rgba(0,212,255,0.05);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 10px;
        padding: 14px;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
    }
    .cap-box {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 12px;
        padding: 16px;
        margin-top: 8px;
    }
    .cap-val {
        font-size: 32px;
        font-weight: 700;
        font-family: monospace;
        color: #10b981;
    }
    .flag-red {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 8px;
        padding: 8px 12px;
        color: #fca5a5;
        font-size: 12px;
        font-family: monospace;
        margin-top: 6px;
    }
    .flag-green {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 8px;
        padding: 8px 12px;
        color: #6ee7b7;
        font-size: 12px;
        font-family: monospace;
        margin-top: 6px;
    }
    .section-label {
        font-family: monospace;
        font-size: 10px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 8px;
        margin-top: 16px;
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2d45;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_CAPACITY_KW  = 100
SAFETY_MARGIN        = 0.80
CURTAILMENT_THRESH   = 0.75
CAPACITY_TIERS       = [1.5, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
FEATURE_COLS         = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
                        'solar_penetration', 'demand_volatility', 'available_headroom',
                        'export_ratio']


@st.cache_data(show_spinner="Loading transformer data…")
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    unique_customers = df.drop_duplicates(subset=['TRANSFORMER_CODE', 'ACCOUNT_NO'])
    solar_agg = unique_customers.groupby('TRANSFORMER_CODE').agg(
        total_solar_capacity=('INV_CAPACITY', 'sum'),
        solar_connections=('HAS_SOLAR', 'sum'),
    ).reset_index()

    consumption_agg = df.groupby('TRANSFORMER_CODE').agg(
        TRANSFORMER_LAT=('TRANSFORMER_LAT', 'first'),
        TRANSFORMER_LON=('TRANSFORMER_LON', 'first'),
        avg_consumption=('NET_CONSUMPTION_kWh', 'mean'),
        consumption_std=('NET_CONSUMPTION_kWh', 'std'),
        avg_import=('IMPORT_kWh', 'mean'),
        avg_export=('EXPORT_kWh', 'mean'),
        num_customers=('ACCOUNT_NO', 'nunique'),
    ).reset_index()

    agg = consumption_agg.merge(solar_agg, on='TRANSFORMER_CODE', how='left')
    agg['total_solar_capacity'] = agg['total_solar_capacity'].fillna(0)
    agg['solar_connections'] = agg['solar_connections'].fillna(0)

    cap = DEFAULT_CAPACITY_KW
    agg['ESTIMATED_CAPACITY_kW'] = cap
    agg['current_load_kW'] = (agg['avg_consumption'] / 720).fillna(0)
    agg['utilization_rate'] = (agg['current_load_kW'] / cap).clip(0, 1)
    agg['available_headroom'] = cap - agg['current_load_kW'] - agg['total_solar_capacity']
    agg['solar_penetration'] = (agg['total_solar_capacity'] / cap).clip(0)
    agg['demand_volatility'] = (agg['consumption_std'] / 720).fillna(0)
    agg['export_ratio'] = (agg['avg_export'] / (agg['avg_import'] + 1)).fillna(0)

    return agg.fillna(0)


@st.cache_resource(show_spinner="Training models…")
def train_models(csv_path: str):
    df = load_data(csv_path)
    X_raw = df[FEATURE_COLS].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    y = (df['utilization_rate'] < df['utilization_rate'].median()).astype(int)

    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_scaled, y)

    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    raw_labels = km.fit_predict(X_scaled)
    centroid_util = km.cluster_centers_[:, 2]
    rank_order = np.argsort(centroid_util)
    label_map = {orig: new for new, orig in enumerate(rank_order)}
    labels = np.vectorize(label_map.get)(raw_labels)

    lr = LinearRegression()
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['current_load_kW'].values)

    return scaler, rf, km, lr, label_map


CLUSTER_NAMES = {
    0: 'Underutilised — Low Risk',
    1: 'Balanced Load — Medium Risk',
    2: 'High Utilisation — High Risk',
}


def score_color(s):
    if s >= 80: return "score-green"
    if s >= 60: return "score-blue"
    if s >= 40: return "score-amber"
    return "score-red"


def score_label(s):
    if s >= 80: return "IDEAL"
    if s >= 60: return "GOOD"
    if s >= 40: return "FAIR"
    return "POOR"


def marker_color(s):
    if s >= 80: return "green"
    if s >= 60: return "blue"
    if s >= 40: return "orange"
    return "red"


def util_color(u):
    if u <= 70: return "score-green"
    if u <= 85: return "score-amber"
    return "score-red"


def get_capacity_recommendation(score, available_headroom):
    if score < 60:
        return None
    safe_max = available_headroom * 0.80
    recommended = None
    for tier in CAPACITY_TIERS:
        if tier <= safe_max:
            recommended = tier
    if recommended is None:
        return None
    return {'kw': recommended, 'safe_max': round(safe_max, 1)}


def generate_recommendation(score, headroom):
    if score >= 80:
        return f"Highly suitable. {headroom['available_headroom_kW']:.1f} kW available headroom. Proceed with grid connection."
    if score >= 60:
        return f"Conditionally suitable. Post-connection utilisation will be {headroom['utilization_after'] * 100:.1f}%. Review with utility provider."
    if score >= 40:
        return "Marginal suitability due to limited headroom. An upgrade assessment is recommended before connecting."
    return f"Not suitable. Only {headroom['available_headroom_kW']:.1f} kW headroom available. Select a different transformer."


def run_assessment(transformer_data, scaler, rf, km, lr, label_map,
                   user_lat, user_lon, solar_kw, radius_m):
    df = transformer_data.copy()

    # Distance filter
    df['DISTANCE_M'] = df.apply(
        lambda r: geodesic((user_lat, user_lon),
                           (r['TRANSFORMER_LAT'], r['TRANSFORMER_LON'])).meters, axis=1
    )
    nearby = df[df['DISTANCE_M'] <= radius_m].copy().reset_index(drop=True)

    if nearby.empty:
        return None

    # ML predictions
    X_scaled = scaler.transform(nearby[FEATURE_COLS].fillna(0))
    proba = rf.predict_proba(X_scaled)
    ml_scores = proba[:, 1] * 100 if proba.shape[1] > 1 else np.full(len(nearby), 50.0)

    raw_clusters = km.predict(X_scaled)
    clusters = np.vectorize(label_map.get)(raw_clusters)

    results = []
    for i, row in nearby.iterrows():
        cap = float(row['ESTIMATED_CAPACITY_kW'])
        current_load = float(row['current_load_kW'])
        existing_sol = float(row['total_solar_capacity'])
        dist = float(row['DISTANCE_M'])

        total_before = current_load + existing_sol
        available = cap - total_before
        safe_headroom = cap * SAFETY_MARGIN - total_before
        total_after = total_before + solar_kw

        can_support = available >= solar_kw
        curtailment_risk = (total_after / cap) > CURTAILMENT_THRESH
        util_before = total_before / cap
        util_after = total_after / cap

        headroom = {
            'available_headroom_kW': available,
            'safe_headroom_kW': safe_headroom,
            'utilization_before': util_before,
            'utilization_after': util_after,
        }

        # Headroom score
        ratio = safe_headroom / max(solar_kw, 0.001)
        if ratio >= 1.5:
            h_score = 100
        elif ratio >= 1.0:
            h_score = 80
        elif available >= solar_kw:
            h_score = 50
        else:
            h_score = 0

        # Distance score
        d_score = max(0.0, 100 * math.exp(-dist / 1000))

        # Stability score
        if util_after <= 0.70:
            s_score = 100
        elif util_after <= 0.85:
            s_score = 75
        elif util_after <= 0.95:
            s_score = 40
        else:
            s_score = 0

        rule_score = h_score * 0.40 + d_score * 0.30 + s_score * 0.30
        ml_score = float(ml_scores[i])
        blended = rule_score * 0.55 + ml_score * 0.45

        future_load = float(lr.predict([[12]])[0])
        cap_rec = get_capacity_recommendation(blended, available)
        recommendation = generate_recommendation(blended, headroom)

        results.append({
            'code': str(row['TRANSFORMER_CODE']),
            'lat': float(row['TRANSFORMER_LAT']),
            'lon': float(row['TRANSFORMER_LON']),
            'distance': round(dist, 0),
            'score': round(blended, 2),
            'ruleScore': round(rule_score, 2),
            'mlScore': round(ml_score, 2),
            'headroomScore': round(h_score, 2),
            'distanceScore': round(d_score, 2),
            'stabilityScore': round(s_score, 2),
            'label': score_label(blended),
            'capacity': cap,
            'currentLoad': round(current_load, 2),
            'existingSolar': round(existing_sol, 2),
            'availableHeadroom': round(available, 2),
            'safeHeadroom': round(safe_headroom, 2),
            'utilBefore': round(util_before * 100, 1),
            'utilAfter': round(util_after * 100, 1),
            'canSupport': bool(can_support),
            'curtailmentRisk': bool(curtailment_risk),
            'cluster': CLUSTER_NAMES.get(int(clusters[i]), 'Unknown'),
            'futureLoad12m': round(future_load, 2),
            'recommendation': recommendation,
            'capacityRec': cap_rec,
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    for idx, r in enumerate(results):
        r['rank'] = idx + 1

    return results
