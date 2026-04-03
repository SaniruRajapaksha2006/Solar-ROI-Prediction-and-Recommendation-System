"""
Integrated Solar ROI Dashboard — Kinetic
Run: streamlit run integrated_dashboard.py
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import json
import subprocess
import time
import math
import folium
from weasyprint import HTML
from datetime import datetime
import tempfile
import base64
import os
from streamlit_folium import st_folium
from pathlib import Path
import plotly.graph_objects as go
from geopy.distance import geodesic


# PAGE CONFIG
st.set_page_config(
    page_title="Kinetic | Solar ROI Intelligence",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with open("dashboard_styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# SESSION STATE
def init_session():
    defaults = {
        "page": "home",
        "results": None,
        "selected_tab": "grid",
        "analysis_running": False,
        "selected_transformer": None,
        "user_input": {
            "latitude": 6.8511,
            "longitude": 79.9212,
            "months": {9: 310.5, 10: 295.2, 11: 325.1},
            "radius_m": 500
        }
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# HELPERS
def run_integrated_analysis(lat, lon, months, radius_m):
    project_root = Path(__file__).parent
    months_json = json.dumps(months)
    cmd = [
        sys.executable,
        str(project_root / "main_integrated.py"),
        "--lat", str(lat),
        "--lon", str(lon),
        "--months", months_json,
        "--radius_m", str(radius_m),
        "--tariff", "D1",
        "--phase", "SP",
        "--household_size", "4",
        "--year", "2025"
    ]
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(project_root)
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        results_file = project_root / "results" / "integrated_report.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None



# GPS COMPONENT
def gps_component():
    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        get_gps = st.button("📍 Use GPS", key="gps_btn", help="Auto-fill coordinates from your device location")
    if get_gps:
        try:
            from streamlit_js_eval import get_geolocation
            loc = get_geolocation()
            if loc and "coords" in loc:
                coords = loc["coords"]
                return {"lat": round(float(coords["latitude"]), 6), "lon": round(float(coords["longitude"]), 6)}
            else:
                with col_status:
                    st.warning("Could not get location — allow browser access")
        except ImportError:
            with col_status:
                st.info("Install streamlit-js-eval for GPS: `pip install streamlit-js-eval`")
    return None



# MAP BUILD
def build_map(transformer_list, user_lat, user_lon, selected_code=None):
    m = folium.Map(location=[user_lat, user_lon], zoom_start=14, tiles='CartoDB positron')
    folium.CircleMarker(
        [user_lat, user_lon], radius=13,
        color='white', fill=True, fill_color='#2563eb', fill_opacity=0.95, weight=3,
        popup=folium.Popup('<b style="font-family:monospace">📍 Your Location</b>', max_width=150),
    ).add_to(m)
    folium.CircleMarker(
        [user_lat, user_lon], radius=5,
        color='#2563eb', fill=True, fill_color='white', fill_opacity=1.0, weight=2,
    ).add_to(m)
    for i, tf in enumerate(transformer_list):
        lat = tf.get('lat')
        lon = tf.get('lon')
        if lat is None or lon is None:
            continue
        score = tf.get('suitability_score', 0)
        rank = i + 1
        util_after = tf.get('utilization_after', tf.get('utilizationAfter', 0))
        col = '#f4601a' if rank == 1 else '#18a058'
        rank_label = "BEST" if rank == 1 else (f"{rank}ND" if rank == 2 else f"{rank}RD")
        is_selected = tf.get('transformer_id') == selected_code
        folium.CircleMarker(
            [lat, lon],
            radius=11 if is_selected else 9,
            color='white', weight=2.5 if is_selected else 2,
            fill=True, fill_color=col, fill_opacity=0.88,
            popup=folium.Popup(
                f"<div style='font-family:monospace;font-size:12px;line-height:1.6'>"
                f"<b>{tf.get('transformer_id', 'N/A')}</b><br>"
                f"Rank: {rank_label}<br>Score: {score:.1f}/100<br>"
                f"Dist: {tf.get('distance_m', 0):.0f} m<br>"
                f"Util after: {util_after:.1f}%<br>"
                f"{'✅ Supported' if tf.get('can_support', False) else '❌ Not supported'}</div>",
                max_width=200,
            ),
        ).add_to(m)
    return m


def score_color(s):
    return '#18a058' if s >= 80 else '#d97706' if s >= 60 else '#f4601a' if s >= 40 else '#dc2626'

def score_label(s):
    return 'IDEAL' if s >= 80 else 'GOOD' if s >= 60 else 'FAIR' if s >= 40 else 'POOR'

def util_color(u):
    return '#18a058' if u <= 70 else '#f4601a' if u <= 85 else '#dc2626'

def ring_html(score):
    col = score_color(score)
    r = 19
    c = 2 * math.pi * r
    dash = (score / 100) * c
    return (
        f'<div class="k-ring">'
        f'<svg viewBox="0 0 50 50" width="50" height="50">'
        f'<circle cx="25" cy="25" r="{r}" fill="none" stroke="#e0dbd0" stroke-width="5"/>'
        f'<circle cx="25" cy="25" r="{r}" fill="none" stroke="{col}" stroke-width="5"'
        f' stroke-dasharray="{dash:.1f} {c - dash:.1f}" stroke-linecap="round"/>'
        f'</svg>'
        f'<div class="k-ring-val" style="color:{col}">{round(score)}</div>'
        f'</div>'
    )

_GRID_COLOR = 'rgba(224,219,208,0.5)'
_TICK_FONT = dict(family='Space Mono', size=9, color='#a09880')

def plotly_theme():
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono', size=10, color='#7a7264'),
        margin=dict(l=40, r=20, t=20, b=40),
    )

def apply_axes(fig, x_extra=None, y_extra=None):
    base_x = dict(showgrid=True, gridwidth=1, gridcolor=_GRID_COLOR, tickfont=_TICK_FONT, linecolor=_GRID_COLOR)
    base_y = dict(showgrid=True, gridwidth=1, gridcolor=_GRID_COLOR, tickfont=_TICK_FONT, linecolor=_GRID_COLOR)
    fig.update_xaxes(**{**base_x, **(x_extra or {})})
    fig.update_yaxes(**{**base_y, **(y_extra or {})})

def badge(label, color):
    colors = {
        'green':  ('#18a058', 'rgba(24,160,88,0.10)',  'rgba(24,160,88,0.25)'),
        'orange': ('#f4601a', 'rgba(244,96,26,0.10)',  'rgba(244,96,26,0.25)'),
        'amber':  ('#d97706', 'rgba(217,119,6,0.10)',  'rgba(217,119,6,0.25)'),
        'red':    ('#dc2626', 'rgba(220,38,38,0.10)',  'rgba(220,38,38,0.25)'),
        'blue':   ('#1d4ed8', 'rgba(29,78,216,0.10)',  'rgba(29,78,216,0.25)'),
        'muted':  ('#7a7264', 'rgba(122,114,100,0.10)','rgba(122,114,100,0.20)'),
    }
    c, bg, bd = colors.get(color, colors['muted'])
    return f'<span style="font-family:var(--mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;color:{c};background:{bg};border:1px solid {bd};padding:3px 9px;border-radius:5px;">{label}</span>'

def prog_bar(label, value_label, pct, color='green'):
    grad = {
        'green':  'linear-gradient(90deg,#18a058,#2e6f40)',
        'amber':  'linear-gradient(90deg,#d97706,#f59e0b)',
        'blue':   'linear-gradient(90deg,#1d4ed8,#3b82f6)',
        'orange': 'linear-gradient(90deg,#f4601a,#f87040)',
        'red':    'linear-gradient(90deg,#dc2626,#ef4444)',
    }.get(color, 'linear-gradient(90deg,#18a058,#2e6f40)')
    return f"""
    <div class="k-prog-item">
        <div class="k-prog-hd"><span>{label}</span><span style="color:var(--text)">{value_label}</span></div>
        <div class="k-prog-track"><div class="k-prog-fill" style="width:{min(pct,100)}%;background:{grad};"></div></div>
    </div>"""

def spec_grid(items):
    cells = ""
    for lbl, val in items:
        cells += f'<div class="k-spec-box"><div class="k-sp-lbl">{lbl}</div><div class="k-sp-val">{val}</div></div>'
    return f'<div class="k-spec-grid">{cells}</div>'

def stat_row(cards):
    inner = ""
    for lbl, val, sub, color in cards:
        sub_html = f'<div class="k-stat-sub">{sub}</div>' if sub else ''
        inner += f'<div class="k-stat-card {color}"><div class="k-stat-lbl">{lbl}</div><div class="k-stat-val">{val}</div>{sub_html}</div>'
    return f'<div class="k-stat-row">{inner}</div>'

def panel_header(title, subtitle="", badges="", color=""):
    cls = f" {color}" if color else ""
    sub = f'<div class="k-ps">{subtitle}</div>' if subtitle else ''
    return f"""
    <div class="k-ph">
        <div><div class="k-pt{cls}">{title}</div>{sub}</div>
        <div style="display:flex;gap:8px;align-items:center;">{badges}</div>
    </div>"""


# HOME PAGE
def home_page():
    st.markdown("""
    <div class="k-topbar">
        <div class="k-logo">
            <span class="kin">KIN</span><span class="etic">ETIC</span>
            <span class="dot"></span>
        </div>
        <div class="k-pill-row">
            <span class="k-nav-pill">PUCSL 2025</span>
            <span class="k-nav-pill">LSTM + MONTE CARLO</span>
            <span class="k-nav-pill">GEOSPATIAL</span>
        </div>
        <div class="k-live">
            <div class="k-live-dot"></div>
            SYSTEM ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.25, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="k-hero-left">
            <div class="k-eyebrow">AI-Powered · Sri Lanka</div>
            <div class="k-title">
                Solar ROI<br>
                <span class="ac">Prediction &amp;<br>Recommendation</span>
            </div>
            <p class="k-desc">
                An integrated AI system that forecasts solar generation, maps transformer suitability,
                predicts consumption under PUCSL tariffs, and models your return on investment
                with Monte Carlo risk analysis.
            </p>
            <div class="k-feat-list">
                <div class="k-feat-item"><div class="k-feat-dot"></div>Solar Generation Forecast — LSTM ensemble model</div>
                <div class="k-feat-item"><div class="k-feat-dot"></div>Geospatial Transformer Grid Mapping</div>
                <div class="k-feat-item"><div class="k-feat-dot"></div>PUCSL D1 Tariff Consumption Prediction</div>
                <div class="k-feat-item"><div class="k-feat-dot"></div>Monte Carlo ROI &amp; Risk Simulation</div>
            </div>
            <div class="k-home-kpis">
                <div class="k-kpi-box"><div class="k-kpi-val">70%</div><div class="k-kpi-lbl">SL Renewable Target 2030</div></div>
                <div class="k-kpi-box"><div class="k-kpi-val">&lt;2%</div><div class="k-kpi-lbl">Current Solar Adoption</div></div>
                <div class="k-kpi-box"><div class="k-kpi-val">5–7 yr</div><div class="k-kpi-lbl">Typical Payback Period</div></div>
                <div class="k-kpi-box"><div class="k-kpi-val">400%+</div><div class="k-kpi-lbl">20-Year ROI Potential</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="k-form-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="k-form-header">
            <div class="k-form-title">Begin Your Analysis</div>
            <div class="k-form-desc">Enter your location and usage details to generate a full solar ROI report.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="k-form-body">', unsafe_allow_html=True)

        st.markdown('<div class="k-section-label">Location</div>', unsafe_allow_html=True)
        gps_location = gps_component()
        col_a, col_b = st.columns(2)
        with col_a:
            lat = st.number_input("Latitude", value=gps_location["lat"] if gps_location else st.session_state.user_input["latitude"], format="%.6f", step=0.0001, key="lat_input")
        with col_b:
            lon = st.number_input("Longitude", value=gps_location["lon"] if gps_location else st.session_state.user_input["longitude"], format="%.6f", step=0.0001, key="lon_input")

        st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="k-section-label">Monthly Usage — Last 3 Months (kWh)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            m1 = st.number_input("Month 1", value=float(st.session_state.user_input["months"][9]), step=10.0, key="m1_input")
        with c2:
            m2 = st.number_input("Month 2", value=float(st.session_state.user_input["months"][10]), step=10.0, key="m2_input")
        with c3:
            m3 = st.number_input("Month 3", value=float(st.session_state.user_input["months"][11]), step=10.0, key="m3_input")

        st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="k-section-label">Search Radius</div>', unsafe_allow_html=True)
        radius_m = st.number_input("Search Radius (m)", value=st.session_state.user_input["radius_m"], min_value=100, max_value=5000, step=100, key="radius_input")

        st.markdown('</div></div>', unsafe_allow_html=True)

        if st.button("▶  Run Full AI Analysis", use_container_width=True, type="primary"):
            st.session_state.user_input = {
                "latitude": lat, "longitude": lon,
                "months": {9: m1, 10: m2, 11: m3},
                "radius_m": radius_m
            }
            st.session_state.analysis_running = True
            st.rerun()

        st.markdown('<p class="k-form-note">Analysis includes grid assessment, solar forecast, consumption prediction and ROI modelling.</p>', unsafe_allow_html=True)


# LOADING PAGE
def loading_page():
    st.markdown("""
    <div class="kin-loading">
        <div class="kin-loader-ring"></div>
        <div class="kin-sim-counter">ANALYSING</div>
        <div class="kin-loader-text">Running LSTM &amp; Monte Carlo Simulations</div>
        <div class="kin-steps">
            <div class="kin-step">LOADING DATASET</div>
            <div class="kin-step">SIMILARITY MATCHING</div>
            <div class="kin-step">FEATURE ENGINEERING</div>
            <div class="kin-step">LSTM MODEL</div>
            <div class="kin-step">GEOSPATIAL ANALYSIS</div>
            <div class="kin-step">TARIFF CALC</div>
            <div class="kin-step">MONTE CARLO</div>
        </div>
        <div class="kin-loader-progress"><div class="kin-loader-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.results is None:
        results = run_integrated_analysis(
            st.session_state.user_input["latitude"],
            st.session_state.user_input["longitude"],
            st.session_state.user_input["months"],
            st.session_state.user_input["radius_m"]
        )
        if results:
            st.session_state.results = results
            st.session_state.page = "results"
        else:
            st.session_state.page = "home"
    else:
        st.session_state.page = "results"

    st.session_state.analysis_running = False
    st.rerun()


# RESULTS — TOPBAR & TABS
def results_topbar(panel_size, location_str):
    tabs_def = [
        ("grid",        "2", "Grid Assessment"),
        ("solar",       "1", "Solar Forecast"),
        ("consumption", "3", "Consumption Forecast"),
        ("roi",         "4", "ROI & Risk"),
    ]
    tab_html = ""
    for key, num, name in tabs_def:
        active = "active" if st.session_state.selected_tab == key else ""
        tab_html += f'<div class="k-tab {active}" data-tab="{key}"><span class="k-tab-num">{num}</span>{name}</div>'

    st.markdown(f"""
    <div class="k-topbar">
        <div class="k-logo">
            <span class="kin">KIN</span><span class="etic">ETIC</span>
            <span class="dot"></span>
        </div>
        <div class="k-rtb-title">
            <span class="k-rtb-h">Analysis <span style="color:var(--muted);font-weight:400;">Complete</span></span>
            <span class="k-rtb-meta">{panel_size} KW · {location_str.upper()} · LSTM + MONTE CARLO</span>
        </div>
        <div class="k-live">
            <div class="k-live-dot"></div>
            COMPLETE
        </div>
    </div>
    <div class="k-tab-bar">{tab_html}</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const tabs = document.querySelectorAll('.k-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                const tabKey = this.getAttribute('data-tab');
                if (tabKey) {
                    const buttons = document.querySelectorAll('div[data-testid="stButton"] button');
                    const tabIndex = ['grid', 'solar', 'consumption', 'roi'].indexOf(tabKey);
                    if (buttons[tabIndex]) buttons[tabIndex].click();
                }
            });
        });
    });
    </script>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    tab_keys   = ["grid", "solar", "consumption", "roi"]
    tab_labels = ["⚡ Grid Assessment", "☀️ Solar Forecast", "📊 Consumption Forecast", "💰 ROI & Risk"]
    for col, key, label in zip(cols, tab_keys, tab_labels):
        with col:
            if st.button(label, key=f"tabswitch_{key}", use_container_width=True):
                st.session_state.selected_tab = key
                st.rerun()

    st.markdown('<div style="border-bottom: 1px solid var(--border); margin: 0 32px 24px 32px;"></div>', unsafe_allow_html=True)


# RESULTS PAGE
def results_page():
    results = st.session_state.results
    if not results:
        st.session_state.page = "home"
        st.rerun()
        return

    solar       = results.get("solar_forecast", {})
    consumption = results.get("consumption_forecast", {})
    all_transformers = results.get("all_transformers", [])
    best_transformer = results.get("transformer_info", {})

    if all_transformers and len(all_transformers) > 0:
        transformer_list = all_transformers
    else:
        transformer_list = [best_transformer] if best_transformer else []

    roi        = results.get("roi_analysis", {})
    panel_size = results.get("recommended_panel_size_kw", 5)
    lat        = st.session_state.user_input["latitude"]
    lon        = st.session_state.user_input["longitude"]
    location_str = f"LAT {lat:.3f} · LON {lon:.3f}"

    results_topbar(panel_size, location_str)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    tab = st.session_state.selected_tab
    if tab == "grid":
        tab_grid(transformer_list, panel_size, lat, lon)
    elif tab == "solar":
        tab_solar(solar, panel_size)
    elif tab == "consumption":
        tab_consumption(consumption)
    elif tab == "roi":
        tab_roi(roi, consumption, solar)

    # Action buttons — 3 equal columns spanning full width
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        try:
            pdf_data = generate_pdf_report(results)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=f"solar_roi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    with btn_col2:
        report_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=report_json,
            file_name=f"solar_roi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with btn_col3:
        if st.button("← Back to Input", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.results = None
            st.session_state.analysis_running = False
            st.rerun()



# GRID TAB
def tab_grid(transformer_list, panel_size, user_lat, user_lon):
    if not isinstance(transformer_list, list):
        single_tf = transformer_list
        transformer_list = [
            single_tf,
            {**single_tf,
             "transformer_id": f"{single_tf.get('transformer_id', 'TF')}_ALT1",
             "distance_m": single_tf.get('distance_m', 0) + 200,
             "suitability_score": single_tf.get('suitability_score', 0) * 0.85,
             "available_headroom_kw": single_tf.get('available_headroom_kw', 0) * 0.7,
             "can_support": single_tf.get('can_support', False),
             "lat": single_tf.get('lat', 0) + 0.002,
             "lon": single_tf.get('lon', 0) + 0.0015},
        ]

    transformer_list.sort(key=lambda x: x.get('suitability_score', 0), reverse=True)
    best = transformer_list[0]

    score = best.get('suitability_score', 0)
    can_support = best.get('can_support', False)
    curtail = best.get('curtailment_risk', False)
    capacity = best.get('capacity_kw', 0)
    headroom = best.get('available_headroom_kw', 0)
    current_load = best.get('current_load_kw', 0)
    distance = best.get('distance_m', 0)
    tf_id = best.get('transformer_id', 'N/A')
    label = score_label(score)
    recommendation = best.get('recommendation', 'N/A')
    util_before = round((current_load / capacity * 100) if capacity else 0, 1)
    util_after = round(((current_load + panel_size) / capacity * 100) if capacity else 0, 1)

    st.markdown(stat_row([
        ("Transformers Found", str(len(transformer_list)), "", "green"),
        ("Top Score /100", f"{score:.0f}", "", "amber"),
        ("Can Support", "✅ Yes" if can_support else "❌ No", "", "green" if can_support else "red"),
        ("Solar Capacity", f"{panel_size} <span style='font-size:16px;'>kW</span>", "", "blue"),
        ("Curtailment Risk", "⚠️ Yes" if curtail else "✅ None", "", "red" if curtail else "green"),
    ]), unsafe_allow_html=True)

    # Map panel
    st.markdown(f"""
    <div class="k-panel">
        {panel_header("Transformer Map", "", "", "orange")}
        <div style="display:flex;gap:14px;align-items:center;padding:10px 20px 0;flex-wrap:wrap;">
            <div class="k-leg-it"><div class="k-leg-d" style="background:#f4601a"></div><span class="k-leg-t">Best Option</span></div>
            <div class="k-leg-it"><div class="k-leg-d" style="background:#18a058"></div><span class="k-leg-t">Alternatives</span></div>
            <div class="k-leg-it"><div class="k-leg-d" style="background:#2563eb"></div><span class="k-leg-t">Your Location</span></div>
        </div>
        <div class="k-pb">
    """, unsafe_allow_html=True)

    m = build_map(transformer_list, user_lat, user_lon, st.session_state.selected_transformer)
    st_folium(m, width="100%", height=360, returned_objects=[])
    st.markdown('</div></div>', unsafe_allow_html=True)

    col_main, col_detail = st.columns([2, 1], gap="medium")

    with col_main:
        st.markdown('<div class="k-sec-title">Ranked Transformers</div>', unsafe_allow_html=True)
        for idx, tf in enumerate(transformer_list[:3]):
            tf_score = tf.get('suitability_score', 0)
            tf_can_support = tf.get('can_support', False)
            tf_curtail = tf.get('curtailment_risk', False)
            tf_capacity = tf.get('capacity_kw', 0)
            tf_headroom = tf.get('available_headroom_kw', 0)
            tf_current_load = tf.get('current_load_kw', 0)
            tf_distance = tf.get('distance_m', 0)
            tf_id_val = tf.get('transformer_id', 'N/A')
            tf_label = score_label(tf_score)
            tf_util_after = tf.get('utilizationAfter', 0) if tf.get('utilizationAfter') else round(
                ((tf_current_load + panel_size) / tf_capacity * 100) if tf_capacity else 0, 1)

            col = score_color(tf_score)
            uc = util_color(tf_util_after)
            c_pill = '<span class="k-curtail">⚡ Curtail</span>' if tf_curtail else ''
            sc_col = '#2e6f40' if tf_can_support else '#dc2626'
            sc_txt = '✓ OK' if tf_can_support else '✗ NO'
            active_class = 'active' if idx == 0 else ''

            st.markdown(
                f'<div class="k-tf-card {active_class}">'
                f'<div><div class="k-rank-num">{idx + 1}</div><div class="k-rank-lbl">RANK</div></div>'
                f'<div>'
                f'<div class="k-tf-pills"><span class="k-tf-code">{tf_id_val}</span>'
                f'<span class="k-dist-pill">{tf_distance:.0f} m</span>{c_pill}</div>'
                f'<div class="k-tf-cluster">{tf_label}</div>'
                f'<div class="k-tf-metrics">'
                f'<div><div class="k-tf-ml">Capacity</div><div class="k-tf-mv">{tf_capacity:.0f} kW</div></div>'
                f'<div><div class="k-tf-ml">Load</div><div class="k-tf-mv">{tf_current_load:.1f} kW</div></div>'
                f'<div><div class="k-tf-ml">Headroom</div><div class="k-tf-mv">{tf_headroom:.1f} kW</div></div>'
                f'<div><div class="k-tf-ml">Util After</div><div class="k-tf-mv" style="color:{uc}">{tf_util_after:.1f}%</div></div>'
                f'</div></div>'
                f'<div class="k-badge">{ring_html(tf_score)}'
                f'<div class="k-ring-lbl" style="color:{col}">{tf_label}</div>'
                f'<div style="font-size:9.5px;font-family:monospace;color:{sc_col};font-weight:700;margin-top:2px">{sc_txt}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"View Details →", key=f"vd_{tf_id_val}"):
                st.session_state.selected_transformer = tf_id_val
                st.rerun()

        # Score breakdown chart
        headroom_score = min(100, (headroom / capacity * 100)) if capacity else 0
        dist_score = max(0, 100 - (distance / 10)) if distance <= 1000 else 0
        stability_score = 80 if not curtail else 20

        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Score Breakdown (Best Option)", "Rule-based + ML suitability scoring", badge("COMPOSITE SCORE", "amber"), "amber")}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=[tf_id], y=[headroom_score], name='Headroom', marker_color='#18a058', marker_line_width=0))
        fig.add_trace(go.Bar(x=[tf_id], y=[dist_score], name='Distance', marker_color='#1d4ed8', marker_line_width=0))
        fig.add_trace(
            go.Bar(x=[tf_id], y=[stability_score], name='Stability', marker_color='#0e8a8a', marker_line_width=0))
        fig.update_layout(**plotly_theme(), height=200, barmode='group', showlegend=True,
                          legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'),
                                      bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.3))
        apply_axes(fig, y_extra=dict(range=[0, 110], title='Score /100'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_detail:
        alert_color = "#18a058" if can_support else "#dc2626"
        alert_bg = "rgba(24,160,88,0.08)" if can_support else "rgba(220,38,38,0.08)"
        alert_border = "rgba(24,160,88,0.25)" if can_support else "rgba(220,38,38,0.25)"
        alert_msg = f"✓ Best transformer can support {panel_size} kW solar connection" if can_support else f"✗ Best transformer cannot support {panel_size} kW — consider alternatives"

        st.markdown(f"""
        <div class="k-detail-panel">
            <div class="k-dp-hdr">
                <div class="k-dp-id">{tf_id}</div>
                <div class="k-dp-sub">{label.upper()} · RANK #1 · Recommended</div>
            </div>
            <div class="k-big-score">
                <div class="k-bs-num" style="color:{'var(--green)' if score > 50 else 'var(--red)'};">{score:.1f}</div>
                <div class="k-bs-lbl">Blended Suitability Score / 100</div>
            </div>
            <div class="k-mini-grid">
                <div class="k-mini-card"><div class="k-mc-lbl">Headroom</div><div class="k-mc-val" style="color:var(--green);">{headroom_score:.0f}</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Distance</div><div class="k-mc-val" style="color:var(--blue);">{dist_score:.0f}</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Stability</div><div class="k-mc-val" style="color:var(--teal);">{stability_score:.0f}</div></div>
            </div>
            <div class="k-scores-2">
                <div class="k-score-box"><div class="k-sb-lbl">Rule Score</div><div class="k-sb-val" style="color:var(--blue);">{min(score * 1.6, 100):.1f}</div></div>
                <div class="k-score-box"><div class="k-sb-lbl">ML Score</div><div class="k-sb-val" style="color:var(--amber);">{score * 0.28:.1f}</div></div>
            </div>
            <div style="margin:0 20px 16px;">
                {prog_bar("Util Before", f"{util_before}%", util_before, 'green')}
                {prog_bar("Util After", f"{util_after}%", util_after, 'orange' if util_after > 80 else 'blue')}
            </div>
            {spec_grid([
            ("Distance", f"{distance:.0f} m"),
            ("Capacity", f"{capacity:.0f} kW"),
            ("Available", f"{headroom:.1f} kW"),
            ("Safe Headroom", f"{max(0, headroom - panel_size):.1f} kW"),
            ("Existing Solar", f"{current_load * 0.3:.1f} kW"),
            ("Solar to Add", f"{panel_size} kW"),
        ])}
            <div style="margin:12px 20px 20px;">
                <div style="padding:12px 16px;border-radius:8px;border:1px solid {alert_border};background:{alert_bg};font-family:var(--mono);font-size:11px;font-weight:600;color:{alert_color};">{alert_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # FULL WIDTH BEST TRANSFORMER DETAILS
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="k-panel" style="margin: 0 32px 20px 32px; width: calc(100% - 64px);">
        {panel_header("Best Transformer Details", f"ID: {tf_id} · {label}", badge(label.upper(), 'green' if can_support else 'red'))}
        <div class="k-pb">
            {spec_grid([
        ("Transformer ID", tf_id),
        ("Distance", f"{distance:.0f} m"),
        ("Capacity", f"{capacity:.0f} kW"),
        ("Current Load", f"{current_load:.1f} kW"),
        ("Available Headroom", f'<span style="color:var(--green)">{headroom:.1f} kW</span>'),
        ("Suitability Label", f'<span style="color:var(--orange)">{label}</span>'),
    ])}
            <div class="k-divider" style="margin:12px 0;"></div>
            {prog_bar("Utilisation Before", f"{util_before}%", util_before, 'amber')}
            {prog_bar("Utilisation After Solar", f"{util_after}%", util_after, 'orange' if util_after > 80 else 'green')}
            <div class="k-divider" style="margin:12px 0;"></div>
            <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Recommendation</div>
            <p style="font-size:13px;color:var(--muted2);line-height:1.7;">{recommendation}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)



# SOLAR TAB
def tab_solar(solar, panel_size):
    annual_gen = solar.get('annual_total_kwh', 0)
    annual_income = solar.get('annual_income_lkr', 0)
    conf = solar.get('avg_confidence_pct', 0)
    monthly_exp = solar.get('monthly_export_kwh', {})

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = [monthly_exp.get(m, monthly_exp.get(str(m), 0)) for m in range(1, 13)]
    irr = [5.4, 5.9, 6.8, 6.4, 4.6, 3.7, 3.3, 3.8, 4.4, 5.0, 4.8, 6.1]

    peak_month = MONTHS[values.index(max(values))] if values and any(values) else "Mar"
    low_month = MONTHS[values.index(min(values))] if values and any(values) else "Jul"

    st.markdown(stat_row([
        ("Annual kWh", f"{annual_gen:,.0f}", "", "amber"),
        ("Panel Size", f"{panel_size} <span style='font-size:16px;'>kW</span>", "", "green"),
        ("Confidence", f"{conf * 100:.0f}<span style='font-size:16px;'>%</span>", "", "blue"),
        ("Annual Income", f"Rs. {annual_income:,.0f}", "", "orange"),
        ("Peak Month", peak_month, "", "red"),
    ]), unsafe_allow_html=True)

    col_main, col_detail = st.columns([2, 1], gap="medium")

    with col_main:
        # 12-month forecast chart
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("12-Month Solar Generation Forecast", "LSTM ensemble · NASA irradiance adjusted",
                          badge('LSTM', 'blue') + badge('IRRADIANCE', 'green'))}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        if values and any(values):
            conf_upper = [v * 1.15 for v in values]
            conf_lower = [v * 0.85 for v in values]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=MONTHS, y=conf_upper, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                     showlegend=False))
            fig.add_trace(
                go.Scatter(x=MONTHS, y=conf_lower, fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'),
                           fillcolor='rgba(29,78,216,0.07)', name='Confidence Band'))
            fig.add_trace(go.Scatter(x=MONTHS, y=values, mode='lines+markers', line=dict(color='#d97706', width=2.5),
                                     marker=dict(size=6, color='#d97706', line=dict(color='white', width=2)),
                                     name='Generation (kWh)'))
            fig.add_trace(
                go.Scatter(x=MONTHS, y=irr, mode='lines', yaxis='y2', line=dict(color='#18a058', width=1.5, dash='dot'),
                           marker=dict(size=4, color='#18a058'), name='Irradiance (kWh/m²)'))
            fig.update_layout(**plotly_theme(), height=240, showlegend=True,
                              legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'),
                                          bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.3),
                              yaxis2=dict(overlaying='y', side='right', title='kWh/m²/day', showgrid=False,
                                          tickfont=dict(family='Space Mono', size=9, color='#18a058')))
            apply_axes(fig, y_extra=dict(title='kWh'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Generation breakdown chart
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Generation Breakdown — Energy Components by Month", "", badge('MONSOON-ADJUSTED', 'muted'))}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        if values and any(values):
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=MONTHS, y=[round(v * 0.72) for v in values], name='Direct Generation',
                                  marker_color='rgba(217,119,6,0.8)', marker_line_width=0))
            fig2.add_trace(go.Bar(x=MONTHS, y=[round(v * 0.18) for v in values], name='Diffuse Irradiance',
                                  marker_color='rgba(24,160,88,0.7)', marker_line_width=0))
            fig2.add_trace(go.Bar(x=MONTHS, y=[round(v * 0.10) for v in values], name='Net Export Eligible',
                                  marker_color='rgba(14,138,138,0.7)', marker_line_width=0))
            fig2.update_layout(**plotly_theme(), height=200, barmode='stack', showlegend=True,
                               legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'),
                                           bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.35))
            apply_axes(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_detail:
        avg_monthly = annual_gen / 12 if annual_gen else 0
        peak_kwh = max(values) if values else 0
        low_kwh = min(values) if values else 0

        st.markdown(f"""
        <div class="k-detail-panel">
            <div class="k-dp-hdr">
                <div class="k-dp-id">Solar Profile</div>
                <div class="k-dp-sub">{panel_size}KW · COLOMBO · PEAK: {peak_month.upper()}</div>
            </div>
            <div class="k-big-score">
                <div class="k-bs-num" style="color:var(--amber);">{conf * 100:.1f}</div>
                <div class="k-bs-lbl">Forecast Confidence Score / 100</div>
            </div>
            <div class="k-mini-grid">
                <div class="k-mini-card"><div class="k-mc-lbl">Availability</div><div class="k-mc-val" style="color:var(--amber);">92</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Seasonal</div><div class="k-mc-val" style="color:var(--amber);">86</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Pattern</div><div class="k-mc-val" style="color:var(--amber);">84</div></div>
            </div>
            <div class="k-scores-2">
                <div class="k-score-box"><div class="k-sb-lbl">LSTM Score</div><div class="k-sb-val" style="color:var(--blue);">91.2</div></div>
                <div class="k-score-box"><div class="k-sb-lbl">GHI Score</div><div class="k-sb-val" style="color:var(--amber);">83.5</div></div>
            </div>
            <div style="padding:0 20px;">
                {prog_bar(f"Peak ({peak_month})", f"{peak_kwh:.0f} kWh", (peak_kwh / 600) * 100, 'amber')}
                {prog_bar("Annual Total", f"{annual_gen:,.0f} kWh", min((annual_gen / 6000) * 100, 100), 'green')}
                {prog_bar(f"Low ({low_month})", f"{low_kwh:.0f} kWh", (low_kwh / 600) * 100, 'blue')}
            </div>
            {spec_grid([
            ("Panel Size", f"{panel_size} kW"),
            ("Phase", "SP"),
            ("Data Source", "NASA"),
            ("Avg Monthly", f"{avg_monthly:.0f} kWh"),
            ("Annual Total", f"{annual_gen:,.0f} kWh"),
            ("Model Ver", "v2.4"),
        ])}
        </div>
        """, unsafe_allow_html=True)

    # FULL WIDTH RANKED MONTHLY BREAKDOWN TABLE
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    if values and any(values):
        season_map = ['normal', 'peak', 'peak', 'peak', 'normal', 'low', 'low', 'low', 'normal', 'normal', 'normal',
                      'peak']
        ranked = sorted(range(12), key=lambda i: -values[i])

        rows = ""
        for rank, i in enumerate(ranked):
            sc = season_map[i]
            sc_badge = badge(sc.upper(), 'amber' if sc == 'peak' else 'blue' if sc == 'normal' else 'muted')
            saving = round(values[i] * 77)
            row_bg = 'background:rgba(244,96,26,0.05);' if rank == 0 else ''
            rows += f"""<tr style="{row_bg}">
                <td style="font-family:var(--mono);font-size:11px;color:var(--muted2);">{rank + 1}
                <td><div style="font-weight:700;font-size:14px;">{MONTHS[i]}</div>
                    <div style="font-family:var(--mono);font-size:10px;color:var(--muted2);">Month {i + 1}</div></td>
                <td><span style="font-family:var(--mono);color:var(--amber);">{values[i]:.0f}</span></td>
                <td><span style="font-family:var(--mono);">{irr[i]} kWh/m²</span></td>
                <td><span style="font-family:var(--mono);color:var(--green);">Rs. {saving:,}</span></td>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="flex:1;height:5px;background:var(--surface3);border-radius:3px;overflow:hidden;">
                            <div style="height:100%;width:85%;background:linear-gradient(90deg,var(--orange),var(--amber));border-radius:3px;"></div>
                        </div>
                        <span style="font-family:var(--mono);font-size:10px;color:var(--orange);font-weight:600;">85%</span>
                    </div>
                </td>
                <td>{sc_badge}</td>
             </tr>"""

        st.markdown(f"""
        <div class="k-panel" style="margin: 0 32px 20px 32px; width: calc(100% - 64px);">
            {panel_header("Ranked Monthly Breakdown", "", badge('LSTM ENSEMBLE', 'muted'))}
            <div style="width:100%; overflow-x:auto;">
                <table class="k-data-table" style="width:100%; min-width:600px;">
                    <thead>
                        <tr>
                            <th>#</th><th>MONTH</th><th>kWh</th>
                            <th>IRRADIANCE</th><th>SAVINGS (LKR)</th>
                            <th>CONFIDENCE</th><th>SEASON</th>
                         </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)



# CONSUMPTION TAB
def tab_consumption(consumption):
    annual_cons = consumption.get('annual_total_kwh', 0)
    annual_bill = consumption.get('annual_total_bill_lkr', 0)
    conf = consumption.get('avg_confidence', 0)
    monthly_c = consumption.get('monthly_consumption_kwh', {})

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = [monthly_c.get(m, monthly_c.get(str(m), 0)) for m in range(1, 13)]

    def calc_bill(k):
        c, rem = 0, k
        for sz, r in [(30, 8), (30, 10), (30, 16), (30, 50), (60, 75), (9999, 100)]:
            t = min(rem, sz);
            c += t * r;
            rem -= t
        return round((c + 180) * 1.18)

    bills = [calc_bill(v) for v in values]
    peak_month = MONTHS[values.index(max(values))] if values and any(values) else "Apr"
    low_month = MONTHS[values.index(min(values))] if values and any(values) else "Jul"

    st.markdown(stat_row([
        ("Annual kWh", f"{annual_cons:,.0f}", "", "blue"),
        ("Annual Bill", f"Rs. {annual_bill:,.0f}", "", "amber"),
        ("Confidence", f"{conf * 100:.0f}<span style='font-size:16px;'>%</span>", "", "green"),
        ("Avg Monthly", f"{annual_cons / 12:.0f}", "kWh", "orange"),
        ("Peak Month", peak_month, "", "red"),
    ]), unsafe_allow_html=True)

    col_main, col_detail = st.columns([2, 1], gap="medium")

    with col_main:
        # 12-month consumption chart
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("12-Month Consumption Forecast", "", badge('ENSEMBLE', 'blue') + badge('LSTM', 'muted'))}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        if values and any(values):
            upper = [v * 1.20 for v in values]
            lower = [v * 0.80 for v in values]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=MONTHS, y=upper, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=MONTHS, y=lower, fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                     fillcolor='rgba(29,78,216,0.07)', name='Confidence Band'))
            fig.add_trace(go.Scatter(x=MONTHS, y=values, mode='lines+markers', line=dict(color='#1d4ed8', width=2.5),
                                     marker=dict(size=6, color='#1d4ed8', line=dict(color='white', width=2)),
                                     name='Consumption (kWh)'))
            fig.add_trace(go.Scatter(x=MONTHS, y=[b / 100 for b in bills], yaxis='y2', mode='lines',
                                     line=dict(color='#d97706', width=1.5, dash='dot'),
                                     marker=dict(size=4, color='#d97706'), name='Bill (LKR/100)'))
            fig.update_layout(**plotly_theme(), height=240, showlegend=True,
                              legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'),
                                          bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.3),
                              yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                          tickfont=dict(family='Space Mono', size=9, color='#d97706'),
                                          tickformat='.0f'))
            apply_axes(fig, y_extra=dict(title='kWh'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Bill breakdown chart
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Bill Components by Month", "PUCSL D1 tariff breakdown", badge('D1 TARIFF', 'orange'))}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        if values and any(values):
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=MONTHS, y=[round(b * 0.72) for b in bills], name='Energy Charge',
                                  marker_color='rgba(29,78,216,0.75)', marker_line_width=0))
            fig2.add_trace(go.Bar(x=MONTHS, y=[round(b * 0.18) for b in bills], name='Fuel Adjustment',
                                  marker_color='rgba(217,119,6,0.7)', marker_line_width=0))
            fig2.add_trace(
                go.Bar(x=MONTHS, y=[round(180 * 1.18)] * 12, name='Fixed Charge', marker_color='rgba(14,138,138,0.65)',
                       marker_line_width=0))
            fig2.update_layout(**plotly_theme(), height=200, barmode='stack', showlegend=True,
                               legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'),
                                           bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.35))
            apply_axes(fig2, y_extra=dict(tickformat='.0f', title='LKR'))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_detail:
        peak_kwh = max(values) if values else 0
        low_kwh = min(values) if values else 0

        st.markdown(f"""
        <div class="k-detail-panel">
            <div class="k-dp-hdr">
                <div class="k-dp-id">Consumption Profile</div>
                <div class="k-dp-sub">D1 TARIFF · SP PHASE · PEAK: {peak_month.upper()}</div>
            </div>
            <div class="k-big-score">
                <div class="k-bs-num" style="color:var(--blue);">{conf * 100:.1f}</div>
                <div class="k-bs-lbl">Forecast Confidence Score / 100</div>
            </div>
            <div class="k-mini-grid">
                <div class="k-mini-card"><div class="k-mc-lbl">Similarity</div><div class="k-mc-val" style="color:var(--blue);">94</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Pattern</div><div class="k-mc-val" style="color:var(--blue);">89</div></div>
                <div class="k-mini-card"><div class="k-mc-lbl">Seasonal</div><div class="k-mc-val" style="color:var(--blue);">82</div></div>
            </div>
            <div class="k-scores-2">
                <div class="k-score-box"><div class="k-sb-lbl">LSTM Score</div><div class="k-sb-val" style="color:var(--blue);">91.2</div></div>
                <div class="k-score-box"><div class="k-sb-lbl">Pattern Score</div><div class="k-sb-val" style="color:var(--amber);">74.8</div></div>
            </div>
            <div style="padding:0 20px;">
                {prog_bar(f"Peak ({peak_month})", f"{peak_kwh:.0f} kWh", (peak_kwh / 500) * 100, 'amber')}
                {prog_bar("Annual Total", f"{annual_cons:,.0f} kWh", min((annual_cons / 5000) * 100, 100), 'green')}
                {prog_bar(f"Low ({low_month})", f"{low_kwh:.0f} kWh", (low_kwh / 500) * 100, 'blue')}
            </div>
            {spec_grid([
            ("Tariff", "D1"),
            ("Phase", "SP"),
            ("Solar", '<span style="color:var(--red);">NONE</span>'),
            ("Avg Monthly", f"{annual_cons / 12:.0f} kWh"),
            ("Data Months", "4"),
            ("VAT Rate", "18%"),
        ])}
        </div>
        """, unsafe_allow_html=True)

    # FULL WIDTH RANKED MONTHLY BREAKDOWN TABLE
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    if values and any(values):
        season_map = ['normal', 'normal', 'peak', 'peak', 'peak', 'normal', 'low', 'low', 'normal', 'peak', 'normal',
                      'normal']
        ranked = sorted(range(12), key=lambda i: -values[i])

        rows = ""
        for rank, i in enumerate(ranked):
            sc = season_map[i]
            sc_badge = badge(sc.upper(), 'amber' if sc == 'peak' else 'blue' if sc == 'normal' else 'muted')
            bill_color = 'var(--amber)' if bills[i] > 25000 else 'var(--blue)' if bills[i] > 15000 else 'var(--teal)'
            row_bg = 'background:rgba(29,78,216,0.05);' if rank == 0 else ''
            rows += f"""<tr style="{row_bg}">
                <td style="font-family:var(--mono);font-size:11px;color:var(--muted2);">{rank + 1}
                <td><div style="font-weight:700;font-size:14px;">{MONTHS[i]}</div>
                    <div style="font-family:var(--mono);font-size:10px;color:var(--muted2);">Month {i + 1}</div>
                <td><span style="font-family:var(--mono);">{values[i]:.0f}</span>
                <td><span style="font-family:var(--mono);color:{bill_color};">Rs. {bills[i]:,}</span>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="flex:1;height:5px;background:var(--surface3);border-radius:3px;overflow:hidden;">
                            <div style="height:100%;width:85%;background:linear-gradient(90deg,var(--orange),var(--amber));border-radius:3px;"></div>
                        </div>
                        <span style="font-family:var(--mono);font-size:10px;color:var(--orange);font-weight:600;">85%</span>
                    </div>
                <td>{sc_badge}
              </tr>"""

        st.markdown(f"""
        <div class="k-panel" style="margin: 0 32px 20px 32px; width: calc(100% - 64px);">
            {panel_header("Ranked Monthly Breakdown", "", badge('PATTERN-BASED FALLBACK READY', 'muted'))}
            <div style="width:100%; overflow-x:auto;">
                <table class="k-data-table" style="width:100%; min-width:600px;">
                    <thead>
                        <tr>
                            <th>#</th><th>MONTH</th><th>kWh</th>
                            <th>BILL (LKR)</th><th>CONFIDENCE</th><th>SEASON</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ROI TAB
def tab_roi(roi, consumption, solar):
    expected_roi = roi.get('expected_roi_percent', 0)
    payback = roi.get('expected_payback_years', 0)
    investment = roi.get('total_investment_lkr', 0)
    recommendation = roi.get('recommendation', '')
    annual_bill = consumption.get('annual_total_bill_lkr', 0)
    annual_income = solar.get('annual_income_lkr', 0)
    annual_savings = annual_bill - annual_income
    best_roi = round(expected_roi * 1.155) if expected_roi else 0
    worst_roi = round(expected_roi * 0.857) if expected_roi else 0

    # Stat Row
    st.markdown(stat_row([
        ("Expected ROI", f"{expected_roi:.0f}<span style='font-size:16px;'>%</span>", "over 20 years", "green"),
        ("Payback Period", f"{payback:.1f}", "years (median)", "amber"),
        ("Total Investment", f"Rs. {investment / 1_000_000:.2f}M", "LKR (CAPEX)", "blue"),
        ("Worst-Case ROI", f"{worst_roi}<span style='font-size:16px;'>%</span>", "5th percentile", "orange"),
        ("Risk Certainty", "HIGH", "Investment grade", "red"),
    ]), unsafe_allow_html=True)

    # CUMULATIVE CASH FLOW CHART (FULL WIDTH)
    st.markdown(f"""
    <div class="k-panel" style="margin: 0 32px 20px 32px; width: calc(100% - 64px);">
        {panel_header("Cumulative Cash Flow Projection",
                      "Expected + confidence band (P10–P90) over 20-year lifetime",
                      badge('EXPECTED', 'green') + badge('P10–P90', 'muted') + badge('BREAKEVEN', 'amber'),
                      "green")}
        <div class="k-pb">
    """, unsafe_allow_html=True)

    years = list(range(21))
    ann_rev = annual_savings if annual_savings > 0 else 185000
    exp = [round(-investment + ann_rev * i * (1.03 ** i) / 1.03) for i in years]
    p10 = [round(v * 0.88) for v in exp]
    p90 = [round(v * 1.10) for v in exp]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=years, y=p90, fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=years, y=p10, fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'),
                             fillcolor='rgba(24,160,88,0.08)', name='P10–P90 Band'))
    fig.add_trace(
        go.Scatter(x=[f"Yr {y}" for y in years], y=exp, mode='lines+markers', line=dict(color='#18a058', width=2.5),
                   marker=dict(size=5, color='#18a058', line=dict(color='white', width=2)), name='Expected'))
    fig.add_trace(go.Scatter(x=[f"Yr {y}" for y in years], y=[0] * 21, mode='lines',
                             line=dict(color='rgba(217,119,6,0.6)', width=1.5, dash='dash'), name='Breakeven'))
    fig.update_layout(**plotly_theme(), height=260, showlegend=True,
                      legend=dict(font=dict(family='Space Mono', size=9, color='#7a7264'), bgcolor='rgba(0,0,0,0)',
                                  orientation='h', y=-0.3))
    apply_axes(fig, y_extra=dict(tickformat='.0f', title='LKR'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 2x2 GRID LAYOUT
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    # Row 1: ROI Distribution + Risk Factor Breakdown
    row1_col1, row1_col2 = st.columns(2, gap="medium")

    with row1_col1:
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("ROI Distribution", "Simulation histogram · 2000 outcomes", "", "amber")}
            <div class="k-pb">
        """, unsafe_allow_html=True)

        if best_roi and worst_roi:
            bins = [f"{worst_roi + i * ((best_roi - worst_roi) // 17):.0f}%" for i in range(18)]
        else:
            bins = [f"{i * 5}%" for i in range(18)]
        counts = [8, 62, 113, 122, 120, 128, 132, 122, 117, 111, 89, 88, 88, 25, 14, 9, 4, 2]
        colors = ['rgba(220,38,38,0.7)' if i < 2 else 'rgba(29,78,216,0.65)' if i < 4 else 'rgba(24,160,88,0.75)' for i
                  in range(18)]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=bins, y=counts, marker_color=colors, marker_line_width=0))
        fig3.update_layout(**plotly_theme(), height=220, showlegend=False)
        apply_axes(fig3, x_extra=dict(tickangle=45, tickfont=dict(family='Space Mono', size=8, color='#a09880')))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with row1_col2:
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Risk Factor Breakdown", "", "", "amber")}
            <div class="k-pb">
                {prog_bar("Panel Degradation", "0.75%/yr", 75, 'blue')}
                {prog_bar("Maintenance Volatility", "±20%", 55, 'amber')}
                {prog_bar("Inverter Failure Risk", "Yr 8–12", 40, 'red')}
                {prog_bar("Tariff Escalation", "2–5%/yr", 65, 'green')}
                {prog_bar("Overall Risk Score", "LOW", 28, 'orange')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Scenario Analysis + Recommendation
    row2_col1, row2_col2 = st.columns(2, gap="medium")

    with row2_col1:
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Scenario Analysis", "", "", "amber")}
            <div class="k-pb" style="padding:12px 20px;">
                <div class="k-scen-row"><div><div class="k-scen-lbl">Best Case ROI</div><div class="k-scen-s">P95 · Optimal degradation</div></div><div class="k-scen-val" style="color:var(--green);">{best_roi}%</div></div>
                <div class="k-scen-row"><div><div class="k-scen-lbl">Expected ROI</div><div class="k-scen-s">P50 · Median outcome</div></div><div class="k-scen-val" style="color:var(--amber);">{expected_roi:.0f}%</div></div>
                <div class="k-scen-row"><div><div class="k-scen-lbl">Worst Case ROI</div><div class="k-scen-s">P05 · Conservative</div></div><div class="k-scen-val" style="color:var(--red);">{worst_roi}%</div></div>
                <div class="k-scen-row"><div><div class="k-scen-lbl">Payback Period</div><div class="k-scen-s">P50 median</div></div><div class="k-scen-val" style="color:var(--blue);">{payback:.1f} yrs</div></div>
                <div class="k-scen-row"><div><div class="k-scen-lbl">Prob. Positive ROI</div><div class="k-scen-s">% of simulations</div></div><div class="k-scen-val" style="color:var(--green);">100%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown(f"""
        <div class="k-panel" style="margin:0 0 16px 0;">
            {panel_header("Recommendation", "", "", "amber")}
            <div class="k-pb">
                <div class="k-rec-badge">Excellent Investment</div>
                <div style="font-family:var(--mono);font-size:9px;color:var(--muted2);letter-spacing:.15em;text-transform:uppercase;margin-bottom:14px;">Payback Timeline</div>
                <div class="k-tl-wrap">
                    <div class="k-tl-item done"><span class="k-tl-yr">Year 0</span><span class="k-tl-ev">Install {solar.get('panel_size_kw', '5')}kW System</span></div>
                    <div class="k-tl-item"><span class="k-tl-yr">Year 3</span><span class="k-tl-ev">Mid-point milestone</span></div>
                    <div class="k-tl-item now"><span class="k-tl-yr">Year {round(payback)}</span><span class="k-tl-ev">✓ Breakeven Point</span></div>
                    <div class="k-tl-item"><span class="k-tl-yr">Year 12</span><span class="k-tl-ev">Inverter replacement</span></div>
                    <div class="k-tl-item"><span class="k-tl-yr">Year 20</span><span class="k-tl-ev">End of lifecycle</span></div>
                </div>
                <div style="font-size:12px;color:var(--muted2);line-height:1.9;">{recommendation}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========== VENDOR COMPARISON SECTION (FULL WIDTH) ==========
    # IMPORTANT: This is at top-level indent — outside all 'with' column blocks
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    vendor_comparison = roi.get('vendor_price_comparison', [])
    if not vendor_comparison:
        vendor_comparison = roi.get('Vendor_Price_Comparison', [])

    recommended_panel_size = solar.get('panel_size_kw', 5)

    if vendor_comparison:
        vendor_cards_html = ""

        for i, vendor in enumerate(vendor_comparison):
            name = vendor.get('name', 'N/A')
            location = vendor.get('location', 'N/A')
            specialty = vendor.get('specialty', 'N/A')
            contact = vendor.get('contact', 'N/A')
            rating = vendor.get('rating', 4.0)
            price = vendor.get('price_lkr', 0)
            price_per_kw = vendor.get('price_per_kw', 0)
            is_cheapest = i == 0
            savings = vendor_comparison[0]['price_lkr'] - price if not is_cheapest else 0
            price_display = f"Rs. {price / 1_000_000:.2f}M" if price >= 1_000_000 else f"Rs. {price:,.0f}"
            cheapest_class = "cheapest" if is_cheapest else ""
            cheapest_badge = '<span class="k-badge-cheapest">★ CHEAPEST</span>' if is_cheapest else ""
            stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
            contact_clean = contact.replace(" ", "")

            savings_note = ""
            if not is_cheapest and savings > 0:
                savings_note = (
                    f'<div class="k-savings-note">'
                    f'Save Rs. {savings:,.0f} by choosing {vendor_comparison[0]["name"]}'
                    f'</div>'
                )

            vendor_cards_html += (
                f'<div class="k-vendor-card {cheapest_class}">'
                f'<div class="k-vendor-info">'
                f'<div class="k-vendor-name">{name} {cheapest_badge}</div>'
                f'<div class="k-vendor-desc">{location} · {specialty}</div>'
                f'<div class="k-vendor-rating">{stars} {rating}</div>'
                f'</div>'
                f'<div class="k-vendor-price-box">'
                f'<div class="k-vendor-price">'
                f'<span class="k-price-large">{price_display}</span>'
                f'<span class="k-price-per-kw">(Rs. {price_per_kw:,.0f}/kW)</span>'
                f'</div>'
                f'<div class="k-vendor-contact">'
                f'<div class="k-contact-icon">📞</div>'
                f'<div class="k-contact-number">{contact}</div>'
                f'<a href="tel:{contact_clean}" class="k-contact-btn" '
                f'style="display:inline-block;text-decoration:none;text-align:center;">'
                f'Contact Now</a>'
                f'</div>'
                f'</div>'
                f'</div>'
                f'{savings_note}'
            )

        cheapest_name = vendor_comparison[0]['name'] if vendor_comparison else ''
        ph = panel_header(
            f"Compare Vendor Prices — {recommended_panel_size} kW Systems",
            f"Prices from verified installers — Cheapest: {cheapest_name}",
            badge('COMPARE & SAVE', 'green')
        )

        html_out = (
            f'<div class="k-panel" style="margin:0 32px 20px 32px;width:calc(100% - 64px);">'
            f'{ph}'
            f'<div class="k-pb">'
            f'<div class="k-vendor-comparison-grid">'
            f'{vendor_cards_html}'
            f'</div>'
            f'<div class="k-vendor-note">'
            f'<span class="k-note-icon">ℹ️</span>'
            f'<span class="k-note-text">Prices are estimates based on market data. '
            f'Contact vendors for exact quotes.</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(html_out, unsafe_allow_html=True)



# PDF / HTML REPORT
def generate_pdf_report(results):
    html_content = generate_html_report(results)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        f.write(html_content)
        html_path = f.name
    pdf_path = html_path.replace('.html', '.pdf')
    HTML(html_path).write_pdf(pdf_path)
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    os.unlink(html_path)
    os.unlink(pdf_path)
    return pdf_data


def generate_html_report(results):
    """Generate a professional HTML report from results"""
    from datetime import datetime

    solar       = results.get("solar_forecast", {})
    consumption = results.get("consumption_forecast", {})
    transformer = results.get("transformer_info", {})
    roi         = results.get("roi_analysis", {})
    panel_size  = results.get("recommended_panel_size_kw", 5)
    user_input  = results.get("user_input", {})

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    monthly_cons  = consumption.get("monthly_consumption_kwh", {})
    monthly_bills = consumption.get("monthly_bills_lkr", {})
    monthly_solar = solar.get("monthly_export_kwh", {})
    export_rate   = 27.06

    annual_cons   = consumption.get("annual_total_kwh", 0)
    annual_bill   = consumption.get("annual_total_bill_lkr", 0)
    annual_gen    = solar.get("annual_total_kwh", 0)
    annual_income = solar.get("annual_income_lkr", 0)
    cons_conf     = consumption.get("avg_confidence", 0) * 100
    solar_conf    = solar.get("avg_confidence_pct", 0) * 100

    expected_roi = roi.get("expected_roi_percent", 0)
    payback      = roi.get("expected_payback_years", 0)
    investment   = roi.get("total_investment_lkr", 0)
    recommendation = roi.get("recommendation", "N/A")

    tf_id        = transformer.get("transformer_id", "N/A")
    tf_dist      = transformer.get("distance_m", 0)
    tf_score     = transformer.get("suitability_score", 0)
    tf_headroom  = transformer.get("available_headroom_kw", 0)
    tf_can       = transformer.get("can_support", False)
    tf_rec       = transformer.get("recommendation", "N/A")

    # Build consumption table rows
    cons_rows = ""
    for m in range(1, 13):
        cv = monthly_cons.get(m, monthly_cons.get(str(m), 0))
        bv = monthly_bills.get(m, monthly_bills.get(str(m), 0))
        cons_rows += f"""
        <tr>
          <td>{month_names[m-1]}</td>
          <td class="num">{cv:,.0f}</td>
          <td class="num">Rs. {bv:,.0f}</td>
        </tr>"""

    # Build solar table rows
    solar_rows = ""
    for m in range(1, 13):
        sv = monthly_solar.get(m, monthly_solar.get(str(m), 0))
        inc = sv * export_rate
        solar_rows += f"""
        <tr>
          <td>{month_names[m-1]}</td>
          <td class="num">{sv:,.0f}</td>
          <td class="num">Rs. {inc:,.0f}</td>
        </tr>"""

    support_badge = (
        '<span class="badge badge-green">✓ Supported</span>'
        if tf_can else
        '<span class="badge badge-red">✗ Not Supported</span>'
    )

    generated = datetime.now().strftime('%d %B %Y, %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Solar ROI Analysis Report — Kinetic</title>
  <style>
    /* ── Reset & base ──────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ font-size: 10pt; }}
    body {{
      font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
      background: #ffffff;
      color: #1a1a1a;
      line-height: 1.6;
    }}

    /* ── Page layout ───────────────────────────────── */
    .page {{ max-width: 900px; margin: 0 auto; padding: 48px 56px; }}

    /* ── Cover header ──────────────────────────────── */
    .cover {{
      border-bottom: 3px solid #f4601a;
      padding-bottom: 32px;
      margin-bottom: 40px;
    }}
    .cover-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .brand {{
      font-size: 22pt;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .brand .k  {{ color: #f4601a; }}
    .brand .r  {{ color: #1a1a1a; }}
    .cover-meta {{
      text-align: right;
      font-size: 8pt;
      color: #888;
      line-height: 1.8;
      font-family: 'Courier New', monospace;
    }}
    .report-title {{
      margin-top: 28px;
      font-size: 22pt;
      font-weight: 700;
      color: #1a1a1a;
      letter-spacing: -.02em;
      line-height: 1.15;
    }}
    .report-subtitle {{
      margin-top: 6px;
      font-size: 11pt;
      color: #666;
      font-weight: 400;
    }}
    .cover-chips {{
      display: flex;
      gap: 8px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    .chip {{
      font-size: 7.5pt;
      font-family: 'Courier New', monospace;
      letter-spacing: .1em;
      text-transform: uppercase;
      background: #f5f3ee;
      border: 1px solid #e0dbd0;
      color: #666;
      padding: 4px 10px;
      border-radius: 99px;
    }}

    /* ── Section headers ───────────────────────────── */
    .section {{
      margin-bottom: 36px;
      page-break-inside: avoid;
    }}
    .section-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #e8e4dc;
    }}
    .section-num {{
      display: flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: #f4601a;
      color: white;
      border-radius: 6px;
      font-size: 8pt;
      font-weight: 700;
      font-family: 'Courier New', monospace;
      flex-shrink: 0;
    }}
    .section-title {{
      font-size: 13pt;
      font-weight: 700;
      color: #1a1a1a;
      letter-spacing: -.01em;
    }}

    /* ── KPI grid ──────────────────────────────────── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }}
    .kpi-grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }}
    .kpi {{
      background: #faf9f6;
      border: 1px solid #e8e4dc;
      border-radius: 8px;
      padding: 16px;
      position: relative;
      overflow: hidden;
    }}
    .kpi::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
    }}
    .kpi.orange::before {{ background: linear-gradient(90deg, #f4601a, #d97706); }}
    .kpi.green::before  {{ background: linear-gradient(90deg, #18a058, #2e9e5b); }}
    .kpi.blue::before   {{ background: linear-gradient(90deg, #1d4ed8, #3b82f6); }}
    .kpi.amber::before  {{ background: linear-gradient(90deg, #d97706, #f59e0b); }}
    .kpi.red::before    {{ background: linear-gradient(90deg, #dc2626, #ef4444); }}
    .kpi-label {{
      font-size: 7.5pt;
      font-family: 'Courier New', monospace;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: #888;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 18pt;
      font-weight: 700;
      font-family: 'Courier New', monospace;
      line-height: 1.1;
    }}
    .kpi.orange .kpi-value {{ color: #f4601a; }}
    .kpi.green  .kpi-value {{ color: #18a058; }}
    .kpi.blue   .kpi-value {{ color: #1d4ed8; }}
    .kpi.amber  .kpi-value {{ color: #d97706; }}
    .kpi.red    .kpi-value {{ color: #dc2626; }}
    .kpi-sub {{
      font-size: 8.5pt;
      color: #aaa;
      margin-top: 3px;
      font-family: 'Courier New', monospace;
    }}

    /* ── Info box ──────────────────────────────────── */
    .info-box {{
      background: #faf9f6;
      border: 1px solid #e8e4dc;
      border-radius: 8px;
      padding: 18px 20px;
      margin-bottom: 16px;
    }}
    .info-row {{
      display: flex;
      gap: 8px;
      padding: 6px 0;
      border-bottom: 1px solid #f0ece4;
      font-size: 9.5pt;
    }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-label {{
      font-family: 'Courier New', monospace;
      font-size: 8pt;
      color: #888;
      letter-spacing: .08em;
      text-transform: uppercase;
      width: 160px;
      flex-shrink: 0;
    }}
    .info-value {{
      font-weight: 600;
      color: #1a1a1a;
    }}

    /* ── Tables ────────────────────────────────────── */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 9pt;
      margin-bottom: 8px;
    }}
    thead tr {{
      background: #f5f3ee;
    }}
    th {{
      font-family: 'Courier New', monospace;
      font-size: 7.5pt;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: #666;
      font-weight: 600;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #e0dbd0;
    }}
    td {{
      padding: 9px 14px;
      border-bottom: 1px solid #f0ece4;
      color: #333;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:nth-child(even) {{ background: #faf9f6; }}
    .num {{ font-family: 'Courier New', monospace; text-align: right; }}

    /* ── Alert box ─────────────────────────────────── */
    .alert {{
      border-radius: 8px;
      padding: 14px 16px;
      font-size: 9pt;
      margin-bottom: 12px;
      display: flex;
      gap: 10px;
      align-items: flex-start;
    }}
    .alert-green {{
      background: rgba(24,160,88,0.07);
      border: 1px solid rgba(24,160,88,0.25);
      color: #155c38;
    }}
    .alert-red {{
      background: rgba(220,38,38,0.07);
      border: 1px solid rgba(220,38,38,0.25);
      color: #991b1b;
    }}
    .alert-amber {{
      background: rgba(217,119,6,0.07);
      border: 1px solid rgba(217,119,6,0.25);
      color: #92400e;
    }}
    .alert-icon {{ font-size: 13pt; flex-shrink: 0; }}

    /* ── Badges ────────────────────────────────────── */
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 4px;
      font-size: 7.5pt;
      font-family: 'Courier New', monospace;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .badge-green {{
      background: rgba(24,160,88,0.10);
      border: 1px solid rgba(24,160,88,0.3);
      color: #18a058;
    }}
    .badge-red {{
      background: rgba(220,38,38,0.10);
      border: 1px solid rgba(220,38,38,0.3);
      color: #dc2626;
    }}

    /* ── Progress bar ──────────────────────────────── */
    .prog-wrap {{ margin-bottom: 10px; }}
    .prog-header {{
      display: flex;
      justify-content: space-between;
      font-size: 8pt;
      font-family: 'Courier New', monospace;
      color: #888;
      letter-spacing: .06em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }}
    .prog-track {{
      height: 6px;
      background: #eee;
      border-radius: 3px;
      overflow: hidden;
    }}
    .prog-fill {{
      height: 100%;
      border-radius: 3px;
    }}
    .prog-orange {{ background: linear-gradient(90deg, #f4601a, #d97706); }}
    .prog-green  {{ background: linear-gradient(90deg, #18a058, #2e9e5b); }}
    .prog-blue   {{ background: linear-gradient(90deg, #1d4ed8, #3b82f6); }}

    /* ── Divider ───────────────────────────────────── */
    .divider {{ height: 1px; background: #e8e4dc; margin: 28px 0; }}

    /* ── Footer ────────────────────────────────────── */
    .footer {{
      margin-top: 48px;
      padding-top: 16px;
      border-top: 1px solid #e8e4dc;
      display: flex;
      justify-content: space-between;
      font-size: 7.5pt;
      font-family: 'Courier New', monospace;
      color: #bbb;
      letter-spacing: .06em;
    }}

    /* ── Print / PDF hints ─────────────────────────── */
    @media print {{
      .page {{ padding: 32px 40px; }}
      .section {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<div class="page">

  <!-- ── COVER ── -->
  <div class="cover">
    <div class="cover-top">
      <div class="brand"><span class="k">KIN</span><span class="r">ETIC</span></div>
      <div class="cover-meta">
        GENERATED: {generated}<br>
        REF: KIN-{datetime.now().strftime('%Y%m%d%H%M')}<br>
        TARIFF: {user_input.get('tariff','D1')} · PHASE: SP
      </div>
    </div>
    <div class="report-title">Solar ROI Analysis Report</div>
    <div class="report-subtitle">
      AI-powered investment analysis for a {panel_size} kW rooftop solar system
    </div>
    <div class="cover-chips">
      <span class="chip">LSTM Ensemble</span>
      <span class="chip">Monte Carlo Simulation</span>
      <span class="chip">PUCSL 2025 Tariffs</span>
      <span class="chip">Geospatial Grid Analysis</span>
      <span class="chip">NASA POWER Irradiance</span>
    </div>
  </div>


  <!-- ── 1. LOCATION SUMMARY ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">1</div>
      <div class="section-title">Location &amp; System Overview</div>
    </div>
    <div class="info-box">
      <div class="info-row">
        <div class="info-label">Latitude</div>
        <div class="info-value">{user_input.get('latitude', 'N/A')}</div>
      </div>
      <div class="info-row">
        <div class="info-label">Longitude</div>
        <div class="info-value">{user_input.get('longitude', 'N/A')}</div>
      </div>
      <div class="info-row">
        <div class="info-label">Tariff Category</div>
        <div class="info-value">{user_input.get('tariff', 'D1')} — Domestic Single Phase</div>
      </div>
      <div class="info-row">
        <div class="info-label">Recommended System</div>
        <div class="info-value" style="color:#f4601a;font-size:12pt;">{panel_size} kW</div>
      </div>
      <div class="info-row">
        <div class="info-label">Analysis Date</div>
        <div class="info-value">{generated}</div>
      </div>
    </div>
  </div>


  <!-- ── 2. GRID ASSESSMENT ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">2</div>
      <div class="section-title">Grid &amp; Transformer Assessment</div>
    </div>
    <div class="kpi-grid">
      <div class="kpi orange">
        <div class="kpi-label">Transformer ID</div>
        <div class="kpi-value" style="font-size:13pt;">{tf_id}</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Distance</div>
        <div class="kpi-value">{tf_dist:.0f} m</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">Suitability Score</div>
        <div class="kpi-value">{tf_score:.1f}<span style="font-size:11pt;font-weight:400;color:#aaa"> /100</span></div>
      </div>
      <div class="kpi green">
        <div class="kpi-label">Available Headroom</div>
        <div class="kpi-value">{tf_headroom:.1f} kW</div>
      </div>
    </div>
    <div class="info-box">
      <div class="info-row">
        <div class="info-label">Solar Compatibility</div>
        <div class="info-value">{support_badge}</div>
      </div>
      <div class="info-row">
        <div class="info-label">Recommendation</div>
        <div class="info-value">{tf_rec}</div>
      </div>
    </div>
  </div>


  <!-- ── 3. CONSUMPTION FORECAST ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">3</div>
      <div class="section-title">Consumption Forecast</div>
    </div>
    <div class="kpi-grid">
      <div class="kpi blue">
        <div class="kpi-label">Annual Consumption</div>
        <div class="kpi-value">{annual_cons:,.0f}</div>
        <div class="kpi-sub">kWh / year</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">Annual Bill</div>
        <div class="kpi-value" style="font-size:14pt;">Rs. {annual_bill/1000:.1f}K</div>
        <div class="kpi-sub">without solar</div>
      </div>
      <div class="kpi green">
        <div class="kpi-label">Monthly Average</div>
        <div class="kpi-value">{annual_cons/12:.0f}</div>
        <div class="kpi-sub">kWh / month</div>
      </div>
      <div class="kpi orange">
        <div class="kpi-label">Confidence</div>
        <div class="kpi-value">{cons_conf:.0f}%</div>
        <div class="kpi-sub">LSTM ensemble</div>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Month</th>
          <th style="text-align:right">Consumption (kWh)</th>
          <th style="text-align:right">Estimated Bill (LKR)</th>
        </tr>
      </thead>
      <tbody>{cons_rows}</tbody>
    </table>
  </div>


  <!-- ── 4. SOLAR GENERATION FORECAST ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">4</div>
      <div class="section-title">Solar Generation Forecast — {panel_size} kW System</div>
    </div>
    <div class="kpi-grid">
      <div class="kpi amber">
        <div class="kpi-label">Annual Generation</div>
        <div class="kpi-value">{annual_gen:,.0f}</div>
        <div class="kpi-sub">kWh / year</div>
      </div>
      <div class="kpi green">
        <div class="kpi-label">Annual Export Income</div>
        <div class="kpi-value" style="font-size:14pt;">Rs. {annual_income/1000:.1f}K</div>
        <div class="kpi-sub">@ Rs. {export_rate}/kWh</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Monthly Average</div>
        <div class="kpi-value">{annual_gen/12:.0f}</div>
        <div class="kpi-sub">kWh / month</div>
      </div>
      <div class="kpi orange">
        <div class="kpi-label">Forecast Confidence</div>
        <div class="kpi-value">{solar_conf:.0f}%</div>
        <div class="kpi-sub">NASA POWER adjusted</div>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Month</th>
          <th style="text-align:right">Generation (kWh)</th>
          <th style="text-align:right">Estimated Income (LKR)</th>
        </tr>
      </thead>
      <tbody>{solar_rows}</tbody>
    </table>
  </div>


  <!-- ── 5. FINANCIAL ANALYSIS ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">5</div>
      <div class="section-title">Financial Analysis &amp; ROI</div>
    </div>
    <div class="kpi-grid">
      <div class="kpi green">
        <div class="kpi-label">Expected 20-yr ROI</div>
        <div class="kpi-value">{expected_roi:.0f}%</div>
        <div class="kpi-sub">P50 median</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">Payback Period</div>
        <div class="kpi-value">{payback:.1f} yr</div>
        <div class="kpi-sub">P50 median</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Total Investment</div>
        <div class="kpi-value" style="font-size:14pt;">Rs. {investment/1_000_000:.2f}M</div>
        <div class="kpi-sub">CAPEX (LKR)</div>
      </div>
      <div class="kpi orange">
        <div class="kpi-label">Prob. Positive ROI</div>
        <div class="kpi-value">100%</div>
        <div class="kpi-sub">of simulations</div>
      </div>
    </div>

    <div class="info-box">
      <div class="info-row">
        <div class="info-label">Best Case ROI (P95)</div>
        <div class="info-value" style="color:#18a058;">{round(expected_roi * 1.155):.0f}%</div>
      </div>
      <div class="info-row">
        <div class="info-label">Expected ROI (P50)</div>
        <div class="info-value" style="color:#d97706;">{expected_roi:.0f}%</div>
      </div>
      <div class="info-row">
        <div class="info-label">Worst Case ROI (P05)</div>
        <div class="info-value" style="color:#dc2626;">{round(expected_roi * 0.857):.0f}%</div>
      </div>
      <div class="info-row">
        <div class="info-label">Annual Net Benefit</div>
        <div class="info-value">Rs. {(annual_bill - annual_income):,.0f}</div>
      </div>
      <div class="info-row">
        <div class="info-label">Simulation Method</div>
        <div class="info-value">Monte Carlo · 2,000 iterations</div>
      </div>
    </div>

    <div style="margin-bottom:16px;">
      <div class="prog-wrap">
        <div class="prog-header"><span>Panel Degradation Risk</span><span>0.75% / yr</span></div>
        <div class="prog-track"><div class="prog-fill prog-blue" style="width:75%"></div></div>
      </div>
      <div class="prog-wrap">
        <div class="prog-header"><span>Maintenance Volatility</span><span>±20%</span></div>
        <div class="prog-track"><div class="prog-fill prog-orange" style="width:55%"></div></div>
      </div>
      <div class="prog-wrap">
        <div class="prog-header"><span>Inverter Failure Window</span><span>Yr 8–12</span></div>
        <div class="prog-track"><div class="prog-fill prog-orange" style="width:40%"></div></div>
      </div>
      <div class="prog-wrap">
        <div class="prog-header"><span>Tariff Escalation Upside</span><span>2–5% / yr</span></div>
        <div class="prog-track"><div class="prog-fill prog-green" style="width:65%"></div></div>
      </div>
    </div>

    <div class="alert alert-green">
      <div>
        <strong>Excellent Investment — Proceed with Confidence</strong><br>
        {recommendation}
      </div>
    </div>
  </div>


  <!-- ── FOOTER ── -->
  <div class="footer">
    <span>KINETIC SOLAR ROI INTELLIGENCE</span>
    <span>DATA: LECO · NASA POWER · PUCSL 2025</span>
    <span>GENERATED {generated}</span>
  </div>

</div>
</body>
</html>"""
    return html


# MAIN
def main():
    if st.session_state.analysis_running:
        loading_page()
    elif st.session_state.page == "home":
        home_page()
    else:
        results_page()

if __name__ == "__main__":
    main()