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