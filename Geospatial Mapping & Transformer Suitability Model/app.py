"""
streamlit_app.py — Kinetic · Solar Transformer Suitability
Fixed: GPS session state ordering, number_input value seeding,
       query-param sync, consistent backend wiring,
       and removed orphaned HTML div wrappers that caused blank rectangles.
Run: streamlit run streamlit_app.py
"""

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
    page_title="Kinetic · Solar Transformer Suitability",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise FIRST, before anything else runs
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    """Initialise session state defaults. Called once at startup."""
    if "page"       not in st.session_state: st.session_state.page       = "home"
    if "results"    not in st.session_state: st.session_state.results    = None
    if "selected"   not in st.session_state: st.session_state.selected   = None
    if "error"      not in st.session_state: st.session_state.error      = None
    if "solar_kw"   not in st.session_state: st.session_state.solar_kw   = 5.0
    if "radius_m"   not in st.session_state: st.session_state.radius_m   = 5000
    if "lat"        not in st.session_state: st.session_state.lat        = 6.849000
    if "lon"        not in st.session_state: st.session_state.lon        = 79.924700
    if "gps_active" not in st.session_state: st.session_state.gps_active = False


init_session()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

:root {
  --bg:      #f7f5f0;
  --surface: #ffffff;
  --surface2:#faf9f6;
  --surface3:#f2efe8;
  --border:  #e0dbd0;
  --text:    #18160f;
  --text2:   #3d3929;
  --muted:   #7a7264;
  --muted2:  #a09880;
  --orange:  #f4601a;
  --amber:   #d97706;
  --green:   #18a058;
  --dkgreen: #2e6f40;
  --red:     #dc2626;
  --mono:   'Space Mono', monospace;
  --display:'Syne', sans-serif;
  --sans:   'DM Sans', sans-serif;
  --sh-sm: 0 2px 8px rgba(20,16,5,.07);
  --sh-md: 0 6px 24px rgba(20,16,5,.09),0 2px 8px rgba(20,16,5,.06);
  --sh-lg: 0 16px 48px rgba(20,16,5,.12),0 4px 16px rgba(20,16,5,.08);
  --sh-xl: 0 32px 80px rgba(20,16,5,.14),0 8px 24px rgba(20,16,5,.09);
  --sh-or: 0 8px 32px rgba(244,96,26,.22),0 2px 8px rgba(244,96,26,.14);
  --sh-gr: 0 8px 32px rgba(24,160,88,.22),0 2px 8px rgba(24,160,88,.14);
}

/* ── SPACING TOKENS ── */
:root {
  --page-x:  32px;   /* horizontal page gutter */
  --page-top:20px;   /* top breathing room below topbar */
  --gap-sec: 20px;   /* gap between major sections */
  --gap-card:10px;   /* gap between cards in a list */
  --col-gap:  24px;  /* gap between left/right columns */
}

html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
#MainMenu,header,footer{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="stHorizontalBlock"]{gap:var(--col-gap)!important;}
[data-testid="column"]{padding:0!important;}
.stApp>div{position:relative;z-index:1;}
.stApp::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 900px 600px at 10% 0%,rgba(244,96,26,.05) 0%,transparent 70%),
    radial-gradient(ellipse 700px 500px at 90% 100%,rgba(24,160,88,.04) 0%,transparent 70%);
}
input[type=number]{-moz-appearance:textfield;}
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;}

/* ── TOPBAR ── */
.k-topbar{
  position:sticky;top:0;z-index:200;
  background:rgba(247,245,240,.94);
  backdrop-filter:blur(20px) saturate(160%);
  border-bottom:1px solid var(--border);
  box-shadow:0 1px 0 var(--border),0 4px 20px rgba(20,16,5,.05);
  padding:0 var(--page-x);height:56px;
  display:flex;align-items:center;gap:14px;
}
.k-logo{font-family:var(--display);font-size:17px;font-weight:800;letter-spacing:.12em;display:flex;align-items:center;}
.k-logo .kin{color:var(--orange);}
.k-logo .etic{color:var(--text2);}
.k-logo .dot{
  display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:50%;margin-left:5px;
  background:linear-gradient(135deg,var(--orange),var(--amber));
  box-shadow:var(--sh-or);
  animation:pulse-dot 3s ease-in-out infinite;
}
.k-logo .dot::after{content:'';width:6px;height:6px;background:#fff;border-radius:50%;}
@keyframes pulse-dot{
  0%,100%{box-shadow:0 0 0 0 rgba(244,96,26,.5),var(--sh-or);}
  50%{box-shadow:0 0 0 6px rgba(244,96,26,0),var(--sh-or);}
}
.k-tag{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);border:1px solid var(--border);background:var(--surface2);padding:4px 10px;border-radius:99px;font-family:var(--mono);}
.k-live{margin-left:auto;display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px;color:var(--muted);}
.k-live-dot{width:7px;height:7px;border-radius:50%;background:var(--dkgreen);animation:pulse-live 2.4s ease infinite;}
@keyframes pulse-live{0%{box-shadow:0 0 0 0 rgba(24,160,88,.6);}70%{box-shadow:0 0 0 8px rgba(24,160,88,0);}100%{box-shadow:0 0 0 0 rgba(24,160,88,0);}}

/* ── HERO ── */
.k-eyebrow{font-family:var(--mono);font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--orange);margin-bottom:20px;display:flex;align-items:center;gap:14px;}
.k-eyebrow::before{content:'';width:28px;height:1.5px;background:linear-gradient(90deg,var(--orange),var(--amber));border-radius:2px;}
.k-title{font-family:var(--display);font-size:clamp(36px,3.8vw,56px);font-weight:700;line-height:1.0;letter-spacing:-.03em;margin-bottom:24px;}
.k-title .lt{font-weight:400;color:var(--text2);}
.k-title .ac{color:var(--orange);}
.k-desc{color:var(--muted);font-size:14.5px;line-height:1.75;font-weight:300;max-width:460px;}
.k-pills{display:flex;gap:8px;margin-top:36px;flex-wrap:wrap;}
.k-pill{display:flex;align-items:center;gap:6px;padding:6px 12px;border-radius:99px;font-size:10.5px;font-family:var(--mono);border:1px solid;}
.k-pill.or{background:rgba(244,96,26,.08);border-color:rgba(244,96,26,.28);color:var(--orange);}
.k-pill.am{background:rgba(217,119,6,.08);border-color:rgba(217,119,6,.28);color:var(--amber);}
.k-pill.gr{background:rgba(24,160,88,.08);border-color:rgba(24,160,88,.28);color:var(--dkgreen);}
.k-pill.rd{background:rgba(220,38,38,.08);border-color:rgba(220,38,38,.28);color:var(--red);}

/* ── FORM ── */
.k-sec-label{font-family:var(--mono);font-size:9px;letter-spacing:.22em;text-transform:uppercase;color:var(--orange);margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.k-sec-label::before{content:'';width:12px;height:1px;background:currentColor;opacity:.6;}
.k-divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:16px 0;}
.k-error{background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.25);border-radius:6px;padding:10px 14px;font-size:12px;color:#b91c1c;font-family:var(--mono);margin-top:8px;}
.k-gps-active{background:rgba(24,160,88,.1);border:1px solid rgba(24,160,88,.3);border-radius:6px;padding:8px 12px;font-size:11px;font-family:var(--mono);color:#15803d;margin-bottom:10px;}

/* ── PRIMARY BUTTON ── */
.stButton>button[kind="primary"]{
  background:linear-gradient(145deg,var(--orange),#e8530f 50%,var(--amber))!important;
  border:none!important;border-radius:10px!important;
  font-family:var(--display)!important;font-weight:700!important;
  letter-spacing:.1em!important;text-transform:uppercase!important;
  box-shadow:var(--sh-or)!important;padding:14px 0!important;
}

/* ── STAT TILES ── */
.k-stat{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 16px;position:relative;overflow:hidden;box-shadow:var(--sh-md);}
.k-stat::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;border-radius:0 0 14px 14px;}
.k-stat.c0::after{background:var(--orange);}
.k-stat.c1::after{background:var(--amber);}
.k-stat.c2::after{background:var(--dkgreen);}
.k-stat.c3::after{background:#60a5fa;}
.k-stat.c4::after{background:var(--red);}
.k-stat-val{font-family:var(--mono);font-size:25px;font-weight:700;line-height:1.1;}
.k-stat-val.gr{color:var(--green);}
.k-stat-val.am{color:var(--amber);}
.k-stat-val.rd{color:var(--red);}
.k-stat-val.bl{color:#3b82f6;}
.k-stat-lbl{font-size:9px;color:var(--muted);margin-top:6px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;}

/* ── MAP ── */
.k-map-outer{border:1px solid var(--border);border-radius:14px;overflow:hidden;}
.k-map-header{background:var(--surface2);padding:12px 18px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border);}
.k-map-title{font-size:11px;font-family:var(--mono);letter-spacing:.14em;color:var(--text2);text-transform:uppercase;}
.k-legend{margin-left:auto;display:flex;gap:12px;align-items:center;flex-wrap:wrap;}
.k-legend-item{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);font-family:var(--mono);}
.k-legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.k-map-body{padding:10px;background:var(--surface2);}

/* ── TF CARDS ── */
.k-tf-card{background:var(--surface);border:1.5px solid var(--border);border-radius:12px;padding:14px 16px;margin-bottom:var(--gap-card);display:grid;grid-template-columns:44px 1fr auto;gap:14px;align-items:center;box-shadow:var(--sh-sm);}
.k-tf-card.active{border-color:rgba(244,96,26,.55);background:rgba(244,96,26,.03);box-shadow:var(--sh-or);}
.k-rank-num{font-family:var(--mono);font-size:20px;font-weight:700;line-height:1;text-align:center;}
.k-rank-lbl{font-size:8px;font-family:var(--mono);color:var(--muted);letter-spacing:.1em;text-transform:uppercase;text-align:center;margin-top:2px;}
.k-tf-code{font-family:var(--display);font-size:14px;font-weight:700;}
.k-tf-cluster{font-size:11px;color:var(--muted);margin:3px 0 8px;}
.k-tf-pills{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px;}
.k-tf-metrics{display:flex;gap:14px;flex-wrap:wrap;}
.k-tf-ml{color:var(--muted2);font-family:var(--mono);font-size:8px;letter-spacing:.1em;text-transform:uppercase;}
.k-tf-mv{font-family:var(--mono);font-weight:700;font-size:12px;color:var(--text2);margin-top:2px;}
.k-badge{display:flex;flex-direction:column;align-items:center;gap:4px;flex-shrink:0;}
.k-ring{width:50px;height:50px;position:relative;}
.k-ring svg{transform:rotate(-90deg);}
.k-ring-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:12px;font-weight:700;}
.k-ring-lbl{font-family:var(--mono);font-size:8.5px;letter-spacing:.1em;text-transform:uppercase;}
.k-dist-pill{background:var(--surface3);border:1px solid var(--border);border-radius:99px;padding:2px 8px;font-family:var(--mono);font-size:9.5px;color:var(--muted);}
.k-curtail{background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.3);border-radius:99px;padding:2px 8px;font-family:var(--mono);font-size:9.5px;color:var(--red);font-weight:700;}

/* ── SECTION TITLE ── */
.k-sec-title{font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:10px;}
.k-sec-title::after{content:'';flex:1;height:1px;background:var(--border);}

/* ── DETAIL PANEL ── */
.k-detail-code{font-family:var(--display);font-size:20px;font-weight:800;margin-bottom:2px;}
.k-detail-cluster{font-size:12px;color:var(--muted);margin-bottom:18px;}
.k-score-big{background:linear-gradient(135deg,rgba(217,119,6,.09),rgba(244,96,26,.05));border:1px solid rgba(217,119,6,.2);border-radius:12px;padding:14px;text-align:center;margin-bottom:12px;}
.k-score-big-val{font-family:var(--mono);font-size:36px;font-weight:700;line-height:1.1;}
.k-score-big-lbl{font-size:9px;color:var(--muted);margin-top:4px;font-family:var(--mono);letter-spacing:.12em;}
.k-sub-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px;}
.k-sub-cell{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:9px;text-align:center;}
.k-sub-lbl{font-size:8px;color:var(--muted);font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;}
.k-sub-val{font-family:var(--mono);font-size:15px;font-weight:700;margin-top:3px;}
.k-m2-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;}
.k-mcell{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:9px 11px;}
.k-mlbl{font-size:8.5px;color:var(--muted);font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;}
.k-mval{font-family:var(--mono);font-size:13px;font-weight:700;}
.k-bar-wrap{margin-bottom:8px;}
.k-bar-hdr{display:flex;justify-content:space-between;font-size:10px;font-family:var(--mono);color:var(--muted);margin-bottom:4px;}
.k-bar-track{height:5px;background:var(--surface3);border-radius:99px;overflow:hidden;}
.k-bar-fill{height:100%;border-radius:99px;}
.k-flag{border-radius:6px;padding:8px 12px;font-size:11.5px;font-family:var(--mono);display:flex;align-items:center;gap:7px;margin-bottom:7px;}
.k-flag.rd{background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.22);color:#b91c1c;}
.k-flag.gr{background:rgba(24,160,88,.08);border:1px solid rgba(24,160,88,.22);color:#15803d;}
.k-cap-box{background:rgba(24,160,88,.08);border:1px solid rgba(24,160,88,.25);border-radius:10px;padding:13px;margin-bottom:9px;}
.k-cap-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px;}
.k-cap-lbl{font-family:var(--mono);font-size:9.5px;color:#16a34a;letter-spacing:.08em;text-transform:uppercase;}
.k-cap-val{font-family:var(--mono);font-size:24px;font-weight:700;color:#18a058;}
.k-cap-sub{border-top:1px solid rgba(24,160,88,.15);padding-top:7px;display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted);}
.k-cap-msg{margin-top:7px;font-size:11.5px;color:var(--text2);line-height:1.5;}
.k-mini-lbl{font-family:var(--mono);font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:12px 0 6px;}
.k-rec-box{background:linear-gradient(135deg,rgba(244,96,26,.07),rgba(244,96,26,.02));border:1px solid rgba(244,96,26,.2);border-radius:9px;padding:13px;font-size:12px;line-height:1.7;color:var(--text2);}
.k-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:180px;gap:12px;}
.k-empty-ic{width:44px;height:44px;border-radius:50%;background:var(--surface3);border:1.5px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:18px;opacity:.5;}
.k-empty-tx{font-size:13px;color:var(--muted);text-align:center;line-height:1.6;}

/* ── INPUT OVERRIDES ── */
div[data-testid="stNumberInput"] input{
  background:var(--surface2)!important;border:1.5px solid var(--border)!important;
  border-radius:7px!important;font-family:var(--mono)!important;
  font-size:13px!important;color:var(--text)!important;
}
div[data-testid="stNumberInput"] input:focus{
  border-color:var(--orange)!important;box-shadow:0 0 0 3px rgba(244,96,26,.12)!important;
}
div[data-testid="stNumberInput"] label{
  font-family:var(--mono)!important;font-size:10px!important;
  color:var(--muted)!important;letter-spacing:.06em!important;text-transform:uppercase!important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH       = "../processed/MASTER_DATASET_RESIDENTIAL_ONLY.csv"
DEFAULT_CAP_KW = 100
SAFETY_MARGIN  = 0.80
CURTAIL_THRESH = 0.75
CAP_TIERS      = [1.5, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
FEATURE_COLS   = [
    'current_load_kW', 'total_solar_capacity', 'utilization_rate',
    'solar_penetration', 'demand_volatility', 'available_headroom', 'export_ratio'
]
CLUSTER_NAMES = {
    0: 'Underutilised — Low Risk',
    1: 'Balanced Load — Medium Risk',
    2: 'High Utilisation — High Risk',
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading transformer data…")
def load_data():
    df        = pd.read_csv(CSV_PATH)
    unique    = df.drop_duplicates(subset=['TRANSFORMER_CODE', 'ACCOUNT_NO'])
    solar_agg = unique.groupby('TRANSFORMER_CODE').agg(
        total_solar_capacity=('INV_CAPACITY', 'sum'),
        solar_connections=('HAS_SOLAR', 'sum'),
    ).reset_index()
    cons_agg  = df.groupby('TRANSFORMER_CODE').agg(
        TRANSFORMER_LAT=('TRANSFORMER_LAT', 'first'),
        TRANSFORMER_LON=('TRANSFORMER_LON', 'first'),
        avg_consumption=('NET_CONSUMPTION_kWh', 'mean'),
        consumption_std=('NET_CONSUMPTION_kWh', 'std'),
        avg_import=('IMPORT_kWh', 'mean'),
        avg_export=('EXPORT_kWh', 'mean'),
        num_customers=('ACCOUNT_NO', 'nunique'),
    ).reset_index()
    agg = cons_agg.merge(solar_agg, on='TRANSFORMER_CODE', how='left')
    agg['total_solar_capacity'] = agg['total_solar_capacity'].fillna(0)
    agg['solar_connections']    = agg['solar_connections'].fillna(0)
    cap = DEFAULT_CAP_KW
    agg['ESTIMATED_CAPACITY_kW'] = cap
    agg['current_load_kW']       = (agg['avg_consumption'] / 720).fillna(0)
    agg['utilization_rate']      = (agg['current_load_kW'] / cap).clip(0, 1)
    agg['available_headroom']    = cap - agg['current_load_kW'] - agg['total_solar_capacity']
    agg['solar_penetration']     = (agg['total_solar_capacity'] / cap).clip(0)
    agg['demand_volatility']     = (agg['consumption_std'] / 720).fillna(0)
    agg['export_ratio']          = (agg['avg_export'] / (agg['avg_import'] + 1)).fillna(0)
    return agg.fillna(0)


@st.cache_resource(show_spinner="Training ML models…")
def train_models():
    df    = load_data()
    X     = df[FEATURE_COLS].fillna(0)
    sc    = StandardScaler()
    Xs    = sc.fit_transform(X)
    y     = (df['utilization_rate'] < df['utilization_rate'].median()).astype(int)
    rf    = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(Xs, y)
    km    = KMeans(n_clusters=3, random_state=42, n_init='auto')
    km.fit(Xs)
    order = np.argsort(km.cluster_centers_[:, 2])
    lmap  = {orig: new for new, orig in enumerate(order)}
    lr    = LinearRegression()
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['current_load_kW'].values)
    return sc, rf, km, lr, lmap


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def score_color(s):
    return '#18a058' if s >= 80 else '#d97706' if s >= 60 else '#f4601a' if s >= 40 else '#dc2626'

def score_label(s):
    return 'IDEAL' if s >= 80 else 'GOOD' if s >= 60 else 'FAIR' if s >= 40 else 'POOR'

def util_color(u):
    return '#18a058' if u <= 70 else '#f4601a' if u <= 85 else '#dc2626'

def get_cap_rec(score, headroom):
    if score < 60:
        return None
    safe = headroom * 0.80
    rec  = None
    for t in CAP_TIERS:
        if t <= safe:
            rec = t
    return {'kw': rec, 'safe_max': round(safe, 1)} if rec else None

def get_rec_text(score, avail):
    if score >= 80: return f"Highly suitable. {avail:.1f} kW available headroom. Proceed with grid connection."
    if score >= 60: return "Conditionally suitable. Review with utility provider before connecting."
    if score >= 40: return "Marginal suitability. Upgrade assessment recommended before connecting."
    return f"Not suitable. Only {avail:.1f} kW headroom available. Select a different transformer."

def ring_html(score):
    col  = score_color(score)
    r    = 19
    c    = 2 * math.pi * r
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

def bar_html(label, value):
    col = util_color(value)
    return (
        f'<div class="k-bar-wrap">'
        f'<div class="k-bar-hdr"><span>{label}</span>'
        f'<span style="color:{col};font-weight:700">{value:.1f}%</span></div>'
        f'<div class="k-bar-track">'
        f'<div class="k-bar-fill" style="width:{min(value, 100):.0f}%;background:{col}"></div>'
        f'</div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# ASSESSMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_assessment(user_lat, user_lon, solar_kw, radius_m):
    df = load_data().copy()
    sc, rf, km, lr, lmap = train_models()

    df['DISTANCE_M'] = df.apply(
        lambda r: geodesic(
            (user_lat, user_lon),
            (r['TRANSFORMER_LAT'], r['TRANSFORMER_LON'])
        ).meters,
        axis=1,
    )
    nearby = df[df['DISTANCE_M'] <= radius_m].copy().reset_index(drop=True)
    if nearby.empty:
        return None

    Xs       = sc.transform(nearby[FEATURE_COLS].fillna(0))
    proba    = rf.predict_proba(Xs)
    ml_sc    = proba[:, 1] * 100 if proba.shape[1] > 1 else np.full(len(nearby), 50.0)
    clusters = np.vectorize(lmap.get)(km.predict(Xs))

    out = []
    for i, row in nearby.iterrows():
        cap   = float(row['ESTIMATED_CAPACITY_kW'])
        load  = float(row['current_load_kW'])
        sol   = float(row['total_solar_capacity'])
        dist  = float(row['DISTANCE_M'])
        tb    = load + sol
        avail = cap - tb
        sh    = cap * SAFETY_MARGIN - tb
        ta    = tb + solar_kw
        ub    = tb / cap
        ua    = ta / cap
        ratio = sh / max(solar_kw, 0.001)
        h = 100 if ratio >= 1.5 else 80 if ratio >= 1.0 else 50 if avail >= solar_kw else 0
        d = max(0.0, 100 * math.exp(-dist / 1000))
        s = 100 if ua <= 0.70 else 75 if ua <= 0.85 else 40 if ua <= 0.95 else 0
        rule    = h * 0.40 + d * 0.30 + s * 0.30
        ml      = float(ml_sc[i])
        blended = rule * 0.55 + ml * 0.45
        out.append({
            'rank': 0,
            'code': str(row['TRANSFORMER_CODE']),
            'lat':  float(row['TRANSFORMER_LAT']),
            'lon':  float(row['TRANSFORMER_LON']),
            'distance':         round(dist, 0),
            'score':            round(blended, 2),
            'ruleBasedScore':   round(rule, 2),
            'mlScore':          round(ml, 2),
            'headroomScore':    round(h, 2),
            'distanceScore':    round(d, 2),
            'stabilityScore':   round(s, 2),
            'suitabilityLabel': score_label(blended),
            'capacity':         cap,
            'currentLoad':      round(load, 2),
            'existingSolar':    round(sol, 2),
            'availableHeadroom':round(avail, 2),
            'safeHeadroom':     round(sh, 2),
            'utilizationBefore':round(ub * 100, 1),
            'utilizationAfter': round(ua * 100, 1),
            'canSupport':       bool(avail >= solar_kw),
            'curtailmentRisk':  bool((ta / cap) > CURTAIL_THRESH),
            'cluster':          CLUSTER_NAMES.get(int(clusters[i]), 'Unknown'),
            'futureLoad12m':    round(float(lr.predict([[12]])[0]), 2),
            'newSolar':         solar_kw,
            'recommendation':   get_rec_text(blended, avail),
            'capacityRecommendation': get_cap_rec(blended, avail),
        })

    out.sort(key=lambda x: x['score'], reverse=True)
    out = out[:5]  # top 5 only
    for i, r in enumerate(out):
        r['rank'] = i + 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────────────────────
def build_map(results, user_lat, user_lon, selected_code=None):
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
    for tf in results:
        col = score_color(tf['score'])
        folium.CircleMarker(
            [tf['lat'], tf['lon']],
            radius=11 if tf['code'] == selected_code else 9,
            color='white', weight=2.5 if tf['code'] == selected_code else 2,
            fill=True, fill_color=col, fill_opacity=0.88,
            popup=folium.Popup(
                f"<div style='font-family:monospace;font-size:12px;line-height:1.6'>"
                f"<b>{tf['code']}</b><br>Score: {tf['score']:.1f}/100<br>"
                f"Dist: {tf['distance']:.0f} m<br>Util after: {tf['utilizationAfter']:.1f}%<br>"
                f"{'✅ Supported' if tf['canSupport'] else '❌ Not supported'}</div>",
                max_width=200,
            ),
        ).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# GPS COMPONENT
# ─────────────────────────────────────────────────────────────────────────────
def gps_component():
    """Show GPS button. On click, get coords and write directly to session_state."""

    col_btn, col_status = st.columns([1, 2])

    with col_btn:
        get_gps = st.button(
            "📍 Use GPS",
            key="gps_btn",
            help="Auto-fill coordinates from your device location",
        )

    if get_gps:
        try:
            from streamlit_js_eval import get_geolocation
            loc = get_geolocation()
            if loc and "coords" in loc:
                coords = loc["coords"]
                st.session_state.lat = round(float(coords["latitude"]), 6)
                st.session_state.lon = round(float(coords["longitude"]), 6)
                st.session_state.gps_active = True
                st.rerun()
            else:
                with col_status:
                    st.warning("Could not get location — allow browser access")
        except ImportError:
            with col_status:
                st.info("Install streamlit-js-eval for GPS: `pip install streamlit-js-eval`")

    if st.session_state.gps_active:
        st.markdown(
            f'<div class="k-gps-active">📍 GPS: {st.session_state.lat:.6f}, '
            f'{st.session_state.lon:.6f}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar(subtitle="Solar Transformer Suitability"):
    st.markdown(
        f'<div class="k-topbar">'
        f'<div class="k-logo">'
        f'<span class="kin">KIN</span><span class="etic">ETIC</span>'
        f'<span class="dot"></span></div>'
        f'<div class="k-tag">{subtitle}</div>'
        f'<div class="k-live"><div class="k-live-dot"></div>LIVE</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    render_topbar()

    col_left, col_right = st.columns([1.3, 1], gap="large")

    # ── LEFT: hero copy ──────────────────────────────────────────────────────
    with col_left:
        st.markdown("""
<div style="padding:48px 20px 48px 32px">
  <div class="k-eyebrow">Geospatial AI Assessment</div>
  <div class="k-title">
    <span class="lt">Transformer</span><br>
    <span class="ac">Suitability</span><br>
    Intelligence
  </div>
  <p class="k-desc">
    AI-powered analysis that maps nearby grid transformers, estimates headroom
    capacity, predicts curtailment risk, and ranks your best solar connection
    options with explainable recommendations.
  </p>
  <div class="k-pills">
    <span class="k-pill or">◈ Geospatial Proximity</span>
    <span class="k-pill am">▦ Headroom Analysis</span>
    <span class="k-pill gr">◉ ML Overload Prediction</span>
    <span class="k-pill rd">⚡ Curtailment Risk</span>
    <span class="k-pill or">✦ Explainable AI</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── RIGHT: form card ─────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div style="padding:48px 48px 40px 0px">', unsafe_allow_html=True)

        # Top colour stripe
        st.markdown("""
<div style="height:3px;border-radius:3px 3px 0 0;
  background:linear-gradient(90deg,#dc2626,#f4601a 30%,#d97706 55%,#2e6f40);
  margin-bottom:-1px;"></div>""", unsafe_allow_html=True)

        with st.container(border=True):

            # ── LOCATION ────────────────────────────────────────────────────
            st.markdown('<div class="k-sec-label">Location</div>', unsafe_allow_html=True)

            # GPS button — writes directly to session_state.lat/lon then reruns
            gps_component()

            lc, rc = st.columns(2)
            lat = lc.number_input(
                "Latitude",
                value=st.session_state.lat,
                min_value=-90.0, max_value=90.0,
                step=0.000100, format="%.6f",
            )
            lon = rc.number_input(
                "Longitude",
                value=st.session_state.lon,
                min_value=-180.0, max_value=180.0,
                step=0.000100, format="%.6f",
            )

            st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

            # ── SOLAR ────────────────────────────────────────────────────────
            st.markdown('<div class="k-sec-label">Solar Installation</div>', unsafe_allow_html=True)

            lc2, rc2 = st.columns(2)
            solar_kw = lc2.number_input(
                "Capacity (kW)",
                value=st.session_state.solar_kw,
                min_value=0.5, max_value=200.0, step=0.5,
            )
            radius_m = rc2.number_input(
                "Search Radius (m)",
                value=float(st.session_state.radius_m),
                min_value=100.0, max_value=20000.0, step=100.0,
            )

            if st.session_state.error:
                st.markdown(
                    f'<p class="k-error">⚠ {st.session_state.error}</p>',
                    unsafe_allow_html=True,
                )

            run = st.button("▶  Run Assessment", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Handle run ───────────────────────────────────────────────────────────
    if run:
        st.session_state.lat      = lat
        st.session_state.lon      = lon
        st.session_state.solar_kw = solar_kw
        st.session_state.radius_m = int(radius_m)

        with st.spinner("Analysing transformers…"):
            results = run_assessment(lat, lon, solar_kw, radius_m)

        if results is None:
            st.session_state.error = (
                f"No transformers found within {radius_m:.0f} m. "
                f"Try increasing the search radius or check your coordinates."
            )
            st.rerun()
        else:
            st.session_state.results  = results
            st.session_state.selected = results[0]
            st.session_state.error    = None
            st.session_state.page     = 'results'
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_results():
    results  = st.session_state.results
    selected = st.session_state.selected

    supported = sum(1 for t in results if t['canSupport'])
    curtail   = sum(1 for t in results if t['curtailmentRisk'])
    top       = results[0]['score']
    top_cls   = 'gr' if top >= 80 else 'am' if top >= 60 else 'rd'

    render_topbar("Assessment Results")

    # ── Back ─────────────────────────────────────────────────────────────────
    st.markdown('<div style="padding:16px 32px 0">', unsafe_allow_html=True)
    if st.button("← Back to Assessment"):
        st.session_state.page     = 'home'
        st.session_state.results  = None
        st.session_state.selected = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Stat tiles — single flex row ──────────────────────────────────────────
    tiles_data = [
        ('c0', str(len(results)),              '',      'Transformers Found'),
        ('c1', f"{top:.1f}",                   top_cls, 'Top Score / 100'),
        ('c2', str(supported),                 'gr',    'Can Support'),
        ('c3', f"{results[0]['newSolar']} kW", 'bl',    'Solar Capacity'),
        ('c4', str(curtail),                   'am',    'Curtailment Risk'),
    ]
    tiles_html = ''.join(
        f'<div class="k-stat {cls}" style="flex:1;min-width:0">'
        f'<div class="k-stat-val {vcls}">{val}</div>'
        f'<div class="k-stat-lbl">{lbl}</div>'
        f'</div>'
        for cls, val, vcls, lbl in tiles_data
    )
    st.markdown(
        f'<div style="display:flex;gap:10px;padding:12px 32px 0;align-items:stretch">'
        f'{tiles_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Map ───────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="padding:20px 32px 0">'
        '<div class="k-map-outer">'
        '<div class="k-map-header">'
        '<span class="k-map-title">Transformer Map</span>'
        '<div class="k-legend">'
        '<div class="k-legend-item"><div class="k-legend-dot" style="background:#18a058"></div>Ideal</div>'
        '<div class="k-legend-item"><div class="k-legend-dot" style="background:#d97706"></div>Good</div>'
        '<div class="k-legend-item"><div class="k-legend-dot" style="background:#f4601a"></div>Fair</div>'
        '<div class="k-legend-item"><div class="k-legend-dot" style="background:#dc2626"></div>Poor</div>'
        '<div class="k-legend-item"><div class="k-legend-dot" style="background:#2563eb"></div>You</div>'
        '</div></div>'
        '<div class="k-map-body">',
        unsafe_allow_html=True,
    )

    sel_code = selected['code'] if selected else None
    m = build_map(results, st.session_state.lat, st.session_state.lon, sel_code)
    st_folium(m, width="100%", height=360, returned_objects=[])

    st.markdown('</div></div></div>', unsafe_allow_html=True)  # k-map-body / k-map-outer / padding
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # ── List + Detail ─────────────────────────────────────────────────────────
    list_col, detail_col = st.columns([1.6, 1], gap="large")

    with list_col:
        st.markdown('<div style="padding:0 8px 40px 32px">', unsafe_allow_html=True)
        st.markdown('<div class="k-sec-title">Ranked Transformers</div>', unsafe_allow_html=True)

        for tf in results:
            col    = score_color(tf['score'])
            uc     = util_color(tf['utilizationAfter'])
            active = 'active' if (selected and selected['code'] == tf['code']) else ''
            c_pill = '<span class="k-curtail">⚡ Curtail</span>' if tf['curtailmentRisk'] else ''
            sc_col = '#2e6f40' if tf['canSupport'] else '#dc2626'
            sc_txt = '✓ OK' if tf['canSupport'] else '✗ NO'

            st.markdown(
                f'<div class="k-tf-card {active}">'
                f'<div><div class="k-rank-num">{tf["rank"]}</div>'
                f'<div class="k-rank-lbl">RANK</div></div>'
                f'<div>'
                f'<div class="k-tf-pills">'
                f'<span class="k-tf-code">{tf["code"]}</span>'
                f'<span class="k-dist-pill">{tf["distance"]:.0f} m</span>'
                f'{c_pill}</div>'
                f'<div class="k-tf-cluster">{tf["cluster"]}</div>'
                f'<div class="k-tf-metrics">'
                f'<div><div class="k-tf-ml">Capacity</div><div class="k-tf-mv">{tf["capacity"]:.0f} kW</div></div>'
                f'<div><div class="k-tf-ml">Load</div><div class="k-tf-mv">{tf["currentLoad"]:.1f} kW</div></div>'
                f'<div><div class="k-tf-ml">Headroom</div><div class="k-tf-mv">{tf["availableHeadroom"]:.1f} kW</div></div>'
                f'<div><div class="k-tf-ml">Util After</div>'
                f'<div class="k-tf-mv" style="color:{uc}">{tf["utilizationAfter"]:.1f}%</div></div>'
                f'</div></div>'
                f'<div class="k-badge">'
                f'{ring_html(tf["score"])}'
                f'<div class="k-ring-lbl" style="color:{col}">{tf["suitabilityLabel"]}</div>'
                f'<div style="font-size:9.5px;font-family:monospace;color:{sc_col};font-weight:700;margin-top:2px">'
                f'{sc_txt}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if st.button("View Details →", key=f"vd_{tf['code']}"):
                st.session_state.selected = tf
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with detail_col:
        st.markdown('<div style="padding:0 32px 40px 8px">', unsafe_allow_html=True)
        st.markdown('<div class="k-sec-title">Details</div>', unsafe_allow_html=True)

        if not selected:
            st.markdown(
                '<div style="background:#fff;border:1.5px solid #e0dbd0;border-radius:16px;'
                'padding:22px;box-shadow:0 32px 80px rgba(20,16,5,.13)">'
                '<div class="k-empty">'
                '<div class="k-empty-ic">◎</div>'
                '<div class="k-empty-tx">Select a transformer<br>to view detailed analysis</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            tf  = selected
            col = score_color(tf['score'])

            if tf['capacityRecommendation']:
                cr = tf['capacityRecommendation']
                cap_html = (
                    f'<div class="k-mini-lbl">Recommended Solar Capacity</div>'
                    f'<div class="k-cap-box">'
                    f'<div class="k-cap-top">'
                    f'<span class="k-cap-lbl">Optimal Size</span>'
                    f'<span class="k-cap-val">{cr["kw"]} kW</span>'
                    f'</div>'
                    f'<div class="k-cap-sub">'
                    f'<span>Safe Maximum</span>'
                    f'<span style="color:#16a34a;font-weight:700">{cr["safe_max"]} kW</span>'
                    f'</div>'
                    f'<div class="k-cap-msg">'
                    f'A {cr["kw"]} kW system is optimal based on available headroom.'
                    f'</div></div>'
                )
            elif tf['score'] >= 60:
                cap_html = '<div class="k-flag rd">⚠ Headroom too limited for a standard capacity tier</div>'
            else:
                cap_html = ''

            support_flag = (
                f'<div class="k-flag gr">✓ &nbsp;Can support {tf["newSolar"]} kW</div>'
                if tf['canSupport'] else
                f'<div class="k-flag rd">✗ &nbsp;Insufficient capacity for {tf["newSolar"]} kW</div>'
            )
            curtail_flag = (
                '<div class="k-flag rd">⚡ &nbsp;Curtailment risk — utilisation &gt; 75%</div>'
                if tf['curtailmentRisk'] else ''
            )

            st.markdown(
                f'<div style="background:#fff;border:1.5px solid #e0dbd0;border-radius:16px;'
                f'padding:22px;box-shadow:0 32px 80px rgba(20,16,5,.13);'
                f'max-height:calc(100vh - 120px);overflow-y:auto;">'

                f'<div class="k-detail-code">{tf["code"]}</div>'
                f'<div class="k-detail-cluster">{tf["cluster"]} · Rank #{tf["rank"]}</div>'

                f'<div class="k-score-big">'
                f'<div class="k-score-big-val" style="color:{col}">{tf["score"]:.1f}</div>'
                f'<div class="k-score-big-lbl">Blended Suitability Score / 100</div>'
                f'</div>'

                f'<div class="k-sub-grid">'
                f'<div class="k-sub-cell"><div class="k-sub-lbl">Headroom</div>'
                f'<div class="k-sub-val" style="color:#f4601a">{tf["headroomScore"]:.0f}</div></div>'
                f'<div class="k-sub-cell"><div class="k-sub-lbl">Distance</div>'
                f'<div class="k-sub-val" style="color:#d97706">{tf["distanceScore"]:.0f}</div></div>'
                f'<div class="k-sub-cell"><div class="k-sub-lbl">Stability</div>'
                f'<div class="k-sub-val" style="color:#2e6f40">{tf["stabilityScore"]:.0f}</div></div>'
                f'</div>'

                f'<div class="k-m2-grid">'
                f'<div class="k-mcell"><div class="k-mlbl">Rule Score</div>'
                f'<div class="k-mval" style="color:#f4601a">{tf["ruleBasedScore"]:.1f}</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">ML Score</div>'
                f'<div class="k-mval" style="color:#d97706">{tf["mlScore"]:.1f}</div></div>'
                f'</div>'

                f'{bar_html("Utilisation Before", tf["utilizationBefore"])}'
                f'{bar_html("Utilisation After",  tf["utilizationAfter"])}'

                f'<div class="k-m2-grid" style="margin-top:10px">'
                f'<div class="k-mcell"><div class="k-mlbl">Distance</div>'
                f'<div class="k-mval">{tf["distance"]:.0f} m</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">Capacity</div>'
                f'<div class="k-mval">{tf["capacity"]:.0f} kW</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">Available</div>'
                f'<div class="k-mval">{tf["availableHeadroom"]:.1f} kW</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">Safe Headroom</div>'
                f'<div class="k-mval">{tf["safeHeadroom"]:.1f} kW</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">Existing Solar</div>'
                f'<div class="k-mval">{tf["existingSolar"]:.1f} kW</div></div>'
                f'<div class="k-mcell"><div class="k-mlbl">Load +12M</div>'
                f'<div class="k-mval">{tf["futureLoad12m"]:.1f} kW</div></div>'
                f'</div>'

                f'{support_flag}'
                f'{curtail_flag}'
                f'{cap_html}'

                f'<div class="k-mini-lbl">Recommendation</div>'
                f'<div class="k-rec-box">{tf["recommendation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if st.session_state.page == 'home':
        page_home()
    else:
        page_results()


if __name__ == '__main__':
    main()

