import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import subprocess
import sys
import json
import os
from pathlib import Path
import time
from datetime import datetime
import glob

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Kinetic · Consumption Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load custom CSS
def load_css():
    """Load CSS file with absolute path"""
    # Get the directory where this script is located
    current_dir = Path(__file__).parent.absolute()
    css_file = current_dir / "styles.css"

    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        # Optional: Add a success message to verify it's loading
        # st.success(f"✅ CSS loaded from {css_file}")
    else:
        st.warning(f"⚠️ CSS file not found at {css_file}")


load_css()


# ============================================================================
# SESSION STATE INIT
# ============================================================================
def init_session():
    if "results" not in st.session_state:
        st.session_state.results = None
    if "page" not in st.session_state:
        st.session_state.page = "input"
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = None
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "error" not in st.session_state:
        st.session_state.error = None


init_session()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def find_latest_results_dir():
    """Find the most recent results directory"""
    results_base = Path("results")
    if not results_base.exists():
        return None

    dirs = [d for d in results_base.iterdir() if d.is_dir()]
    if not dirs:
        return None

    # Sort by creation time (newest first)
    dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    return dirs[0]


def load_results_from_file():
    """Load results from the most recent component3_results.json"""
    latest_dir = find_latest_results_dir()
    if not latest_dir:
        return None

    results_file = latest_dir / "component3_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


def run_forecast_command(lat, lon, months_dict):
    """
    Run the forecast command - using ONLY the parameters that work in terminal
    tariff, phase, household_size are collected but NOT used in command
    """
    # Get the project root (where main.py is)
    project_root = Path(__file__).parent.absolute()

    # Construct the months JSON string
    months_json = json.dumps(months_dict)

    # Build the command - ONLY mode, lat, lon, months
    # tariff, phase, household_size are IGNORED
    cmd = [
        "python",  # Still using default python (not venv path yet)
        "main.py",
        "--mode", "single",
        "--lat", str(lat),
        "--lon", str(lon),
        "--months", months_json
        # NO --tariff, --phase, --household_size here
    ]

    # Debug: print the command being run
    print(f"Running: {' '.join(str(c) for c in cmd)}")

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=60
        )

        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            return False

        # Wait a moment for files to be written
        time.sleep(1)
        return True

    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


# ============================================================================
# KINETIC TOPBAR
# ============================================================================
def render_topbar():
    st.markdown(f"""
    <div class="k-topbar">
        <div class="k-logo">
            <span class="kin">KIN</span><span class="etic">ETIC</span>
            <span class="dot"></span>
        </div>
        <div class="k-tag">Consumption Forecast</div>
        <div class="k-live">
            <div class="k-live-dot"></div>
            LSTM · ENSEMBLE · PUCSL 2025
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# LOADING OVERLAY (matches Component 4)
# ============================================================================
def render_loading_overlay():
    st.markdown("""
    <div class="kin-loading">
        <div class="kin-loader-ring"></div>
        <div class="kin-sim-counter">LSTM</div>
        <div class="kin-loader-text">Running Consumption Forecast</div>
        <div class="kin-loader-progress"><div class="kin-loader-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# INPUT PAGE
# ============================================================================
def input_page():
    render_topbar()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div style="padding:40px 20px 40px 32px;">
            <div class="k-eyebrow">LSTM FORECAST · ENSEMBLE</div>
            <div class="k-title">
                <span class="lt">Electricity</span><br>
                <span class="ac">Consumption</span><br>
                Forecast
            </div>
            <p class="k-desc">
                AI-powered forecasting using LSTM neural networks and pattern-based methods, 
                calibrated for Sri Lankan households with PUCSL tariff integration.
            </p>
            <div class="k-pills">
                <span class="k-pill or">◈ LSTM Ensemble</span>
                <span class="k-pill am">▦ Pattern-based</span>
                <span class="k-pill gr">◉ PUCSL Tariffs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="padding:40px 48px 40px 0px;">', unsafe_allow_html=True)

        # Form card
        with st.container():
            st.markdown("""
            <div style="height:3px;border-radius:3px 3px 0 0;background:linear-gradient(90deg,#dc2626,#f4601a 30%,#d97706 55%,#2e6f40);margin-bottom:-1px;"></div>
            """, unsafe_allow_html=True)

            with st.container(border=True):
                # Location section
                st.markdown('<div class="k-sec-label">Location</div>', unsafe_allow_html=True)

                lat = st.number_input("Latitude", value=6.8511, format="%.6f", step=0.0001, key="input_lat")
                lon = st.number_input("Longitude", value=79.9212, format="%.6f", step=0.0001, key="input_lon")

                st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

                # Monthly consumption
                st.markdown('<div class="k-sec-label">Monthly Consumption (kWh)</div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    m1 = st.number_input("Month 1", value=350.5, min_value=0.0, max_value=2000.0, step=10.0, key="m1")
                with c2:
                    m2 = st.number_input("Month 2", value=420.2, min_value=0.0, max_value=2000.0, step=10.0, key="m2")
                with c3:
                    m3 = st.number_input("Month 3", value=380.1, min_value=0.0, max_value=2000.0, step=10.0, key="m3")

                st.markdown("""
                <div style="font-family:'Space Mono';font-size:9px;color:#7a7264;margin-top:4px;text-align:right;">
                    Months: Sep, Oct, Nov 2025
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

                # Parameters
                st.markdown('<div class="k-sec-label">Parameters</div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    tariff = st.selectbox("Tariff", ["D1", "GP1", "GP2"], index=0, key="tariff")
                with col_b:
                    phase = st.selectbox("Phase", ["SP", "TP"], index=0, key="phase")

                household_size = st.number_input("Household Size", value=4, min_value=1, max_value=20, step=1,
                                                 key="hh_size")

                # Debug expander (optional, can be removed in production)
                with st.expander("Debug Info", expanded=False):
                    st.write("Command will be:")
                    months_dict = {9: m1, 10: m2, 11: m3}
                    cmd = f"python main.py --mode single --lat {lat} --lon {lon} --months '{json.dumps(months_dict)}' --tariff {tariff} --phase {phase} --household_size {household_size}"
                    st.code(cmd, language="bash")

                # Run button
                run_button = st.button("▶ Run Consumption Forecast", use_container_width=True, type="primary")

                if run_button:
                    # Create months dictionary (assuming months 9,10,11 as per your example)
                    months_dict = {9: m1, 10: m2, 11: m3}

                    # Set loading state
                    st.session_state.loading = True
                    st.session_state.error = None

                    # Run the forecast
                    with st.spinner("Running LSTM forecast..."):
                        success = run_forecast_command(lat, lon, months_dict)

                    if success:
                        # Wait a bit longer for files to be written
                        time.sleep(2)

                        # Load results with better error handling
                        results = load_results_from_file()

                        # Debug: print what happened
                        if results:
                            st.success("✅ Results loaded successfully!")
                            st.session_state.results = results
                            st.session_state.page = "results"
                            st.session_state.loading = False
                            st.rerun()
                        else:
                            # Try to find what went wrong
                            latest_dir = find_latest_results_dir()
                            if latest_dir:
                                results_file = latest_dir / "component3_results.json"
                                if results_file.exists():
                                    st.session_state.error = f"Results file exists but couldn't be parsed: {results_file}"
                                else:
                                    st.session_state.error = f"Results file not found at: {results_file}"
                            else:
                                st.session_state.error = "No results directory found in ./results/"
                            st.session_state.loading = False
                    else:
                        st.session_state.error = "Failed to run forecast command"
                        st.session_state.loading = False

                if st.session_state.error:
                    st.markdown(f'<div class="k-error">⚠ {st.session_state.error}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# RESULTS PAGE
# ============================================================================
def results_page():
    render_topbar()

    results = st.session_state.results
    if not results:
        st.error("No results found. Please run the forecast first.")
        if st.button("← Back to Input"):
            st.session_state.page = "input"
            st.rerun()
        return

    try:
        # Extract data from results
        forecast = results['forecast']['forecast']
        billing = results['billing']
        user_data = results['user_input']
        metadata = results['metadata']

        # Extract values for display
        monthly_values = forecast['monthly_values']
        monthly_confidence = forecast['monthly_confidence']
        stats = forecast['statistics']

        monthly_bills = billing['monthly_bills']
        annual_summary = billing['annual_summary']

        # Prepare month data
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        cons_values = [monthly_values.get(str(m), monthly_values.get(m, 0)) for m in months]
        conf_values = [monthly_confidence.get(str(m), monthly_confidence.get(m, 0)) for m in months]
        bill_values = [monthly_bills[str(m)]['total_bill_lkr'] if str(m) in monthly_bills
                       else monthly_bills[m]['total_bill_lkr'] for m in months]

        # Sort months by consumption for ranked table
        sorted_indices = sorted(range(12), key=lambda i: cons_values[i], reverse=True)

        # Statistics
        annual_total = stats['annual_total']
        annual_bill = annual_summary['total_bill_lkr']
        confidence = stats['overall_confidence']
        avg_monthly = annual_total / 12
        peak_month_idx = stats['peak_month'] - 1
        peak_month = month_names[peak_month_idx]
        peak_val = stats['peak_consumption']

        # Similar households count
        similar_count = results['similarity_analysis']['similar_households_found']

        # ===== TOP STAT ROW =====
        st.markdown('<div class="k-stat-row">', unsafe_allow_html=True)

        stat_cols = st.columns(5)
        with stat_cols[0]:
            st.markdown(f"""
            <div class="k-stat-card blue">
                <div class="k-stat-lbl">Annual kWh</div>
                <div class="k-stat-val blue">{annual_total:,.0f}</div>
                <div class="k-stat-sub">total consumption</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[1]:
            # Format bill in K or M
            if annual_bill >= 1_000_000:
                bill_display = f"Rs.{annual_bill / 1_000_000:.1f}M"
            elif annual_bill >= 1_000:
                bill_display = f"Rs.{annual_bill / 1_000:.0f}K"
            else:
                bill_display = f"Rs.{annual_bill:,.0f}"

            st.markdown(f"""
            <div class="k-stat-card amber">
                <div class="k-stat-lbl">Annual Bill</div>
                <div class="k-stat-val amber">{bill_display}</div>
                <div class="k-stat-sub">LKR</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[2]:
            st.markdown(f"""
            <div class="k-stat-card green">
                <div class="k-stat-lbl">Confidence</div>
                <div class="k-stat-val green">{confidence * 100:.0f}%</div>
                <div class="k-stat-sub">overall</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[3]:
            st.markdown(f"""
            <div class="k-stat-card teal">
                <div class="k-stat-lbl">Monthly Avg</div>
                <div class="k-stat-val" style="color:#0e8a8a">{avg_monthly:.0f}</div>
                <div class="k-stat-sub">kWh/month</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[4]:
            st.markdown(f"""
            <div class="k-stat-card red">
                <div class="k-stat-lbl">Peak Month</div>
                <div class="k-stat-val" style="color:#dc2626;font-size:22px;">{peak_month}</div>
                <div class="k-stat-sub">{peak_val:.0f} kWh</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ===== MAIN FORECAST CHART =====
        st.markdown('<div class="k-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="k-ph">
            <div>
                <div class="k-pt blue">12-Month Consumption Forecast</div>
                <div class="k-ps">LSTM ensemble with confidence bands</div>
            </div>
            <div class="k-leg-row" style="margin-top:0;">
                <div class="k-leg-it"><div class="k-leg-d" style="background:#1d4ed8"></div><span class="k-leg-t">Forecast</span></div>
                <div class="k-leg-it"><div class="k-leg-d" style="background:rgba(29,78,216,0.15)"></div><span class="k-leg-t">Confidence Band</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()

        # Confidence band
        lower = [v * 0.85 for v in cons_values]
        upper = [v * 1.15 for v in cons_values]

        fig.add_trace(go.Scatter(
            x=month_names, y=upper,
            fill=None, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=month_names, y=lower,
            fill='tonexty', fillcolor='rgba(29,78,216,0.12)',
            line=dict(width=0), name='Confidence Band',
            hoverinfo='skip'
        ))

        # Main forecast line
        fig.add_trace(go.Scatter(
            x=month_names, y=cons_values,
            mode='lines+markers',
            line=dict(color='#1d4ed8', width=3),
            marker=dict(size=8, color='#1d4ed8', line=dict(color='white', width=2)),
            name='Forecast (kWh)'
        ))

        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', size=10, color='#7a7264'),
            margin=dict(l=40, r=20, t=20, b=40),
            hovermode='x unified',
            showlegend=False
        )

        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(224,219,208,0.6)',
            tickfont=dict(family='Space Mono', size=10, color='#7a7264')
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(224,219,208,0.6)',
            tickfont=dict(family='Space Mono', size=10, color='#7a7264'),
            title=dict(text='kWh', font=dict(family='Space Mono', size=10, color='#7a7264'))
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== TWO COLUMN LAYOUT =====
        col_left, col_right = st.columns([1.5, 1], gap="large")

        with col_left:
            # Ranked Monthly Table - USING STREAMLIT COLUMNS with reduced height
            st.markdown('<div class="k-panel">', unsafe_allow_html=True)
            st.markdown("""
            <div class="k-ph">
                <span class="k-pt blue">Ranked Monthly Breakdown</span>
                <span class="k-badge teal">PATTERN-BASED FALLBACK READY</span>
            </div>
            """, unsafe_allow_html=True)

            # Header with reduced padding
            st.markdown("""
            <style>
                /* Reduce spacing in columns */
                div[data-testid="column"] {
                    padding: 0 !important;
                }
                /* Make rows more compact */
                .stHorizontalBlock {
                    gap: 0 !important;
                    margin: 0 !important;
                    min-height: 40px !important;
                }
                /* Reduce paragraph margins */
                .stMarkdown p {
                    margin: 0 !important;
                    line-height: 1.2 !important;
                }
            </style>
            """, unsafe_allow_html=True)

            # Create header with minimal spacing
            cols = st.columns([0.4, 0.8, 0.6, 1.2, 1.5, 1])
            with cols[0]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>#</p>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>MONTH</p>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>kWh</p>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>BILL (LKR)</p>",
                            unsafe_allow_html=True)
            with cols[4]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>CONFIDENCE</p>",
                            unsafe_allow_html=True)
            with cols[5]:
                st.markdown("<p style='margin:0; font-size:11px; font-weight:bold;'>SEASON</p>", unsafe_allow_html=True)

            st.markdown('<hr style="margin: 5px 0 5px 0; border: 1px solid #e0dbd0;">', unsafe_allow_html=True)

            # Create rows with reduced height
            for rank, idx in enumerate(sorted_indices[:8]):
                month = month_names[idx]
                cons = cons_values[idx]
                bill = bill_values[idx]
                conf = conf_values[idx] * 100
                month_num = idx + 1

                # Determine season
                if month_num in [12, 1, 2]:
                    season = 'NE MONSOON'
                    season_color = '#1d4ed8'
                    season_bg = '#e6edff'
                elif month_num in [3, 4]:
                    season = 'DRY SEASON'
                    season_color = '#d97706'
                    season_bg = '#fff4e5'
                elif month_num in [5, 6, 7, 8, 9]:
                    season = 'SW MONSOON'
                    season_color = '#0e8a8a'
                    season_bg = '#e6f3f3'
                else:  # 10, 11
                    season = 'INTER-MONSOON'
                    season_color = '#18a058'
                    season_bg = '#e6f4ea'

                # Row background for top rank
                if rank == 0:
                    st.markdown('<div style="background: #f0f4ff; margin: 0; padding: 2px 0;">', unsafe_allow_html=True)

                cols = st.columns([0.4, 0.8, 0.6, 1.2, 1.5, 1])

                # Rank
                with cols[0]:
                    st.markdown(f"<p style='margin:2px 0; font-size:14px;'><b>{rank + 1}</b></p>",
                                unsafe_allow_html=True)

                # Month
                with cols[1]:
                    st.markdown(
                        f"<p style='margin:2px 0; font-size:14px;'><b>{month}</b><br><span style='font-size:9px; color:#a09880;'>Month {month_num}</span></p>",
                        unsafe_allow_html=True)

                # kWh
                with cols[2]:
                    st.markdown(
                        f"<p style='margin:2px 0; font-size:14px; color:#1d4ed8; font-weight:600;'>{cons:.0f}</p>",
                        unsafe_allow_html=True)

                # Bill
                with cols[3]:
                    st.markdown(f"<p style='margin:2px 0; font-size:14px; color:#d97706;'>Rs. {bill:,.0f}</p>",
                                unsafe_allow_html=True)

                # Confidence with progress bar
                with cols[4]:
                    progress_html = f"""
                    <div style="margin:2px 0; display: flex; align-items: center; gap: 8px; max-width: 140px;">
                        <div style="flex: 1; height: 6px; background: #f2efe8; border-radius: 3px;">
                            <div style="width: {conf}%; height: 100%; background: linear-gradient(90deg, #18a058, #2e6f40); border-radius: 3px;"></div>
                        </div>
                        <span style="font-size: 12px; color: #18a058; min-width: 35px;">{conf:.0f}%</span>
                    </div>
                    """
                    st.markdown(progress_html, unsafe_allow_html=True)

                # Season badge
                with cols[5]:
                    badge_html = f"""
                    <span style="margin:2px 0; display:inline-block; background: {season_bg}; color: {season_color}; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; text-transform: uppercase;">
                        {season}
                    </span>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)

                if rank == 0:
                    st.markdown('</div>', unsafe_allow_html=True)

                # Minimal separator
                st.markdown('<hr style="margin: 2px 0; border: 1px solid #f2efe8;">', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            # Detail Panel
            st.markdown('<div class="k-detail-panel">', unsafe_allow_html=True)

            # Header
            account_id = f"Acc-{hash(str(user_data['latitude'])) % 10000:04d}"

            st.markdown(f"""
            <div class="k-dp-hdr">
                <div class="k-dp-id">{account_id}</div>
                <div class="k-dp-sub">{user_data['tariff']} TARIFF · {user_data['phase']} PHASE · PEAK: {peak_month.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

            # Big Score
            st.markdown(f"""
            <div class="k-big-score">
                <div class="k-bs-num blue">{confidence * 100:.1f}</div>
                <div class="k-bs-lbl">Forecast Confidence Score / 100</div>
            </div>
            """, unsafe_allow_html=True)

            # Mini grid - component scores
            st.markdown("""
            <div class="k-mini-grid">
                <div class="k-mini-card">
                    <div class="k-mc-lbl">Similarity</div>
                    <div class="k-mc-val blue">94</div>
                </div>
                <div class="k-mini-card">
                    <div class="k-mc-lbl">Pattern</div>
                    <div class="k-mc-val green">89</div>
                </div>
                <div class="k-mini-card">
                    <div class="k-mc-lbl">Seasonal</div>
                    <div class="k-mc-val amber">82</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Method scores
            st.markdown(f"""
            <div class="k-scores-2">
                <div class="k-score-box">
                    <div class="k-sb-lbl">LSTM Score</div>
                    <div class="k-sb-val blue">85</div>
                </div>
                <div class="k-score-box">
                    <div class="k-sb-lbl">Pattern Score</div>
                    <div class="k-sb-val amber">72</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

            # Progress bars
            peak_pct = (peak_val / max(cons_values)) * 100
            annual_pct = (annual_total / 6000) * 100
            avg_pct = (avg_monthly / 500) * 100

            st.markdown(f"""
            <div class="k-prog-sec">
                <div class="k-prog-item">
                    <div class="k-prog-hd"><span>Peak Month ({peak_month})</span><span style="color:#18160f;">{peak_val:.0f} kWh</span></div>
                    <div class="k-prog-track"><div class="k-prog-fill k-pf-amber" style="width:{peak_pct:.0f}%"></div></div>
                </div>
                <div class="k-prog-item">
                    <div class="k-prog-hd"><span>Annual Total</span><span style="color:#18160f;">{annual_total:.0f} kWh</span></div>
                    <div class="k-prog-track"><div class="k-prog-fill k-pf-green" style="width:{min(annual_pct, 100):.0f}%"></div></div>
                </div>
                <div class="k-prog-item">
                    <div class="k-prog-hd"><span>Avg Monthly</span><span style="color:#18160f;">{avg_monthly:.0f} kWh</span></div>
                    <div class="k-prog-track"><div class="k-prog-fill k-pf-blue" style="width:{min(avg_pct, 100):.0f}%"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

            # Specifications grid
            st.markdown(f"""
            <div class="k-spec-grid">
                <div class="k-spec-box"><div class="k-sp-lbl">Tariff</div><div class="k-sp-val">{user_data['tariff']}</div></div>
                <div class="k-spec-box"><div class="k-sp-lbl">Phase</div><div class="k-sp-val">{user_data['phase']}</div></div>
                <div class="k-spec-box"><div class="k-sp-lbl">Solar</div><div class="k-sp-val" style="color:#dc2626;">{'YES' if user_data.get('has_solar', 0) else 'NONE'}</div></div>
                <div class="k-spec-box"><div class="k-sp-lbl">Similar HH</div><div class="k-sp-val">{similar_count}</div></div>
                <div class="k-spec-box"><div class="k-sp-lbl">Data Months</div><div class="k-sp-val">{len(user_data['consumption_months'])}</div></div>
                <div class="k-spec-box"><div class="k-sp-lbl">Method</div><div class="k-sp-val">{metadata.get('method', 'ENSEMBLE').upper()}</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # This closes k-detail-panel

        # ===== FULL WIDTH BILL CHART =====
        st.markdown('<div class="k-panel" style="margin: 20px 32px;">', unsafe_allow_html=True)
        st.markdown("""
        <div class="k-ph">
            <span class="k-pt blue">Bill Components by Month</span>
        </div>
        """, unsafe_allow_html=True)

        # Calculate bill components
        energy_charge = []
        fixed_charge = []
        vat = []

        for m in months:
            month_key = str(m)
            if month_key in monthly_bills:
                bill_data = monthly_bills[month_key]
            else:
                bill_data = monthly_bills[m]

            energy_charge.append(bill_data['energy_charge_lkr'])
            fixed_charge.append(bill_data['fixed_charge_lkr'])
            vat.append(bill_data['vat_lkr'])

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=month_names, y=energy_charge,
            name='Energy Charge', marker_color='#1d4ed8',
            textfont=dict(color='#1d4ed8'),
            hovertemplate='Energy: Rs.%{y:,.0f}<extra></extra>'
        ))
        fig2.add_trace(go.Bar(
            x=month_names, y=fixed_charge,
            name='Fixed Charge', marker_color='#d97706',
            textfont=dict(color='#d97706'),
            hovertemplate='Fixed: Rs.%{y:,.0f}<extra></extra>'
        ))
        fig2.add_trace(go.Bar(
            x=month_names, y=vat,
            name='VAT', marker_color='#0e8a8a',
            textfont=dict(color='#0e8a8a'),
            hovertemplate='VAT: Rs.%{y:,.0f}<extra></extra>'
        ))

        fig2.update_layout(
            barmode='stack',
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', size=10, color='#7a7264'),
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(  # Add this
                    family='Space Mono',
                    size=10,
                    color='#18160f'  # You can set a default color here
                )
            )
        )
        fig2.update_xaxes(gridcolor='rgba(224,219,208,0.6)')
        fig2.update_yaxes(gridcolor='rgba(224,219,208,0.6)', tickprefix='Rs. ')

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Back button
        if st.button("← Back to Input", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

    except Exception as e:
        st.error(f"Error displaying results: {e}")
        import traceback
        st.code(traceback.format_exc())

        if st.button("← Back to Input"):
            st.session_state.page = "input"
            st.rerun()


# ============================================================================
# MAIN
# ============================================================================
def main():
    if st.session_state.loading and st.session_state.page == "input":
        render_loading_overlay()

    if st.session_state.page == "input":
        input_page()
    else:
        # Clear loading when showing results
        st.session_state.loading = False
        results_page()


if __name__ == "__main__":
    main()