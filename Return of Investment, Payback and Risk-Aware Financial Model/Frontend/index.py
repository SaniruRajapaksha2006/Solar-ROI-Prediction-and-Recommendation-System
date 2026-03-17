import streamlit as st
import sys
import os
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------------------------------------------------------
# BACKEND CONNECTION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
backend_dir = os.path.join(parent_dir, 'Backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from financeml import SolarFinancialModel

def run_roi_analysis(size, gen, cons):
    model = SolarFinancialModel()
    return model.calculate_financial_report(size, gen, cons)

# ---------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Kinetic | ROI & Risk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = os.path.join(current_dir, "style.css")
load_css(css_path)

# ---------------------------------------------------------
# READ INPUTS FROM SESSION STATE
# These will be set by the other model tabs when integrated.
# Falls back to defaults if running standalone.
# ---------------------------------------------------------
system_size_kw             = st.session_state.get("system_size_kw", 5)
predicted_annual_gen_kwh   = st.session_state.get("predicted_annual_gen_kwh", 7200)
predicted_annual_cons_kwh  = st.session_state.get("predicted_annual_cons_kwh", 4800)

# ---------------------------------------------------------
# SESSION STATE for results
# ---------------------------------------------------------
if "roi_results" not in st.session_state:
    st.session_state.roi_results = None
if "roi_ran_for" not in st.session_state:
    st.session_state.roi_ran_for = None

# Auto-run when inputs change (new values from other tabs)
current_inputs = (system_size_kw, predicted_annual_gen_kwh, predicted_annual_cons_kwh)
if st.session_state.roi_ran_for != current_inputs:
    st.session_state.roi_results = None  # invalidate old results

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def fmt_val(value):
    if isinstance(value, str): return value
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}<span class='s-unit'>M</span>"
    elif value >= 1_000:
        return f"{value/1_000:.1f}<span class='s-unit'>K</span>"
    return f"{value:,.0f}"

def stat_card(label, value, sub, cls):
    st.markdown(f"""
    <div class="stat-card {cls}">
        <div class="s-lbl">{label}</div>
        <div class="s-val">{value}</div>
        <div class="s-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def vendor_card(name, location, specialty, contact):
    st.markdown(f"""
    <div class="kin-vendor-card">
        <div class="kin-vendor-info">
            <div class="kin-vendor-name">{name}</div>
            <div class="kin-vendor-desc">📍 {location} &nbsp;·&nbsp; {specialty}</div>
        </div>
        <div class="kin-vendor-contact">📞 {contact}</div>
    </div>""", unsafe_allow_html=True)

# Plotly base — warm beige grid
PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7a7264', family='Space Mono', size=10),
    margin=dict(l=8, r=8, t=16, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10, family='Space Mono', color='#7a7264'))
)
GRID = dict(showgrid=True, gridwidth=1, gridcolor='rgba(224,219,208,0.6)',
            zeroline=False, tickfont=dict(family='Space Mono', size=10, color='#7a7264'))

# ---------------------------------------------------------
# RESULTS TOPBAR
# ---------------------------------------------------------
ts = datetime.now().strftime("%I:%M:%S %p")
st.markdown(f"""
<div class="kin-rtb">
    <div class="kin-rtb-row">
        <div>
            <div class="kin-rtb-title">ROI &amp; Risk <span>Intelligence</span></div>
            <div class="kin-rtb-meta">
                {system_size_kw} KW · {predicted_annual_gen_kwh:,} KWH/YR · 2,000 SIMULATIONS · MONTE CARLO
            </div>
        </div>
        <div class="kin-rtb-clock">{ts}</div>
    </div>
    <div class="kin-tab-bar">
        <div class="kin-tab active">
            <span class="kin-tab-num">4</span> ROI &amp; Risk
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# RUN BUTTON  (minimal — no config form)
# ---------------------------------------------------------
_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    run_btn = st.button("▶  Run Financial Analysis", use_container_width=True)

# ---------------------------------------------------------
# LOADING + RUN
# ---------------------------------------------------------
if run_btn:
    overlay = st.empty()
    overlay.markdown("""
    <div class="kin-loading">
        <div class="kin-loader-ring"></div>
        <div class="kin-sim-counter">2,000</div>
        <div class="kin-loader-text">Running Monte Carlo Simulations</div>
        <div class="kin-loader-progress"><div class="kin-loader-bar"></div></div>
    </div>""", unsafe_allow_html=True)

    results = run_roi_analysis(
        system_size_kw,
        predicted_annual_gen_kwh,
        predicted_annual_cons_kwh
    )
    time.sleep(1.2)
    overlay.empty()

    if "error" not in results:
        st.session_state.roi_results  = results
        st.session_state.roi_ran_for  = current_inputs
    else:
        st.error(f"Error: {results['error']}")

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
res = st.session_state.roi_results

if res and "error" not in res:

    # ── STAT ROW ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat_card("Expected ROI",
                        f"{res['Expected_ROI_Percent']}<span class='s-unit'>%</span>",
                        "over 20 years", "sc-orange")
    with c2: stat_card("Payback Period",
                        str(res['Expected_Payback_Years']),
                        "years (median)", "sc-amber")
    with c3: stat_card("Total Investment",
                        fmt_val(res['Total_Investment_LKR']),
                        "LKR (CAPEX)", "sc-green")
    with c4: stat_card("Worst-Case ROI",
                        f"{res['Risk_Analysis']['Worst_Case_ROI_Percent']}<span class='s-unit'>%</span>",
                        "5th percentile", "sc-blue")
    with c5: stat_card("Risk Certainty",
                        res['Risk_Analysis']['Certainty_Score'],
                        "Investment grade", "sc-red")

    # ── CASHFLOW CHART — full width ───────────────────
    cd    = res["Chart_Data"]
    years = cd["Years_Labels_0_to_20"]

    st.markdown("""
    <div class="kin-panel">
        <div class="kin-ph">
            <div>
                <div class="kin-pt orange">Cumulative Cash Flow Projection</div>
                <div class="kin-ps">Expected + confidence band (P10–P90) over 20-year lifetime</div>
            </div>
            <div class="kin-leg-row">
                <div class="kin-leg-it"><div class="kin-leg-d" style="background:#f4601a"></div><span class="kin-leg-t">Expected</span></div>
                <div class="kin-leg-it"><div class="kin-leg-d" style="background:rgba(244,96,26,0.25)"></div><span class="kin-leg-t">P10–P90 Band</span></div>
                <div class="kin-leg-it"><div class="kin-leg-d" style="background:#d97706"></div><span class="kin-leg-t">Breakeven</span></div>
            </div>
        </div>
        <div class="kin-pb">""", unsafe_allow_html=True)

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_P90"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_P10"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(244,96,26,0.08)', showlegend=False))
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_Expected"],
        mode='lines+markers', name='Expected',
        line=dict(color='#f4601a', width=2.5),
        marker=dict(color='#f4601a', size=4, line=dict(color='#fff', width=2))))
    fig_cf.add_hline(y=0, line_dash="dash", line_color="rgba(217,119,6,0.7)",
                     line_width=1.5, annotation_text="Breakeven",
                     annotation_font_color="#d97706", annotation_font_size=10)
    fig_cf.update_layout(height=240, hovermode="x unified", **PLOTLY)
    fig_cf.update_xaxes(**GRID)
    fig_cf.update_yaxes(**GRID, tickprefix="LKR ", tickformat=",")
    st.plotly_chart(fig_cf, use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── ROI DIST + REVENUE — two columns ─────────────
    cl, cr = st.columns(2, gap="small")

    with cl:
        st.markdown("""
        <div class="kin-panel" style="margin-bottom:0">
            <div class="kin-ph"><div>
                <div class="kin-pt orange">ROI Distribution</div>
                <div class="kin-ps">Simulation histogram · 2,000 outcomes</div>
            </div></div>
            <div class="kin-pb">""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=cd["Monte_Carlo_NPV_Distribution"],
            nbinsx=18, marker_color='rgba(24,160,88,0.7)', marker_line_width=0))
        fig2.add_vline(x=res["Expected_NPV_LKR"], line_dash="dash",
                       line_color="rgba(220,38,38,0.7)", line_width=1.5,
                       annotation_text="Expected", annotation_font_size=10,
                       annotation_font_color="#dc2626")
        fig2.add_vline(x=0, line_color="rgba(24,22,15,0.2)", line_width=1)
        fig2.update_layout(height=220, showlegend=False, **PLOTLY)
        fig2.update_xaxes(**GRID)
        fig2.update_yaxes(**GRID)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with cr:
        st.markdown("""
        <div class="kin-panel" style="margin-bottom:0">
            <div class="kin-ph"><div>
                <div class="kin-pt orange">Annual Revenue Forecast</div>
                <div class="kin-ps">With tariff escalation &amp; panel degradation</div>
            </div></div>
            <div class="kin-pb">""", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=years[1:], y=cd["Yearly_Revenue_Forecast"],
            marker_color='rgba(244,96,26,0.55)', marker_line_width=0))
        fig3.update_layout(height=220, showlegend=False, **PLOTLY)
        fig3.update_xaxes(**GRID, dtick=2)
        fig3.update_yaxes(**GRID, tickprefix="LKR ", tickformat=",")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── BOTTOM 3-COL: Risk | Scenario | Recommendation ──
    b1, b2, b3 = st.columns(3, gap="small")
    scen = res["Scenario_Analysis"]
    risk = res["Risk_Analysis"]

    with b1:
        overall_pct = 28 if risk.get('Certainty_Score') == 'High' else 60
        st.markdown(f"""
        <div class="kin-panel" style="margin-bottom:0">
            <div class="kin-ph"><span class="kin-pt amber">Risk Factor Breakdown</span></div>
            <div class="kin-pb">
                <div class="kin-risk-item">
                    <div class="kin-risk-hd"><span>Panel Degradation</span><span class="kin-risk-v">0.75%/yr</span></div>
                    <div class="kin-prog-track"><div class="kin-prog-fill kpf-orange" style="width:75%"></div></div>
                </div>
                <div class="kin-risk-item">
                    <div class="kin-risk-hd"><span>Maintenance Volatility</span><span class="kin-risk-v">±20%</span></div>
                    <div class="kin-prog-track"><div class="kin-prog-fill kpf-amber" style="width:55%"></div></div>
                </div>
                <div class="kin-risk-item">
                    <div class="kin-risk-hd"><span>Inverter Failure Risk</span><span class="kin-risk-v">Yr 8–12</span></div>
                    <div class="kin-prog-track"><div class="kin-prog-fill kpf-red" style="width:40%"></div></div>
                </div>
                <div class="kin-risk-item">
                    <div class="kin-risk-hd"><span>Tariff Escalation</span><span class="kin-risk-v">2–5%/yr</span></div>
                    <div class="kin-prog-track"><div class="kin-prog-fill kpf-green" style="width:65%"></div></div>
                </div>
                <div class="kin-risk-item">
                    <div class="kin-risk-hd"><span>Overall Risk Score</span><span class="kin-risk-v">—</span></div>
                    <div class="kin-prog-track"><div class="kin-prog-fill kpf-purple" style="width:{overall_pct}%"></div></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class="kin-panel" style="margin-bottom:0">
            <div class="kin-ph"><span class="kin-pt amber">Scenario Analysis</span></div>
            <div class="kin-pb" style="padding:12px 20px;">
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Best Case ROI</div><div class="kin-scen-s">P95 · Optimal degradation</div></div>
                    <div class="kin-scen-val" style="color:#18a058">{scen['Best_Case_ROI_Percent']}%</div>
                </div>
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Expected ROI</div><div class="kin-scen-s">P50 · Median outcome</div></div>
                    <div class="kin-scen-val" style="color:#f4601a">{res['Expected_ROI_Percent']}%</div>
                </div>
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Worst Case ROI</div><div class="kin-scen-s">P05 · Conservative</div></div>
                    <div class="kin-scen-val" style="color:#dc2626">{risk['Worst_Case_ROI_Percent']}%</div>
                </div>
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Shortest Payback</div><div class="kin-scen-s">P10 estimate</div></div>
                    <div class="kin-scen-val" style="color:#3b82f6">{scen['Shortest_Payback_Years']} yrs</div>
                </div>
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Longest Payback</div><div class="kin-scen-s">P90 estimate</div></div>
                    <div class="kin-scen-val" style="color:#dc2626">{risk['Worst_Case_Payback_Years']} yrs</div>
                </div>
                <div class="kin-scen-row">
                    <div><div class="kin-scen-lbl">Prob. Positive ROI</div><div class="kin-scen-s">% of simulations</div></div>
                    <div class="kin-scen-val" style="color:#18a058">{scen['Probability_Positive_ROI']}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with b3:
        rec = res["Recommendation"]
        try:    pb = float(str(res['Expected_Payback_Years']).replace(' yrs','').strip())
        except: pb = 5.0
        mid_yr   = round(pb * 0.5)
        break_yr = round(pb)
        if "Excellent" in rec:   bcls, blbl = "excellent", "Excellent Investment"
        elif "Good" in rec:      bcls, blbl = "good",      "Good Investment"
        else:                    bcls, blbl = "good",      "Marginal Return"

        st.markdown(f"""
        <div class="kin-panel" style="margin-bottom:0">
            <div class="kin-ph"><span class="kin-pt amber">Recommendation</span></div>
            <div class="kin-pb">
                <div class="kin-rec-badge {bcls}">{blbl}</div>
                <p class="kin-rec-text">{rec}</p>
                <div class="kin-tl-wrap">
                    <div class="kin-tl-lbl">Payback Timeline</div>
                    <div class="kin-tl-item done">
                        <span class="kin-tl-yr">Year 0</span>
                        <span class="kin-tl-ev">Install {system_size_kw}kW System</span>
                    </div>
                    <div class="kin-tl-item">
                        <span class="kin-tl-yr">Year {mid_yr}</span>
                        <span class="kin-tl-ev">Mid-point milestone</span>
                    </div>
                    <div class="kin-tl-item now">
                        <span class="kin-tl-yr">Year {break_yr}</span>
                        <span class="kin-tl-ev">✓ Breakeven Point</span>
                    </div>
                    <div class="kin-tl-item">
                        <span class="kin-tl-yr">Year 12</span>
                        <span class="kin-tl-ev">Inverter replacement</span>
                    </div>
                    <div class="kin-tl-item">
                        <span class="kin-tl-yr">Year 20</span>
                        <span class="kin-tl-ev">End of lifecycle</span>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── VENDOR CARDS ──────────────────────────────────
    st.markdown("""
    <div class="kin-panel" style="margin-top:16px;">
        <div class="kin-ph">
            <div>
                <div class="kin-pt orange">Local Vendor Recommendations</div>
                <div class="kin-ps">Certified installers based on your system size &amp; location</div>
            </div>
        </div>
        <div class="kin-pb kin-vendor-grid">""", unsafe_allow_html=True)

    vendors = res.get("Recommended_Local_Vendors", [])
    # render in columns of 3
    for i in range(0, len(vendors), 3):
        row = vendors[i:i+3]
        cols = st.columns(len(row))
        for col, v in zip(cols, row):
            with col:
                vendor_card(
                    v.get("Name", "—"),
                    v.get("Location", "—"),
                    v.get("Specialty", "—"),
                    v.get("Contact", "—")
                )

    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------
else:
    st.markdown(f"""
    <div class="kin-empty">
        <div class="kin-empty-icon">⚡</div>
        <div class="kin-empty-title">Ready to Analyse</div>
        <div class="kin-empty-sub">
            Parameters received from your models —
            <strong>{system_size_kw} kW</strong> system,
            <strong>{predicted_annual_gen_kwh:,} kWh/yr</strong> generation,
            <strong>{predicted_annual_cons_kwh:,} kWh/yr</strong> consumption.<br><br>
            Click <strong>Run Financial Analysis</strong> above to generate your report.
        </div>
    </div>""", unsafe_allow_html=True)