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
    page_title="Kinetic | Financial Intelligence",
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
# SESSION STATE
# ---------------------------------------------------------
if "results"     not in st.session_state: st.session_state.results     = None
if "system_size" not in st.session_state: st.session_state.system_size = 5
if "annual_gen"  not in st.session_state: st.session_state.annual_gen  = 7200
if "n_sims"      not in st.session_state: st.session_state.n_sims      = 2000

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def fmt_val(value):
    if isinstance(value, str): return value
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}<span class='unit'>M</span>"
    elif value >= 1_000:
        return f"{value/1_000:.1f}<span class='unit'>K</span>"
    return f"{value:,.0f}"

def render_kpi(label, value, unit, cls):
    st.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

def scenario_row(label, sub, value, vcls):
    return f"""<li class="scenario-item">
        <div><div class="scenario-label">{label}</div>
        <div class="scenario-sub">{sub}</div></div>
        <div class="scenario-val {vcls}">{value}</div>
    </li>"""

def risk_bar(label, detail, pct, color):
    return f"""<div class="risk-bar-row">
        <div class="risk-bar-label"><span>{label}</span><span>{detail}</span></div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>"""

def vendor_card(name, desc, contact, nc="var(--text)"):
    st.markdown(f"""<div class="vendor-card">
        <div class="vendor-info">
            <div class="vendor-name" style="color:{nc}">{name}</div>
            <div class="vendor-desc">{desc}</div>
        </div>
        <div class="vendor-contact">{contact}</div>
    </div>""", unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7a7264', family='Space Mono', size=10),
    margin=dict(l=10, r=10, t=48, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=10, family='Space Mono', color='#7a7264'))
)
GRID = dict(showgrid=True, gridwidth=1, gridcolor='rgba(224,219,208,0.7)', zeroline=False)

# ---------------------------------------------------------
# NAV
# ---------------------------------------------------------
now = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<nav class="kin-nav">
  <div class="kin-nav-logo">
    <span class="kin">Solar</span><span class="etic">Grid</span>
    <span class="dot"></span>
  </div>
  <div class="kin-nav-badge">V2.0 · Intelligence System</div>
  <div class="kin-nav-live"><div class="kin-nav-dot"></div></div>
</nav>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO + INPUT CARD
# ---------------------------------------------------------
hero_l, hero_r = st.columns([3, 2], gap="large")

with hero_l:
    st.markdown("""
    <div class="kin-hero">
        <div class="kin-hero-tag">Financial AI Assessment</div>
        <div class="kin-hero-title">ROI &amp;<br>Risk<br><span class="accent">Intelligence</span></div>
        <p class="kin-hero-sub">Monte Carlo–powered financial modeling that simulates 2,000
        possible futures — quantifying ROI, payback period, and investment risk under
        real-world uncertainty.</p>
        <div class="kin-pills">
            <span class="kin-pill active">Monte Carlo Simulation</span>
            <span class="kin-pill active">Risk Analysis</span>
            <span class="kin-pill active">PUCSL Tariff Integration</span>
            <span class="kin-pill">Panel Degradation</span>
            <span class="kin-pill">Inverter Risk</span>
        </div>
    </div>""", unsafe_allow_html=True)

with hero_r:
    st.markdown('<div class="input-card"><div class="card-title">Financial Parameters</div>',
                unsafe_allow_html=True)

    size_opts = {3:"3 kW — Single Phase", 5:"5 kW — Single/Three Phase",
                 8:"8 kW — Three Phase", 10:"10 kW", 15:"15 kW", 20:"20 kW"}
    system_size = st.selectbox("SYSTEM SIZE (FROM COMPONENT 2)",
                               list(size_opts.keys()),
                               format_func=lambda x: size_opts[x], index=1)
    annual_gen  = st.number_input("ANNUAL GENERATION KWH (FROM COMPONENT 1)",
                                  min_value=1000, max_value=50000, value=7200, step=100)
    ca, cb = st.columns(2)
    with ca: tariff   = st.number_input("EXPORT TARIFF (LKR/KWH)", min_value=1.0, max_value=100.0, value=27.06, step=0.01)
    with cb: discount = st.number_input("DISCOUNT RATE (%)", min_value=1, max_value=30, value=10)
    sim_opts = {500:"500 — Fast Preview", 2000:"2,000 — Standard", 5000:"5,000 — High Precision"}
    n_sims = st.selectbox("SIMULATIONS", list(sim_opts.keys()),
                          format_func=lambda x: sim_opts[x], index=1)
    run_btn = st.button("▶  RUN FINANCIAL ANALYSIS", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOADING + RUN
# ---------------------------------------------------------
if run_btn:
    overlay = st.empty()
    overlay.markdown("""
    <div class="loading-overlay">
        <div class="loader-ring"></div>
        <div class="sim-counter">2,000</div>
        <div class="loader-text">Running Monte Carlo Simulations</div>
        <div class="loader-progress"><div class="loader-bar"></div></div>
    </div>""", unsafe_allow_html=True)

    results = run_roi_analysis(system_size, annual_gen, annual_gen)
    time.sleep(1.2)
    overlay.empty()

    if "error" not in results:
        st.session_state.results     = results
        st.session_state.system_size = system_size
        st.session_state.annual_gen  = annual_gen
        st.session_state.n_sims      = n_sims
    else:
        st.error(f"Error: {results['error']}")

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
res = st.session_state.results
if res and "error" not in res:
    sz  = st.session_state.system_size
    gen = st.session_state.annual_gen
    ns  = st.session_state.n_sims
    ts  = datetime.now().strftime("%H:%M:%S")

    st.divider()

    # Results header
    st.markdown(f"""
    <div class="results-header">
        <div>
            <div class="results-title">Financial Analysis <span>Complete</span></div>
            <div class="results-sub">{sz} kW System · {gen:,} kWh/yr · {ns:,} Simulations</div>
        </div>
        <div class="results-meta">{ts}</div>
    </div>""", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: render_kpi("Expected ROI",     f"{res['Expected_ROI_Percent']}%",                                         "% over 20 years",  "c1")
    with k2: render_kpi("Payback Period",   f"{res['Expected_Payback_Years']}<span class='unit'> yrs</span>",          "years (median)",   "c2")
    with k3: render_kpi("Total Investment", fmt_val(res['Total_Investment_LKR']),                                      "LKR (CAPEX)",      "c3")
    with k4: render_kpi("Worst-Case ROI",   f"{res['Risk_Analysis']['Worst_Case_ROI_Percent']}%",                      "5th percentile",   "c4")
    with k5: render_kpi("Risk Certainty",   res['Risk_Analysis']['Certainty_Score'],                                   "Investment grade", "c5")

    st.divider()

    # Chart data
    cd    = res["Chart_Data"]
    years = cd["Years_Labels_0_to_20"]

    # Wide cashflow chart
    st.markdown("""
    <div class="chart-card-header">
        <div>
            <div class="chart-title">Cumulative Cash Flow Projection</div>
            <div class="chart-sub">Expected + confidence band (P10–P90) over 20-year lifetime</div>
        </div>
        <div class="chart-legend">
            <div class="legend-item"><div class="legend-dot" style="background:#f4601a"></div>Expected</div>
            <div class="legend-item"><div class="legend-dot" style="background:rgba(244,96,26,0.3)"></div>P10–P90 Band</div>
            <div class="legend-item"><div class="legend-dot" style="background:#d97706"></div>Breakeven</div>
        </div>
    </div>""", unsafe_allow_html=True)

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_P90"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_P10"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(244,96,26,0.1)', name='P10–P90 Band'))
    fig_cf.add_trace(go.Scatter(x=years, y=cd["Cumulative_Cash_Flow_Expected"],
        mode='lines+markers', line=dict(color='#f4601a', width=2.5),
        marker=dict(color='#f4601a', size=5), name='Expected'))
    fig_cf.add_hline(y=0, line_dash="dash", line_color="#d97706", line_width=1.5,
                     annotation_text="Breakeven", annotation_font_color="#d97706",
                     annotation_font_size=10)
    fig_cf.update_layout(xaxis_title="Years", yaxis_title="Cumulative Cash (LKR)",
                         hovermode="x unified", **PLOTLY_BASE)
    fig_cf.update_xaxes(**GRID)
    fig_cf.update_yaxes(**GRID, tickprefix="LKR ")
    st.plotly_chart(fig_cf, use_container_width=True)

    # ROI dist + Revenue side by side
    cl, cr = st.columns(2, gap="medium")
    with cl:
        st.markdown("""<div class="chart-card-header">
            <div><div class="chart-title">ROI Distribution</div>
            <div class="chart-sub">Simulation histogram · 2000 outcomes</div></div>
        </div>""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=cd["Monte_Carlo_NPV_Distribution"],
            nbinsx=20, marker_color='#18a058', opacity=0.8))
        fig2.add_vline(x=res["Expected_NPV_LKR"], line_dash="dash", line_color="#dc2626",
                       annotation_text="Expected", annotation_font_color="#dc2626",
                       annotation_font_size=10)
        fig2.add_vline(x=0, line_color="#18160f", line_width=1.5)
        fig2.update_layout(xaxis_title="NPV (LKR)", yaxis_title="Frequency",
                           bargap=0.05, showlegend=False, **PLOTLY_BASE)
        fig2.update_xaxes(**GRID)
        fig2.update_yaxes(**GRID)
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        st.markdown("""<div class="chart-card-header">
            <div><div class="chart-title">Annual Revenue Forecast</div>
            <div class="chart-sub">With tariff escalation &amp; panel degradation</div></div>
        </div>""", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=years[1:], y=cd["Yearly_Revenue_Forecast"],
            marker_color='rgba(244,96,26,0.55)', marker_line_width=0))
        fig3.update_layout(xaxis_title="Year", yaxis_title="Net Cashflow (LKR)",
                           showlegend=False, **PLOTLY_BASE)
        fig3.update_xaxes(**GRID, dtick=1)
        fig3.update_yaxes(**GRID, tickprefix="LKR ")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # 3-col bottom: Risk | Scenario | Recommendation
    b1, b2, b3 = st.columns(3, gap="medium")
    scen = res["Scenario_Analysis"]
    risk = res["Risk_Analysis"]

    with b1:
        overall_pct = 30 if risk.get('Certainty_Score') == 'High' else 65
        st.markdown(f"""<div class="risk-card">
            <div class="risk-title">Risk Factor Breakdown</div>
            <div class="divider-line"></div>
            {risk_bar("Panel Degradation",    "0.75% / yr", 75, "#f4601a")}
            {risk_bar("Maintenance Volatility","±20%",       55, "#d97706")}
            {risk_bar("Inverter Failure Risk", "Year 8–12",  45, "#dc2626")}
            {risk_bar("Tariff Escalation",     "2–5% / yr",  65, "#18a058")}
            {risk_bar("Overall Risk Score",    "—",   overall_pct, "#7c3aed")}
        </div>""", unsafe_allow_html=True)

    with b2:
        rows = (
            scenario_row("Best Case ROI",        "P95 · Optimal degradation", f"{scen['Best_Case_ROI_Percent']}%",        "up")  +
            scenario_row("Expected ROI",          "P50 · Median outcome",      f"{res['Expected_ROI_Percent']}%",          "mid") +
            scenario_row("Worst Case ROI",        "P05 · Conservative",        f"{risk['Worst_Case_ROI_Percent']}%",       "down")+
            scenario_row("Shortest Payback",      "P10 estimate",              f"{scen['Shortest_Payback_Years']} yrs",    "up")  +
            scenario_row("Longest Payback",       "P90 estimate",              f"{risk['Worst_Case_Payback_Years']} yrs",  "down")+
            scenario_row("Prob. of Positive ROI", "% of simulations",         f"{scen['Probability_Positive_ROI']}%",    "up")
        )
        st.markdown(f"""<div class="risk-card">
            <div class="risk-title">Scenario Analysis</div>
            <div class="divider-line"></div>
            <ul class="scenario-list">{rows}</ul>
        </div>""", unsafe_allow_html=True)

    with b3:
        rec       = res["Recommendation"]
        payback_f = res['Expected_Payback_Years']
        try:    pb = float(str(payback_f).replace(' yrs','').strip())
        except: pb = 5.0
        mid_yr   = round(pb * 0.5)
        break_yr = round(pb)

        if "Excellent" in rec:
            bcls, blbl = "excellent", "⬤ &nbsp;Excellent Investment"
        elif "Good" in rec:
            bcls, blbl = "good", "⬤ &nbsp;Good Investment"
        else:
            bcls, blbl = "good", "⬤ &nbsp;Marginal Return"

        st.markdown(f"""<div class="risk-card">
            <div class="risk-title">Recommendation</div>
            <div class="divider-line"></div>
            <div class="rec-badge {bcls}">{blbl}</div>
            <div class="rec-detail">{rec}</div>
            <div class="tl-header">Payback Timeline</div>
            <div class="timeline">
                <div class="timeline-item done">
                    <div class="tl-year">Year 0</div>
                    <div class="tl-label">Install {sz}kW System</div>
                </div>
                <div class="timeline-item">
                    <div class="tl-year">Year {mid_yr}</div>
                    <div class="tl-label">Mid-point milestone</div>
                </div>
                <div class="timeline-item active">
                    <div class="tl-year">Year {break_yr}</div>
                    <div class="tl-label">✓ Breakeven Point</div>
                </div>
                <div class="timeline-item">
                    <div class="tl-year">Year 12</div>
                    <div class="tl-label">Inverter replacement</div>
                </div>
                <div class="timeline-item">
                    <div class="tl-year">Year 20</div>
                    <div class="tl-label">End of lifecycle</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Vendors
    st.markdown(f'<div class="kin-section-label">Local Vendor Recommendations · {sz} kW System</div>',
                unsafe_allow_html=True)
    vcols = st.columns(3)
    for i, v in enumerate(res["Recommended_Local_Vendors"]):
        with vcols[i % 3]:
            vendor_card(v['Name'],
                        f"📍 {v['Location']} &nbsp;|&nbsp; {v['Specialty']}",
                        f"📞 {v['Contact']}")

else:
    # Empty state
    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:60px 0 40px;">
        <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.2em;
                    text-transform:uppercase;color:#a09880;margin-bottom:16px;">
            Awaiting Simulation
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
                    color:#3d3929;margin-bottom:12px;">
            Configure parameters and run the simulation
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:14px;color:#7a7264;font-weight:300;">
            Fill in the parameters on the right and click
            <strong style="color:#f4601a;">Run Financial Analysis</strong>
        </div>
    </div>""", unsafe_allow_html=True)