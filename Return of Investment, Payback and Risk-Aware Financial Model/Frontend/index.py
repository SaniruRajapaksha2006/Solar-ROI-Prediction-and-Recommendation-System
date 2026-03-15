import streamlit as st
import sys
import os
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# ROBUST BACKEND CONNECTION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
backend_dir = os.path.join(parent_dir, 'Backend')

if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from financial_controller import run_roi_analysis
except ImportError:
    from monte_carlo_engine import SolarFinancialModel


    def run_roi_analysis(size, gen, cons):
        model = SolarFinancialModel()
        return model.calculate_financial_report(size, gen, cons)

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CSS LOADER
# ---------------------------------------------------------
st.set_page_config(page_title="Kinetic | Financial Intelligence", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = os.path.join(current_dir, "style.css")
load_css(css_path)

# --- TEMPORARY INLINE CSS (Vendor, Scenario, and Stat Cards left to migrate) ---
st.markdown("""
<style>
    /* VENDOR CARD CSS */
    .vendor-card {
        background-color: #ffffff;
        border: 1px solid #e0dbd0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .vendor-info {
        display: flex;
        flex-direction: column;
    }
    .vendor-name {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 15px;
        color: #1f2937;
    }
    .vendor-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .vendor-contact {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        color: #18a058;
        font-size: 14px;
    }

    /* SCENARIO ANALYSIS CSS */
    .scen-card {
        background-color: #ffffff;
        border: 1px solid #e0dbd0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    .scen-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px dashed #e0dbd0;
    }
    .scen-row:last-child {
        border-bottom: none;
    }
    .scen-lbl {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 14px;
        color: #4b5563;
    }
    .scen-val {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 15px;
        color: #9ca3af; 
    }
    .val-green { color: #18a058; }
    .val-orange { color: #f4601a; }
    .val-red { color: #dc2626; }

    /* TOP KPI STAT CARDS */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e0dbd0;
        border-radius: 8px;
        padding: 20px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .s-lbl {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 13px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .s-val {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 28px;
        color: #1f2937;
    }
    .s-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MAIN DASHBOARD HEADER
# ---------------------------------------------------------
st.title("⚡ Kinetic: AI-Driven Solar ROI & Risk-Aware Dashboard")
st.markdown(
    "Evaluate the financial feasibility, risks, and payback of your residential solar PV investment in Sri Lanka.")

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.caption("🟢 **System Status:** Online | 🧠 AI Models: Connected | 📡 Market Data: Synced")
with head_col2:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🕒 **Current Session:** {current_time}")

st.divider()

# ---------------------------------------------------------
# 3. SIDEBAR: USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("⚙️ System Inputs")
st.sidebar.markdown("Configure the parameters predicted by the AI models.")

system_size_kw = st.sidebar.number_input("Recommended System Size (kW)", min_value=1.0, max_value=20.0, value=5.0,
                                         step=0.5)
annual_gen_kwh = st.sidebar.number_input("Predicted Annual Generation (kWh)", min_value=1000, max_value=50000,
                                         value=7200, step=100)
annual_cons_kwh = st.sidebar.number_input("Predicted Annual Consumption (kWh)", min_value=1000, max_value=50000,
                                          value=4800, step=100)

st.sidebar.divider()
calculate_btn = st.sidebar.button("📊 Run Financial Simulation", use_container_width=True)

# ---------------------------------------------------------
# 4. BACKEND EXECUTION & DASHBOARD LAYOUT
# ---------------------------------------------------------
if calculate_btn:
    with st.spinner("Running 2,000 Monte Carlo Risk Simulations..."):
        results = run_roi_analysis(system_size_kw, annual_gen_kwh, annual_cons_kwh)

        if "error" in results:
            st.error(f"An error occurred: {results['error']}")
        else:
            st.success("✅ Financial Analysis Complete!")


            def format_currency(value):
                if value >= 1000000:
                    return f"{value / 1000000:.2f}<span style='font-size:16px; color:#9ca3af;'>M</span>"
                elif value >= 1000:
                    return f"{value / 1000:.1f}<span style='font-size:16px; color:#9ca3af;'>K</span>"
                return f"{value:,.0f}"


            # --- SECTION 1: TOP KPI METRICS ---
            st.markdown("### 📊 Expected Financial Outcomes")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

            with kpi_col1:
                st.markdown(f"""
                <div class="stat-card" style="border-top: 4px solid #1f2937;">
                    <div class="s-lbl">Total Investment</div>
                    <div class="s-val">{format_currency(results['Total_Investment_LKR'])}</div>
                    <div class="s-sub">CAPEX (LKR)</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(f"""
                <div class="stat-card" style="border-top: 4px solid #18a058;">
                    <div class="s-lbl">Expected NPV</div>
                    <div class="s-val">{format_currency(results['Expected_NPV_LKR'])}</div>
                    <div class="s-sub">Discounted (LKR)</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f"""
                <div class="stat-card" style="border-top: 4px solid #f4601a;">
                    <div class="s-lbl">Expected ROI</div>
                    <div class="s-val">{results['Expected_ROI_Percent']}<span style='font-size:16px; color:#9ca3af;'>%</span></div>
                    <div class="s-sub">Over 20 Years</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col4:
                st.markdown(f"""
                <div class="stat-card" style="border-top: 4px solid #d97706;">
                    <div class="s-lbl">Payback Period</div>
                    <div class="s-val">{results['Expected_Payback_Years']}<span style='font-size:16px; color:#9ca3af;'>Yrs</span></div>
                    <div class="s-sub">Break-Even Point</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col5:
                certainty = results['Risk_Analysis']['Certainty_Score']
                color = "#18a058" if certainty == "High" else "#dc2626"
                st.markdown(f"""
                <div class="stat-card" style="border-top: 4px solid {color};">
                    <div class="s-lbl">Risk Certainty</div>
                    <div class="s-val" style="color: {color};">{certainty}</div>
                    <div class="s-sub">Based on Monte Carlo</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # --- SECTION 2: CHARTS ---
            st.markdown("### 📈 Visual Risk & Cash Flow Analysis")
            tab1, tab2, tab3 = st.tabs(["Cumulative Cash Flow (Payback)", "Risk Distribution (NPV)", "Yearly Revenue"])

            chart_data = results["Chart_Data"]
            years = chart_data["Years_Labels_0_to_20"]

            with tab1:
                fig_cf = go.Figure()
                fig_cf.add_trace(
                    go.Scatter(x=years, y=chart_data["Cumulative_Cash_Flow_P90"], mode='lines', line=dict(width=0),
                               showlegend=False, hoverinfo='skip'))
                fig_cf.add_trace(
                    go.Scatter(x=years, y=chart_data["Cumulative_Cash_Flow_P10"], mode='lines', line=dict(width=0),
                               fill='tonexty', fillcolor='rgba(244, 96, 26, 0.15)', name='Confidence Band (P10-P90)'))
                fig_cf.add_trace(
                    go.Scatter(x=years, y=chart_data["Cumulative_Cash_Flow_Expected"], mode='lines+markers',
                               line=dict(color='#f4601a', width=3), name='Expected Cumulative Cash Flow'))
                fig_cf.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even Point (Zero)")
                fig_cf.update_layout(title="Cumulative Cash Flow over 20 Years", xaxis_title="Years",
                                     yaxis_title="Cumulative Cash (LKR)", hovermode="x unified",
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color='#1f2937', family="DM Sans"))
                fig_cf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                fig_cf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                st.plotly_chart(fig_cf, use_container_width=True)

            with tab2:
                fig_npv = go.Figure()
                fig_npv.add_trace(
                    go.Histogram(x=chart_data["Monte_Carlo_NPV_Distribution"], nbinsx=50, marker_color='#18a058',
                                 name='NPV Outcomes', opacity=0.75))
                fig_npv.add_vline(x=results["Expected_NPV_LKR"], line_dash="dash", line_color="darkred",
                                  annotation_text=f"Expected: LKR {results['Expected_NPV_LKR']:,.0f}",
                                  annotation_position="top right")
                fig_npv.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
                fig_npv.update_layout(title="Monte Carlo Risk Analysis: NPV Distribution (2000 Scenarios)",
                                      xaxis_title="Net Present Value (LKR)", yaxis_title="Frequency", bargap=0.1,
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color='#1f2937', family="DM Sans"))
                fig_npv.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                fig_npv.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                st.plotly_chart(fig_npv, use_container_width=True)

            with tab3:
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(x=years[1:], y=chart_data["Yearly_Revenue_Forecast"], marker_color='#d97706',
                                         name='Annual Net Cashflow'))
                fig_rev.update_layout(title="Expected Annual Net Cashflow (With Degradation & Maintenance)",
                                      xaxis_title="Year", yaxis_title="Net Cashflow (LKR)", hovermode="x unified",
                                      xaxis=dict(tickmode='linear', dtick=1), paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1f2937', family="DM Sans"))
                fig_rev.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                fig_rev.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0dbd0')
                st.plotly_chart(fig_rev, use_container_width=True)

            st.divider()

            # --- SECTION 3: BOTTOM ROW ---
            bottom_col1, bottom_col2 = st.columns([1, 1])

            with bottom_col1:
                st.markdown("### ⚠️ Scenario Analysis")

                scen_col1, scen_col2 = st.columns(2)
                scen = results["Scenario_Analysis"]
                risk = results["Risk_Analysis"]

                with scen_col1:
                    roi_html = f"""
                    <div class="scen-card">
                        <h5 style="margin-top:0; color:#1f2937; font-family:'Syne', sans-serif;">ROI Profiles</h5>
                        <div class="scen-row">
                            <span class="scen-lbl">Best Case ROI (P95)</span>
                            <span class="scen-val val-green">{scen['Best_Case_ROI_Percent']}%</span>
                        </div>
                        <div class="scen-row">
                            <span class="scen-lbl">Expected ROI (P50)</span>
                            <span class="scen-val val-orange">{results['Expected_ROI_Percent']}%</span>
                        </div>
                        <div class="scen-row">
                            <span class="scen-lbl">Worst Case ROI (P05)</span>
                            <span class="scen-val val-red">{risk['Worst_Case_ROI_Percent']}%</span>
                        </div>
                    </div>
                    """
                    st.markdown(roi_html, unsafe_allow_html=True)

                with scen_col2:
                    payback_html = f"""
                    <div class="scen-card">
                        <h5 style="margin-top:0; color:#1f2937; font-family:'Syne', sans-serif;">Payback & Risk</h5>
                        <div class="scen-row">
                            <span class="scen-lbl">Shortest Payback (P10)</span>
                            <span class="scen-val val-green">{scen['Shortest_Payback_Years']} yrs</span>
                        </div>
                        <div class="scen-row">
                            <span class="scen-lbl">Longest Payback (P95)</span>
                            <span class="scen-val val-red">{risk['Worst_Case_Payback_Years']} yrs</span>
                        </div>
                        <div class="scen-row">
                            <span class="scen-lbl">Win Probability</span>
                            <span class="scen-val val-green">{scen['Probability_Positive_ROI']}%</span>
                        </div>
                    </div>
                    """
                    st.markdown(payback_html, unsafe_allow_html=True)

                st.write("")

                st.markdown("### 💡 Final Recommendation")
                rec_text = results["Recommendation"]

                if "Excellent" in rec_text:
                    st.success(f"**{rec_text}**")
                elif "Good" in rec_text:
                    st.info(f"**{rec_text}**")
                else:
                    st.warning(f"**{rec_text}**")

            with bottom_col2:
                st.markdown("### 🏢 Local Vendor Recommendations")
                st.write(f"Based on your location, here are reputed vendors for a **{system_size_kw}kW** system:")

                for vendor in results["Recommended_Local_Vendors"]:
                    card_html = f"""
                    <div class="vendor-card">
                        <div class="vendor-info">
                            <div class="vendor-name">{vendor['Name']}</div>
                            <div class="vendor-desc">📍 {vendor['Location']} &nbsp;|&nbsp; {vendor['Specialty']}</div>
                        </div>
                        <div class="vendor-contact">📞 {vendor['Contact']}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. EMPTY STATE DASHBOARD (Before click)
# ---------------------------------------------------------
else:
    st.info(
        "👈 **Welcome to Kinetic!** Adjust your system parameters in the sidebar and click **Run Financial Simulation** to generate your personalized report.")

    st.markdown("### 📊 Expected Financial Outcomes")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.markdown(
            """<div class="stat-card" style="border-top: 4px solid #1f2937;"><div class="s-lbl">Total Investment</div><div class="s-val">---</div><div class="s-sub">CAPEX (LKR)</div></div>""",
            unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(
            """<div class="stat-card" style="border-top: 4px solid #18a058;"><div class="s-lbl">Expected NPV</div><div class="s-val">---</div><div class="s-sub">Discounted (LKR)</div></div>""",
            unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(
            """<div class="stat-card" style="border-top: 4px solid #f4601a;"><div class="s-lbl">Expected ROI</div><div class="s-val">---</div><div class="s-sub">Over 20 Years</div></div>""",
            unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(
            """<div class="stat-card" style="border-top: 4px solid #d97706;"><div class="s-lbl">Payback Period</div><div class="s-val">---</div><div class="s-sub">Break-Even Point</div></div>""",
            unsafe_allow_html=True)
    with kpi_col5:
        st.markdown(
            """<div class="stat-card" style="border-top: 4px solid #9ca3af;"><div class="s-lbl">Risk Certainty</div><div class="s-val" style="color: #9ca3af;">---</div><div class="s-sub">Based on Monte Carlo</div></div>""",
            unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📈 Visual Risk & Cash Flow Analysis")
    tab1, tab2, tab3 = st.tabs(["Cumulative Cash Flow (Payback)", "Risk Distribution (NPV)", "Yearly Revenue"])

    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text="Awaiting Simulation Data...", xref="paper", yref="paper", showarrow=False,
                          font=dict(size=18, color="#9ca3af", family="DM Sans"))]
    )

    with tab1:
        st.plotly_chart(empty_fig, use_container_width=True)
    with tab2:
        st.plotly_chart(empty_fig, use_container_width=True)
    with tab3:
        st.plotly_chart(empty_fig, use_container_width=True)

    st.divider()

    bottom_col1, bottom_col2 = st.columns([1, 1])
    with bottom_col1:
        st.markdown("### ⚠️ Scenario Analysis")
        scen_col1, scen_col2 = st.columns(2)
        with scen_col1:
            st.markdown(
                """<div class="scen-card"><h5 style="margin-top:0; color:#1f2937; font-family:'Syne', sans-serif;">ROI Profiles</h5><div class="scen-row"><span class="scen-lbl">Best Case ROI (P95)</span><span class="scen-val">---</span></div><div class="scen-row"><span class="scen-lbl">Expected ROI (P50)</span><span class="scen-val">---</span></div><div class="scen-row"><span class="scen-lbl">Worst Case ROI (P05)</span><span class="scen-val">---</span></div></div>""",
                unsafe_allow_html=True)
        with scen_col2:
            st.markdown(
                """<div class="scen-card"><h5 style="margin-top:0; color:#1f2937; font-family:'Syne', sans-serif;">Payback & Risk</h5><div class="scen-row"><span class="scen-lbl">Shortest Payback (P10)</span><span class="scen-val">---</span></div><div class="scen-row"><span class="scen-lbl">Longest Payback (P95)</span><span class="scen-val">---</span></div><div class="scen-row"><span class="scen-lbl">Win Probability</span><span class="scen-val">---</span></div></div>""",
                unsafe_allow_html=True)
    with bottom_col2:
        st.markdown("### 🏢 Local Vendor Recommendations")
        st.markdown(
            """<div class="vendor-card"><div class="vendor-info"><div class="vendor-name" style="color: #9ca3af;">Awaiting Simulation Data</div><div class="vendor-desc">Run simulation to see certified local vendors</div></div></div>""",
            unsafe_allow_html=True)