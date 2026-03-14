import streamlit as st
import sys
import os
import plotly.graph_objects as go

# ---------------------------------------------------------
# ROBUST BACKEND CONNECTION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
backend_dir = os.path.join(parent_dir, 'Backend')

if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from financeml import SolarFinancialModel


@st.cache_resource
def get_financial_model():
    return SolarFinancialModel()


def run_roi_analysis(size, gen, cons):
    model = get_financial_model()
    return model.calculate_financial_report(size, gen, cons)


# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Kinetic | Financial Intelligence", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------
# 2. MAIN DASHBOARD HEADER
# ---------------------------------------------------------
st.title("⚡ Kinetic: AI-Driven Solar ROI & Risk-Aware Dashboard")
st.markdown(
    "Evaluate the financial feasibility, risks, and payback of your residential solar PV investment in Sri Lanka.")
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

            # --- SECTION 1: TOP KPI METRICS ---
            st.markdown("### 📊 Expected Financial Outcomes")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

            with kpi_col1:
                st.metric(label="Total Investment (CAPEX)", value=f"LKR {results['Total_Investment_LKR']:,.0f}")
            with kpi_col2:
                st.metric(label="Expected NPV", value=f"LKR {results['Expected_NPV_LKR']:,.0f}")
            with kpi_col3:
                st.metric(label="Expected ROI", value=f"{results['Expected_ROI_Percent']}%")
            with kpi_col4:
                st.metric(label="Payback Period", value=f"{results['Expected_Payback_Years']} yrs")
            with kpi_col5:
                certainty = results['Risk_Analysis']['Certainty_Score']
                st.metric(label="Risk Certainty", value=certainty)

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
                                     yaxis_title="Cumulative Cash (LKR)", hovermode="x unified")
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
                                      xaxis_title="Net Present Value (LKR)",
                                      yaxis_title="Frequency (Number of Scenarios)", bargap=0.1)
                st.plotly_chart(fig_npv, use_container_width=True)

            with tab3:
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(x=years[1:], y=chart_data["Yearly_Revenue_Forecast"], marker_color='#d97706',
                                         name='Annual Net Cashflow'))
                fig_rev.update_layout(title="Expected Annual Net Cashflow (With Degradation & Maintenance)",
                                      xaxis_title="Year", yaxis_title="Net Cashflow (LKR)", hovermode="x unified",
                                      xaxis=dict(tickmode='linear', dtick=1))
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
                    st.markdown("##### ROI Profiles")
                    st.metric(label="Best Case ROI (P95)", value=f"{scen['Best_Case_ROI_Percent']}%")
                    st.metric(label="Expected ROI (P50)", value=f"{results['Expected_ROI_Percent']}%")
                    st.metric(label="Worst Case ROI (P05)", value=f"{risk['Worst_Case_ROI_Percent']}%")

                with scen_col2:
                    st.markdown("##### Payback & Risk")
                    st.metric(label="Shortest Payback (P10)", value=f"{scen['Shortest_Payback_Years']} yrs")
                    st.metric(label="Longest Payback (P95)", value=f"{risk['Worst_Case_Payback_Years']} yrs")
                    st.metric(label="Win Probability (ROI > 0)", value=f"{scen['Probability_Positive_ROI']}%")

                st.write("")  # Spacer

                st.markdown("### 💡 Final Recommendation")
                rec_text = results["Recommendation"]

                if "Excellent" in rec_text:
                    st.success(f"**{rec_text}**")
                elif "Good" in rec_text:
                    st.info(f"**{rec_text}**")
                else:
                    st.warning(f"**{rec_text}**")

            # NEW FOR COMMIT 27: Local Vendor Recommendation Cards
            with bottom_col2:
                st.markdown("### 🏢 Local Vendor Recommendations")
                st.write(f"Based on your location, here are reputed vendors for a **{system_size_kw}kW** system:")

                for vendor in results["Recommended_Local_Vendors"]:
                    with st.container(border=True):
                        st.markdown(f"**{vendor['Name']}**")
                        st.markdown(f"📍 {vendor['Location']} | 📞 {vendor['Contact']}")
                        st.caption(f"Specialty: {vendor['Specialty']}")

else:
    st.info("👈 Please adjust the parameters in the sidebar and click **Run Financial Simulation**.")