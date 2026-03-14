import streamlit as st
import sys
import os

# ---------------------------------------------------------
# ROBUST BACKEND CONNECTION (Adapted to your folder structure)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # The Frontend folder
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # The Root folder
backend_dir = os.path.join(parent_dir, 'Backend')  # The Backend folder

# Add the Backend folder to Python's system path so it can find financeml.py
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import the model exactly as it is named in your directory
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
                # We show high/moderate certainty based on the risk profile
                certainty = results['Risk_Analysis']['Certainty_Score']
                st.metric(label="Risk Certainty", value=certainty)

            st.divider()

            # --- SECTION 2: CHARTS SKELETON ---
            st.markdown("### 📈 Visual Risk & Cash Flow Analysis")
            tab1, tab2, tab3 = st.tabs(["Cumulative Cash Flow (Payback)", "Risk Distribution (NPV)", "Yearly Revenue"])

            with tab1:
                st.write("*Line Chart Placeholder*")
            with tab2:
                st.write("*Histogram Placeholder*")
            with tab3:
                st.write("*Bar Chart Placeholder*")

            st.divider()

            # --- SECTION 3: BOTTOM ROW SKELETON ---
            bottom_col1, bottom_col2 = st.columns([1, 1])

            with bottom_col1:
                st.markdown("### ⚠️ Scenario Analysis")
                st.write("*Scenario Metrics Placeholder*")
                st.write("*Recommendation Badge Placeholder*")

            with bottom_col2:
                st.markdown("### 🏢 Local Vendor Recommendations")
                st.write("*Vendor Cards Placeholder*")

else:
    st.info("👈 Please adjust the parameters in the sidebar and click **Run Financial Simulation**.")