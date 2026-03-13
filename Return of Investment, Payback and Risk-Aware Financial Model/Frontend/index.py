import streamlit as st
import sys
import os

# Allow Streamlit to import your backend controller from the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from financial_controller import run_roi_analysis

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Solar ROI Predictor", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# 2. MAIN DASHBOARD HEADER
# ---------------------------------------------------------
st.title("☀️ AI-Driven Solar ROI & Recommendation System")
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
# 4. BACKEND EXECUTION & RAW RESPONSE
# ---------------------------------------------------------
if calculate_btn:
    with st.spinner("Running 2,000 Monte Carlo Risk Simulations..."):
        # Call your actual Python backend!
        results = run_roi_analysis(system_size_kw, annual_gen_kwh, annual_cons_kwh)

        if "error" in results:
            st.error(f"An error occurred: {results['error']}")
        else:
            st.success(
                f"Analysis Complete: {system_size_kw}kW System | {annual_gen_kwh:,.0f} kWh/yr | 2000 Simulations")

            # Temporarily dump the raw JSON to the screen to prove the connection works
            st.write("### Raw Backend Output (To be formatted in next commits):")
            st.json(results)

else:
    st.write("👈 Please adjust the parameters in the sidebar and click **Run Financial Simulation**.")