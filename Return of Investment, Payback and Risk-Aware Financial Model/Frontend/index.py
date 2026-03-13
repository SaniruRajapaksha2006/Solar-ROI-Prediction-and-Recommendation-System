import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solar ROI Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. MAIN DASHBOARD HEADER
# ---------------------------------------------------------
st.title("☀️ AI-Driven Solar ROI & Recommendation System")
st.markdown("Evaluate the financial feasibility, risks, and payback of your residential solar PV investment in Sri Lanka.")
st.divider()

# ---------------------------------------------------------
# 3. SIDEBAR: USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("⚙️ System Inputs")
st.sidebar.markdown("Configure the parameters predicted by the AI models.")

# Input fields for the data your backend requires
system_size_kw = st.sidebar.number_input(
    "Recommended System Size (kW)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Output from the Geospatial Grid Component"
)

annual_gen_kwh = st.sidebar.number_input(
    "Predicted Annual Generation (kWh)",
    min_value=1000,
    max_value=50000,
    value=7200,
    step=100,
    help="Output from the LSTM Solar Forecasting Component"
)

annual_cons_kwh = st.sidebar.number_input(
    "Predicted Annual Consumption (kWh)",
    min_value=1000,
    max_value=50000,
    value=4800,
    step=100,
    help="Output from the Electricity Consumption Component"
)

st.sidebar.divider()

# The trigger button
calculate_btn = st.sidebar.button("📊 Run Financial Simulation", use_container_width=True)

# ---------------------------------------------------------
# 4. PLACEHOLDER CONTENT AREA
# ---------------------------------------------------------
if calculate_btn:
    st.info("Simulation triggered! (Backend integration coming in the next commit...)")
else:
    st.write("👈 Please adjust the parameters in the sidebar and click **Run Financial Simulation** to view your personalized report.")