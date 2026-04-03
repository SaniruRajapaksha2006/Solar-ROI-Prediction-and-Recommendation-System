<div align="center">

# ☀️ Solar ROI Prediction and Recommendation System

**An AI-driven decision-support system for residential solar PV investment in Sri Lanka**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![Streamlit 1.32](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io)

*Submitted in partial fulfilment of the requirements for the BSc (Hons) in Artificial Intelligence and Data Science — Informatics Institute of Technology in collaboration with Robert Gordon University, Aberdeen*

</div>

---

## Overview

Sri Lanka targets **70% renewable energy by 2030**, yet residential solar adoption sits below 2% of total households. Homeowners face financial uncertainty, grid infrastructure constraints, and a complete absence of localized, data-driven decision tools. This system addresses all three challenges through four integrated AI/ML components that guide a homeowner from raw location and usage data to a confident, risk-informed investment decision.

### Problem Statement

> *The absence of a holistic AI-centred system capable of predicting solar generation, assessing grid suitability, predicting consumption on PUCSL tariffs and performing risk-conscious financial modelling is inhibiting effective residential solar investment decision-making in Sri Lanka.*

---

## Research Objectives

| # | Objective |
|---|-----------|
| RO1 | Predict the monthly solar energy (kWh) a rooftop system can generate at the user's location |
| RO2 | Map the nearest electricity transformer and assess its suitability for solar connection |
| RO3 | Forecast user electricity consumption and estimate monthly bills under PUCSL tariff structures |
| RO4 | Predict financial viability — ROI, payback period, and risk factors for solar investment profitability |

---

## System Components

| # | Component | Model | Key Metric |
|---|-----------|-------|------------|
| 1 | **Solar Generation Forecasting** | RandomForest Regression | MAE 18.33 kWh/kW — 72.4% better than physics baseline |
| 2 | **Geospatial & Transformer Suitability** | RandomForest Classifier | 100% accuracy on 10-transformer test dataset |
| 3 | **Electricity Consumption Forecasting** | LSTM (optimized) | MAE 58.70 kWh, R² 0.785, 93.8% within ±150 kWh |
| 4 | **Risk-Aware Financial Modeling** | Monte Carlo (2,000 iterations) | Convergence variance < 0.1%, stable by 500 iterations |

### How the components connect

```
User inputs (location, 3–4 months electricity bills)
        │
        ▼
[1] Solar Generation Forecasting
    → Predicted monthly kWh/kW output for 12 months
        │
        ▼
[2] Geospatial & Transformer Suitability
    → Nearest transformer + grid compatibility score + recommended panel size
        │
        ▼
[3] Electricity Consumption Forecasting & Tariff Integration
    → 12-month consumption profile + PUCSL D1 tariff bills
        │
        ▼
[4] Risk-Aware Financial Model
    → ROI, payback period, NPV, Monte Carlo risk bands
        │
        ▼
    Integrated Dashboard  ←  integrated_dashboard.py
```

---

## Project Structure

```
Solar-ROI-Prediction-and-Recommendation-System/
│
├── Solar Generation Forecasting Model/          # Component 1
│   ├── config.yaml                              # Single source of truth
│   ├── data_pipeline.py                         # ETL pipeline
│   ├── model_trainer.py                         # Model training & tuning
│   ├── predict.py                               # Inference CLI
│   ├── src/
│   │   ├── data/           loader.py, splitter.py
│   │   ├── features/       engineering.py, selection.py
│   │   ├── models/         similarity_engine.py
│   │   ├── preprocessing/  missing.py, outliers.py
│   │   └── training/       baseline.py, evaluator.py, saver.py, tuner.py
│   ├── utils/
│   │   ├── nasa_power.py                        # NASA POWER API client
│   │   ├── fetch_future_weather.py              # C3S SEAS5 forecast
│   │   └── utils_config.py
│   ├── notebooks/
│   │   ├── 00_raw_data_eda.ipynb
│   │   ├── 01_feature_relationship_eda.ipynb
│   │   └── 02_model_evaluation_eda.ipynb
│   └── models/best_solar_pipeline.pkl
│
├── Geospatial Mapping & Transformer Suitability Model/   # Component 2
│   ├── app.py                                   # Streamlit UI
│   ├── backend/
│   │   ├── models/         ml_models.py, suitability_engine.py
│   │   └── utils/          data_preprocessor.py, geo_utils.py
│   ├── test_models.py
│   └── test_suitability.py
│
├── Electricity Consumption Forecasting & Tariff Integration Model/  # Component 3
│   ├── main.py                                  # CLI entry point
│   ├── api/fastapi_app.py                       # REST API service
│   ├── src/
│   │   ├── data_loader.py                       # CSV loading & profiles
│   │   ├── similarity_matcher.py                # Weighted similarity scoring
│   │   ├── pattern_extractor.py                 # Pattern-based fallback
│   │   ├── forecaster.py                        # Ensemble forecaster
│   │   ├── tariff_calculator.py                 # PUCSL D1 tariff
│   │   └── utils.py
│   ├── models/
│   │   ├── lstm_model.py
│   │   └── saved/                               # Trained model artefacts
│   ├── features/
│   │   ├── feature_engineer.py                  # 30-feature engineering
│   │   ├── cyclical_encoder.py                  # Sin/cos time encoding
│   │   └── weather_integrator.py                # NASA POWER integration
│   ├── validation/                              # Cross-validation utilities
│   ├── monitoring/drift_detector.py             # Concept drift detection
│   ├── config/config.yaml
│   └── Dockerfile, docker-compose.yml
│
├── Return of Investment, Payback and Risk-Aware Financial Model/    # Component 4
│   ├── Backend/
│   │   ├── financeml.py                         # Monte Carlo simulation
│   │   ├── market_data.json                     # Vendor pricing & tariffs
│   │   ├── update_live_data.py                  # Web scraping + API updates
│   │   └── unit_test.py
│   └── Frontend/
│       ├── index.py                             # Streamlit UI
│       └── style.css
│
├── shared/models.py                             # Pydantic models for integration
├── integrated_dashboard.py                      # ★ MAIN ENTRY POINT
├── main_integrated.py                           # Orchestrator
├── dashboard_styles.css
├── DATA_SETUP.md
├── pyproject.toml                               # UV package manager
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv) (recommended) or pip
- Dataset — see [Data Availability](#data-availability)

### Installation

```bash
# Clone the repository
git clone https://github.com/SaniruRajapaksha2006/Solar-ROI-Prediction-and-Recommendation-System.git
cd Solar-ROI-Prediction-and-Recommendation-System

# Install dependencies with UV (recommended)
uv sync

# OR with pip
pip install -r requirements.txt
```

### Run the integrated dashboard

```bash
streamlit run integrated_dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Data Availability

The electricity consumption and grid datasets used in this project were obtained from LECO/CEB under a confidentiality agreement and **cannot be shared publicly**. These files contain proprietary utility data and may include personally identifiable household information.

To reproduce the system using alternative data, the following open sources can be substituted:

| Data Type | Suggested Source |
|-----------|-----------------|
| Solar irradiance & weather | [NASA POWER API](https://power.larc.nasa.gov/) |
| Future weather forecasts | [C3S SEAS5 — Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) |
| Transformer locations | [OpenStreetMap](https://www.openstreetmap.org/) |
| Sri Lanka weather data | Sri Lanka Meteorological Department |

If you have access to equivalent utility data, place the processed files at:

```
processed/
├── MASTER_DATASET_ALL_10TRANSFORMERS.csv
└── MASTER_DATASET_RESIDENTIAL_ONLY.csv
```

Full dataset structure and column descriptions are documented in [`DATA_SETUP.md`](DATA_SETUP.md).

> ⚠️ Do not commit any CSV or utility data files to GitHub. They are listed in `.gitignore`.

---

## Running Tests

```bash
# Component 1 — data pipeline & model training
cd "Solar Generation Forecasting Model"
python data_pipeline.py
python model_trainer.py

# Component 2 — model & suitability evaluation
cd "Geospatial Mapping & Transformer Suitability Model"
python test_models.py

# Component 3 — full test suite
cd "Electricity Consumption Forecasting & Tariff Integration Model"
pytest tests/test_all.py -v

# Component 4 — unit tests
cd "Return of Investment, Payback and Risk-Aware Financial Model/Backend"
python -m pytest unit_test.py -v
```

---

## Model Performance

### Component 1 — Solar Generation Forecasting

Multiple regression models were benchmarked against a physics-based formula baseline:

| Model | MAE (kWh/kW) | R² |
|-------|-------------|-----|
| Physics Formula (baseline) | 66.58 | 0.029 |
| Similarity Match | 39.60 | 0.428 |
| Gradient Boosting | 18.43 | 0.729 |
| **RandomForest** *(selected)* | **18.33** | **0.731** |

### Component 2 — Geospatial & Transformer Suitability

A RandomForest classifier powers a multi-criteria scoring engine that weighs distance, load balance, and network constraints to rank transformers from most to least suitable. The model achieved **100% classification accuracy** on the 10-transformer test dataset. The recommended panel size output from this component is passed directly to Component 4.

### Component 3 — Electricity Consumption Forecasting

An optimized LSTM model forecasts 12-month household consumption by matching user input profiles to similar households via weighted similarity scoring. PUCSL D1 increasing-block tariff rates are then applied for per-slab monthly bill estimates.

| Metric | Value |
|--------|-------|
| MAE | 58.70 kWh |
| R² | 0.785 |
| Predictions within ±150 kWh | 93.8% |

### Component 4 — Risk-Aware Financial Model

Monte Carlo simulation (2,000 iterations) models uncertainty across solar irradiance variability, panel degradation, and dynamic electricity prices to produce distributions rather than point estimates for ROI and payback.

| Metric | Value |
|--------|-------|
| Convergence variance | < 0.1% |
| Stable from | 500 iterations |
| Outputs | ROI, payback period, NPV, risk confidence bands |

---

## Technology Stack

### Core

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Primary language |
| Streamlit | 1.32 | Dashboard UI |
| FastAPI | — | REST API (Component 3) |

### Machine Learning

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow / Keras | 2.19 / 3.10 | LSTM models |
| Scikit-learn | 1.8 | RandomForest, preprocessing |
| NumPy | 2.4 | Numerical computation |
| Pandas | 3.0 | Data manipulation |

### Geospatial & Visualisation

| Library | Purpose |
|---------|---------|
| GeoPandas | GIS operations |
| Folium | Interactive transformer maps |
| Plotly | Charts & confidence bands |
| Matplotlib / Seaborn | EDA & model evaluation |

---

## Methodology

### Research Approach

The project adopts a **pragmatic, mixed-methods** philosophy — combining quantitative AI model evaluation with qualitative stakeholder feedback. A deductive approach tests hypotheses derived from prior literature (e.g. benchmarking forecasting accuracy). Explainable AI techniques (SHAP) are applied to interpret key variable influences and build user trust. The time horizon is cross-sectional, using historical solar irradiance and electricity consumption datasets reflective of current climatic and economic conditions.

### Development

- **Lifecycle:** Agile with Scrum — short sprints per module with continuous testing and supervisor review
- **Design:** Object-Oriented Analysis and Design (OOAD) — each component is an independent, reusable module
- **Evaluation:** MAE, RMSE, R² for forecasting accuracy; comparative benchmarking against real-world data and published Sri Lanka studies; user usability surveys with CEB engineers and potential homeowners

### Project Scope

**In scope:** Solar generation prediction, transformer suitability mapping, consumption forecasting with PUCSL tariff integration, Monte Carlo financial modeling, integrated Streamlit dashboard — all localized for Sri Lankan residential users.

**Out of scope:** Real-time grid deployment, live CEB/LECO data feeds, hardware-level panel control, industrial or utility-scale forecasting, national energy policy optimization.

---

## System Outputs

| Output | Format | Location |
|--------|--------|----------|
| Integrated report | JSON | `results/integrated_report.json` |
| PDF report | PDF | Download via dashboard |
| Consumption forecast | JSON / TXT | `results/[timestamp]/` |
| Solar forecast | Console | Terminal output |
| Transformer map | Interactive HTML | Dashboard — Geospatial tab |
| Vendor comparison | Table | Dashboard — ROI tab |

---

## Related Work

| Study | Method | Gap addressed by this project |
|-------|--------|-----------------------------|
| Fernando et al. (2018) | LSTM | Jaffna-specific; not nationally generalizable |
| Mulenga & Etherden (2023) | Monte Carlo | Requires accurate CEB grid data; no consumption link |
| Shiwakoti et al. (2025) | ARIMA, LSTM | Short-term only; no tariff or ROI integration |
| Nascimento et al. (2025) | Time-series + clustering | Non-Sri Lanka context |
| Senevirathne (2018) | ROI / payback modeling | No risk analysis; outdated tariff data |

This project is the first integrated, Sri Lanka–specific system to combine all four dimensions — solar forecasting, geospatial grid assessment, consumption prediction with PUCSL tariff integration, and stochastic financial risk modeling — into a single residential decision-support platform.

---

## Team

| Name | Component | Student ID |
|------|-----------|-----------|
| Saniru Rajapaksha | Consumption Forecasting + Integration | 2425606 |
| Sasiri Akalanka | Solar Generation Forecasting | 2425481 |
| Dewmi Tharunya | Geospatial Mapping | 2425603 |
| Gangamini Harshitha | ROI Financial Model | 2425576 |

**Supervisor:** Mr. Prashan Rathnayaka  
**Institution:** Informatics Institute of Technology (IIT) in collaboration with Robert Gordon University, Aberdeen

---

## Acknowledgements

- **Mr. Prashan Rathnayaka** - Project Supervisor for guidance, feedback, and support throughout the project
- **Dr. Prasan Yapa** - Module Leader
- **Ms. Sulari Fernando** - Tutorial Leader

---

## References

- Asian Development Bank & UNDP (2017). *100% electricity generation through renewable energy by 2050 — Assessment of Sri Lanka's Power Sector.* https://www.adb.org/sites/default/files/publication/354591/sri-lanka-power-2050v2.pdf
- Bandara, T.S.M. & Amarasena, T.S.M. (2020). Factors influencing solar energy technology adoption by households in Western Province Sri Lanka. *Vidyodaya Journal of Management*, 6(2), 131–152.
- Fernando, W.L.M. et al. (2018). *Solar irradiance forecasting using deep learning approaches.* BSc dissertation, University of Jaffna.
- International Energy Agency (2023). *Energy system of Sri Lanka.* https://www.iea.org/countries/sri-lanka
- Mulenga, E. & Etherden, N. (2023). Multiple distribution networks hosting capacity assessment using a stochastic approach. *Sustainable Energy, Grids and Networks*, 36, 101170.
- Nascimento, A.J.P. et al. (2025). Integrated time series analysis, clustering, and forecasting for energy efficiency optimization and tariff management. *IEEE Access*, 13, 59309–59323.
- Senevirathne, S.R.M.P. (2018). Profitability assessment of solar PV installations in Sri Lankan residential buildings. *7th World Construction Symposium*, Colombo.
- Shiwakoti, R.K., Limcharoen, P. & Uduwage, D.N.L.S. (2025). Short-term electricity demand forecasting in Sri Lanka using statistical and deep learning models. *World Construction Symposium*, Colombo, 1260–1272.
- World Bank (2025). *Fossil fuel energy consumption (% of total) – Sri Lanka.* https://data.worldbank.org/indicator/EG.USE.COMM.FO.ZS?locations=LK

---

## License

This project is submitted in partial fulfilment of the requirements for the **BSc (Hons) in Artificial Intelligence and Data Science** degree. All rights reserved by the authors and the Informatics Institute of Technology.

---

<div align="center">
  <sub>Built for Sri Lanka's renewable energy future 🌿</sub>
</div>