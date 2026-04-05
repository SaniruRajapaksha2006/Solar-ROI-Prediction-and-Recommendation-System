<div align="center">

# Component 3 — Electricity Consumption Forecasting & Tariff Integration Model

**Part of the AI-Driven Solar ROI Prediction and Recommendation System**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![FastAPI 0.104](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

</div>

---

## Overview

A production-ready electricity consumption forecasting engine for Sri Lankan residential users. Predicts 12-month household electricity consumption under PUCSL D1 tariff structures using an ensemble of LSTM neural networks and pattern-based similarity matching.

**Why this matters:** Sri Lanka's tariffs follow an increasing-block structure where rates rise steeply with consumption. Accurate forecasting is essential for calculating solar ROI correctly, optimizing system sizing, and estimating bill savings under net metering.

---

## Architecture

```
User Input (location + 3–4 months of bills)
                    │
                    ▼
        Data Loader & Quality Check
         (dataset, quality monitoring)
                    │
                    ▼
           Similarity Matcher
      (find similar households — no look-ahead)
                    │
                    ▼
         Feature Engineering
  (cyclical encoding, weather, lags, rolling stats)
            │               │
            ▼               ▼
     LSTM Forecaster   Pattern Extractor
      (primary)          (fallback)
            │               │
            └──────┬────────┘
                   ▼
          Ensemble Forecaster
       (weighted combination)
                   │
                   ▼
          Tariff Calculator
        (PUCSL D1 + net metering)
                   │
                   ▼
       Output → Component 4
  (forecast + bills + uncertainty bands)
```

---

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| Test MAE | 58.70 kWh | < 60 kWh |
| Test RMSE | 75.01 kWh | — |
| Test MAPE | 13.44% | < 15% |
| Test R² | 0.785 | > 0.75 |
| Predictions within ±150 kWh | 93.8% | — |
| Inference time | < 1 second/user | — |

### Model comparison

| Model | MAE (kWh) |
|-------|-----------|
| Persistence baseline | 503.80 |
| Historical average | 153.94 |
| Random Forest | 62.25 |
| XGBoost | 65.45 |
| Simple LSTM | 394.38 |
| **Optimal LSTM** *(selected)* | **58.70** |

---

## Project Structure

```
Electricity Consumption Forecasting & Tariff Integration Model/
│
├── main.py                          # CLI entry point
├── data_quality.py                  # Data quality monitoring
│
├── api/
│   └── fastapi_app.py               # REST API service
│
├── src/
│   ├── data_loader.py               # CSV loading & customer profiles
│   ├── similarity_matcher.py        # Weighted similarity scoring
│   ├── pattern_extractor.py         # Pattern-based fallback (safe mode)
│   ├── forecaster.py                # Ensemble forecaster
│   ├── tariff_calculator.py         # PUCSL D1 tariff + net metering
│   └── utils.py                     # Shared utilities
│
├── models/
│   ├── lstm_model.py                # LSTM implementation
│   └── saved/
│       ├── lstm_model.h5
│       ├── scaler.pkl
│       ├── feature_info.json
│       └── model_config.json
│
├── features/
│   ├── feature_engineer.py          # 30-feature engineering
│   ├── cyclical_encoder.py          # Sin/cos cyclical encoding
│   └── weather_integrator.py        # NASA POWER API integration
│
├── validation/
│   ├── model_validator.py           # Metrics & cross-validation
│   └── time_series_split.py         # Temporal splitting
│
├── monitoring/
│   └── drift_detector.py            # Concept drift detection
│
├── config/
│   └── config.yaml                  # Component configuration
│
├── scripts/
│   ├── train_model.py               # Model training script
│   └── evaluate_model.py            # Evaluation script
│
├── tests/
│   └── test_all.py                  # Pytest suite
│
├── results/
│   └── YYYYMMDD_HHMMSS/
│       ├── component3_results.json
│       ├── electricity_bills.txt
│       └── for_component4.json
│
├── logs/
│   ├── component3.log
│   └── training.log
│
├── cache/weather/                   # Cached NASA POWER data
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Dataset — see `DATA_SETUP.md` in the project root

### Installation

```bash
cd "Electricity Consumption Forecasting & Tariff Integration Model"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
# or
uv sync
```

### Configure and load data

Edit `config/config.yaml` with your dataset path (default: `../processed/MASTER_DATASET_ALL_10TRANSFORMERS.csv`), then optionally train the LSTM model:

```bash
python scripts/train_model.py
```

If no trained model is found, the system automatically falls back to pattern-based forecasting.

---

## Usage

### Command line

```bash
# Single user forecast
python main.py --mode single --lat 6.9271 --lon 79.8612 \
  --months '{"9":350.5,"10":420.2,"11":380.1}'

# Batch processing
python main.py --mode batch --input_file users.json --output_file results.json

# Train LSTM model
python scripts/train_model.py --epochs 100 --batch_size 32

# Evaluate model with cross-validation and baselines
python scripts/evaluate_model.py \
  --model_path models/saved/lstm_model.h5 --cv --baselines
```

### Python API

```python
from src.data_loader import ElectricityDataLoader
from src.similarity_matcher import SimilarityMatcher
from src.forecaster import EnsembleForecaster
from src.tariff_calculator import PUCsLTariffCalculator
from src.utils import load_config

config = load_config('config/config.yaml')

data_loader = ElectricityDataLoader(config)
data_loader.load_dataset()

similarity  = SimilarityMatcher(data_loader, config)
forecaster  = EnsembleForecaster(config)
tariff      = PUCsLTariffCalculator(config)

user_data = {
    'latitude': 6.9271,
    'longitude': 79.8612,
    'consumption_months': {9: 350.5, 10: 420.2, 11: 380.1},
    'tariff': 'D1',
    'phase': 'SP',
    'has_solar': 0,
    'household_size': 4
}

similar  = similarity.find_similar_households_safe(user_data)
forecast = forecaster.forecast(user_data, similar, {})
bills    = tariff.calculate_annual_bills(
               forecast['forecast']['monthly_values'], 'D1')

print(f"Annual consumption : {forecast['forecast']['statistics']['annual_total']:.0f} kWh")
print(f"Annual bill        : Rs. {bills['annual_summary']['total_bill_lkr']:,.0f}")
```

### REST API

```bash
python api/fastapi_app.py
# Runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/forecast` | POST | Generate 12-month consumption forecast |
| `/tariff/calculate` | POST | Calculate electricity bill |
| `/similar/accounts` | GET | Find similar households |
| `/models/info` | GET | Model metadata |

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 6.9271,
    "longitude": 79.8612,
    "consumption_months": [
      {"month": 9,  "consumption": 350.5},
      {"month": 10, "consumption": 420.2},
      {"month": 11, "consumption": 380.1}
    ],
    "tariff": "D1",
    "phase": "SP",
    "has_solar": 0,
    "household_size": 4
  }'
```

---

## Core Algorithms

### Similarity matching

Households are matched using a weighted scoring approach with no look-ahead bias:

| Component | Weight | Description |
|-----------|--------|-------------|
| Consumption pattern | 0.60 | Magnitude (80%) + shape (20%) |
| Geographic | 0.15 | Within 2 km radius |
| Technical | 0.15 | Tariff, phase, solar status |
| Seasonal compatibility | 0.10 | Trend alignment |

### Feature engineering (30 features)

- **Seasonal** — `is_wet_season`, `is_intermediate` (Sri Lanka monsoon calendar)
- **Lag features** — `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`
- **Rolling statistics** — `rolling_mean_3`, `rolling_std_3`, and variants
- **Differenced** — `diff_1`, `diff_12`, `pct_change_1`
- **Ratios** — `consumption_vs_avg`, `consumption_vs_rolling_3`
- **Static** — `INV_CAPACITY`, `DISTANCE_FROM_TF_M`, `phase_SP`

### Ensemble forecasting

```
Final forecast = LSTM (10%) + Pattern-based matching (90%)
```

The pattern-based component dominates to reduce instability on sparse historical data. Confidence scoring is applied per month.

### PUCSL D1 tariff structure (2025)

| Block (kWh) | Rate (LKR/kWh) |
|-------------|----------------|
| 0 – 30 | 8.00 |
| 31 – 60 | 10.00 |
| 61 – 90 | 16.00 |
| 91 – 120 | 50.00 |
| 121 – 180 | 75.00 |
| 181+ | 100.00 |

Fixed charge: LKR 180/month · Fuel adjustment: 18% · VAT: 18%

---

## Sri Lanka Seasonal Factors

| Month | Factor | Season |
|-------|--------|--------|
| January | 0.85 | NE Monsoon |
| February | 0.90 | NE Monsoon |
| March | 1.15 | Dry Season |
| April | 1.25 | Peak (Avurudu) |
| May | 1.20 | SW Monsoon |
| June – July | 0.85–1.00 | SW Monsoon |
| August | 0.90 | SW Monsoon |
| September | 1.00 | Inter-monsoon |
| October | 1.10 | Dry Season |
| November | 1.05 | Dry Season |
| December | 0.95 | NE Monsoon |

---

## Output Format

Results are written per run to `results/YYYYMMDD_HHMMSS/` and passed to Component 4:

```json
{
  "consumption_forecast": {
    "monthly_kwh": {"1": 282.2, "2": 278.2},
    "monthly_confidence": {"1": 0.90, "2": 0.85},
    "uncertainty_ranges": {
      "1": {"lower_bound": 225.8, "upper_bound": 338.6}
    },
    "annual_total_kwh": 3718.9,
    "overall_confidence": 0.87
  },
  "billing_analysis": {
    "monthly_bills": {"1": 12450},
    "annual_summary": {
      "total_bill_lkr": 145200,
      "effective_rate_lkr_per_kwh": 32.12
    }
  }
}
```

| File | Contents |
|------|----------|
| `component3_results.json` | Full structured results |
| `electricity_bills.txt` | Human-readable bill breakdown |
| `for_component4.json` | Data passed to the ROI calculator |

---

## Monitoring & Drift Detection

The system monitors for concept drift in production:

```python
from monitoring.drift_detector import ConceptDriftDetector

detector = ConceptDriftDetector(config)
result = detector.detect_all(y_true, y_pred, features, clusters, timestamp)
# Returns: {'drift_detected': bool, 'alerts': [...]}
```

| Check | Threshold |
|-------|-----------|
| Prediction error drift | z-score > 2.0 |
| Feature distribution drift | KS test p < 0.05 |
| Cluster stability | Jaccard similarity < 0.8 |

---

## Testing

```bash
# Run all tests
pytest tests/test_all.py -v

# Run with coverage report
pytest tests/test_all.py --cov=. --cov-report=html
```

---

## Data Privacy

All personal data is anonymized before use. No personally identifiable information (PII) is stored in logs or output files. The system complies with Sri Lanka data protection requirements, and the underlying LECO/CEB dataset is held under a confidentiality agreement and is not included in this repository.

---

## References

- Shiwakoti, R.K., Limcharoen, P. & Uduwage, D.N.L.S. (2025). Short-term electricity demand forecasting in Sri Lanka using statistical and deep learning models. *World Construction Symposium*, Colombo, 1260–1272.
- Fu, X. et al. (2018). Clustering-based load forecasting under increasing-block tariffs.
- PUCSL (2025). *Domestic tariff structure D1.*

---

## Author

**Component lead:** Saniru Rajapaksha  
**Project:** AI-Driven Solar ROI Prediction and Recommendation System  
**Institution:** Informatics Institute of Technology (IIT) in collaboration with Robert Gordon University, Aberdeen  
**Repository:** [SaniruRajapaksha2006/Solar-ROI-Prediction-and-Recommendation-System](https://github.com/SaniruRajapaksha2006/Solar-ROI-Prediction-and-Recommendation-System)

---

**Acknowledgements:** Lanka Electricity Company (LECO) for the dataset · PUCSL for tariff information · NASA POWER for weather data

---

<div align="center">
  <sub>Component 3 — Electricity Consumption Forecasting & Tariff Integration</sub>
</div>