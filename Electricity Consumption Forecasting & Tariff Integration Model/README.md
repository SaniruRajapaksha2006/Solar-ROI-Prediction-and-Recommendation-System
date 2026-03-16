# Component 3: Electricity Consumption Forecasting for Sri Lanka

## Overview

This component forecasts residential electricity consumption for Sri Lankan households using an ensemble approach combining LSTM neural networks and pattern-based methods. It is part of the larger AI-Driven Solar ROI Prediction and Recommendation System.

### Key Features

- **Ensemble forecasting**: LSTM as primary, pattern-based as fallback
- **No look-ahead bias**: All methods use only historical data
- **Sri Lanka specific**: Custom seasonal adjustments, holiday effects
- **Net metering support**: Full PUCSL tariff implementation
- **Comprehensive validation**: Cross-validation, baselines, statistical tests

---

## Architecture

```
+-------------------------------------------------------------+
|                        User Input                           |
|               (Location + 3-4 months of bills)              |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Data Loader & Quality                     |
|              (SQLite database, quality monitoring)           |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    Similarity Matcher                       |
|          (Find similar households - no look-ahead)          |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Feature Engineering                       |
|    (Cyclical encoding, weather, lags, rolling stats)        |
+-----------+-------------------------------------+-----------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
|    LSTM Forecaster    |             |   Pattern Extractor   |
|    (Primary method)   |             |   (Fallback method)   |
+-----------+-----------+             +-----------+-----------+
            +------------------+------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Ensemble Forecaster                      |
|             (Weighted combination of methods)               |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    Tariff Calculator                        |
|          (PUCSL rates + Net metering)                       |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    Output for Component 4                   |
|              (Forecast + Bills + Uncertainty)               |
+-------------------------------------------------------------+
```

---

## Project Structure

```
Electricity Consumption Forecasting & Tariff Integration Model/
├── api/
│   ├── __init__.py
│   └── fastapi_app.py               # REST API
├── cache/
│   └── weather/                     # Weather data cache
├── config/
│   └── config.yaml                  # Configuration file
├── features/
│   ├── __init__.py
│   ├── cyclical_encoder.py          # Cyclical encoding
│   ├── feature_engineer.py          # Feature engineering
│   └── weather_integrator.py        # Weather API integration
├── logs/                            # Log files
├── models/
│   ├── saved/                       # Saved model weights
│   ├── __init__.py
│   └── lstm_model.py                # LSTM implementation
├── monitoring/
│   ├── __init__.py
│   └── drift_detector.py            # Concept drift detection
├── results/                         # Output results
├── scripts/
│   ├── __init__.py
│   ├── evaluate_model.py            # Evaluation script
│   └── train_model.py               # Training script
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading
│   ├── forecaster.py                # Ensemble forecasting
│   ├── pattern_extractor.py         # Pattern extraction
│   ├── similarity_matcher.py        # Similarity matching
│   ├── tariff_calculator.py         # Tariff calculation
│   └── utils.py                     # Utilities
├── tests/
│   ├── __init__.py
│   └── test_all.py                  # Unit tests
├── validation/
│   ├── __init__.py
│   ├── model_validator.py           # Model validation
│   └── time_series_split.py         # Temporal splitting
├── venv/                            # Virtual environment
├── .gitignore
├── .gitkeep
├── .python-version
├── data_quality.py                  # Data quality monitoring
├── docker-compose.yml               # Docker Compose
├── Dockerfile                       # Docker configuration
├── main.py                          # Main execution entry point
└── requirements.txt                 # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd component3
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure settings**

Edit `config/config.yaml` with your paths and preferences.

5. **Place your data**

Place your LECO dataset at the path specified in the config (default: `../processed/MASTER_DATASET_ALL_10TRANSFORMERS.csv`).

6. **Train the LSTM model** *(optional — falls back to pattern-based if not trained)*

```bash
python scripts/train_model.py
```

---

## Usage

### Command Line

**Single user forecast:**

```bash
python main.py --mode single --lat 6.9271 --lon 79.8612 --months '{"9":350.5,"10":420.2,"11":380.1}'
```

**Batch processing:**

```bash
python main.py --mode batch --input_file users.json --output_file results.json
```

**Train model:**

```bash
python scripts/train_model.py --epochs 100 --batch_size 32
```

**Evaluate model:**

```bash
python scripts/evaluate_model.py --model_path models/saved/lstm_model.h5 --cv --baselines
```

### Python API

```python
from src.data_loader import ElectricityDataLoader
from src.similarity_matcher import SimilarityMatcher
from src.forecaster import EnsembleForecaster
from src.tariff_calculator import PUCsLTariffCalculator
from src.utils import load_config

# Load config
config = load_config('config/config.yaml')

# Initialize components
data_loader = ElectricityDataLoader(config)
data_loader.load_dataset()

similarity = SimilarityMatcher(data_loader, config)
forecaster = EnsembleForecaster(config)
tariff = PUCsLTariffCalculator(config)

# User data
user_data = {
    'latitude': 6.9271,
    'longitude': 79.8612,
    'consumption_months': {9: 350.5, 10: 420.2, 11: 380.1},
    'tariff': 'D1',
    'phase': 'SP',
    'has_solar': 0,
    'household_size': 4
}

# Find similar households
similar = similarity.find_similar_households_safe(user_data)

# Generate forecast
forecast = forecaster.forecast(user_data, similar, {})

# Calculate bills
bills = tariff.calculate_annual_bills(
    forecast['forecast']['monthly_values'],
    'D1'
)

print(f"Annual consumption: {forecast['forecast']['statistics']['annual_total']:.0f} kWh")
print(f"Annual bill: Rs. {bills['annual_summary']['total_bill_lkr']:,.0f}")
```

### REST API

Start the API server:

```bash
python api/fastapi_app.py
```

Endpoints:

- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Example API call:

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 6.9271,
    "longitude": 79.8612,
    "consumption_months": [
      {"month": 9, "consumption": 350.5},
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

## Output Format

The system produces JSON output compatible with Component 4:

```json
{
  "consumption_forecast": {
    "monthly_kwh": {"1": 282.2, "2": 278.2, "...": "..."},
    "monthly_confidence": {"1": 0.9, "2": 0.85, "...": "..."},
    "uncertainty_ranges": {
      "1": {"lower_bound": 225.8, "upper_bound": 338.6}
    },
    "annual_total_kwh": 3718.9,
    "overall_confidence": 0.87
  },
  "billing_analysis": {
    "monthly_bills": {},
    "annual_summary": {
      "total_bill_lkr": 335368,
      "effective_rate_lkr_per_kwh": 90.18
    }
  },
  "user_profile": {}
}
```

---

## Testing

Run unit tests:

```bash
pytest tests/test_all.py -v
```

Run with coverage:

```bash
pytest tests/test_all.py --cov=. --cov-report=html
```

---

## Docker

Build and run with Docker:

```bash
# Build image
docker build -t component3 .

# Run container
docker run -p 8000:8000 component3

# Or use Docker Compose
docker-compose up
```

---

## Performance Metrics

| Metric | Target              |
|--------|---------------------|
| MAE | < 60 kWh/month      |
| MAPE | < 15%               |
| R-squared | > 0.75              |
| Inference time | < 1 second per user |

---

## Data Privacy

- All personal data is anonymized
- SQLite database with encryption option available
- No personally identifiable information (PII) stored in logs
- Compliant with Sri Lanka data protection laws

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Authors

R. S. P. S. Uthsara

---

## Acknowledgments

- Lanka Electricity Company (Pvt) Ltd (LECO) for the dataset
- PUCSL for tariff information
- NASA POWER for weather data