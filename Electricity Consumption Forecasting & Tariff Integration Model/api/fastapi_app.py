from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import uvicorn
import logging
import sys
from pathlib import Path

def find_project_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / 'src').exists():
            return current
        current = current.parent
    return Path.cwd()

project_root = find_project_root()
sys.path.insert(0, str(project_root))

from ..src.data_loader import ElectricityDataLoader
from ..src.similarity_matcher import SimilarityMatcher
from ..src.pattern_extractor import ConsumptionPatternExtractor
from ..models.lstm_model import LSTMForecaster
from ..src.forecaster import EnsembleForecaster
from ..src.tariff_calculator import PUCsLTariffCalculator, NetMeteringCalculator
from ..features.feature_engineer import FeatureEngineer
from ..features.weather_integrator import WeatherIntegrator
from ..src.utils import load_config, setup_logging, validate_user_input

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
config = load_config(str(config_path))

# Initialize FastAPI
app = FastAPI(
    title="Solar ROI - Consumption Forecasting API",
    description="Component 3: Electricity Consumption Forecasting for Sri Lanka",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
_data_loader = None
_similarity_matcher = None
_pattern_extractor = None
_lstm_forecaster = None
_ensemble_forecaster = None
_tariff_calculator = None
_net_metering = None
_feature_engineer = None
_weather_integrator = None


def get_data_loader():
    # Lazy load data loader
    global _data_loader
    if _data_loader is None:
        _data_loader = ElectricityDataLoader(config)
        _data_loader.load_dataset()
    return _data_loader


def get_similarity_matcher():
    # Lazy load similarity matcher
    global _similarity_matcher
    if _similarity_matcher is None:
        _similarity_matcher = SimilarityMatcher(get_data_loader(), config)
    return _similarity_matcher


def get_pattern_extractor():
    # Lazy load pattern extractor
    global _pattern_extractor
    if _pattern_extractor is None:
        _pattern_extractor = ConsumptionPatternExtractor(get_data_loader(), config)
    return _pattern_extractor


def get_lstm_forecaster():
    # Lazy load LSTM forecaster
    global _lstm_forecaster
    if _lstm_forecaster is None:
        _lstm_forecaster = LSTMForecaster(config)
        # Try to load pre-trained model
        model_path = Path("models/saved/lstm_model.h5")
        if model_path.exists():
            try:
                _lstm_forecaster.load(str(model_path))
                logger.info("Loaded pre-trained LSTM model")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e}")
    return _lstm_forecaster


def get_ensemble_forecaster():
    # Lazy load ensemble forecaster
    global _ensemble_forecaster
    if _ensemble_forecaster is None:
        _ensemble_forecaster = EnsembleForecaster(config)
        _ensemble_forecaster.add_forecaster(
            'lstm',
            get_lstm_forecaster(),
            config['forecasting']['ensemble']['lstm_weight']
        )
        _ensemble_forecaster.add_forecaster(
            'pattern',
            get_pattern_extractor(),
            config['forecasting']['ensemble']['pattern_weight']
        )
    return _ensemble_forecaster


def get_tariff_calculator():
    # Lazy load tariff calculator
    global _tariff_calculator
    if _tariff_calculator is None:
        _tariff_calculator = PUCsLTariffCalculator(config)
    return _tariff_calculator


def get_net_metering():
    # Lazy load net metering calculator
    global _net_metering
    if _net_metering is None:
        _net_metering = NetMeteringCalculator(get_tariff_calculator())
    return _net_metering


def get_feature_engineer():
    # Lazy load feature engineer
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer(config)
    return _feature_engineer


def get_weather_integrator():
    # Lazy load weather integrator
    global _weather_integrator
    if _weather_integrator is None and config['features']['weather']['enabled']:
        _weather_integrator = WeatherIntegrator(config)
    return _weather_integrator

# Pydantic models for request/response
class ConsumptionMonth(BaseModel):
    month: int = Field(..., ge=1, le=12, description="Month number (1-12)")
    consumption: float = Field(..., ge=0, le=2000, description="Consumption in kWh")


class UserInput(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    consumption_months: List[ConsumptionMonth] = Field(..., min_items=2, max_items=12)
    tariff: str = Field("D1", description="Tariff category (D1, GP1, GP2)")
    phase: str = Field("SP", description="Phase (SP=Single, TP=Three)")
    has_solar: int = Field(0, ge=0, le=1, description="Has solar installed (0/1)")
    household_size: int = Field(4, ge=1, le=20, description="Number of people")

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v):
        if not (5.9 <= v <= 9.8):
            raise ValueError('Latitude must be within Sri Lanka (5.9 to 9.8)')
        return v

    @field_validator('longitude')
    @classmethod
    def validate_longitude(cls, v):
        if not (79.6 <= v <= 81.9):
            raise ValueError('Longitude must be within Sri Lanka (79.6 to 81.9)')
        return v

    @field_validator('tariff')
    @classmethod
    def validate_tariff(cls, v):
        if v not in ['D1', 'GP1', 'GP2']:
            raise ValueError('Tariff must be D1, GP1, or GP2')
        return v

    @field_validator('phase')
    @classmethod
    def validate_phase(cls, v):
        if v not in ['SP', 'TP']:
            raise ValueError('Phase must be SP or TP')
        return v


class ForecastResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# API Endpoints
@app.get("/")
async def root():
    # Root endpoint
    return {
        "service": "Solar ROI - Consumption Forecasting",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/forecast",
            "/health",
            "/tariff/calculate",
            "/similar/accounts",
            "/models/info"
        ]
    }


@app.get("/health")
async def health_check():
    # Health check endpoint
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_loader": get_data_loader() is not None,
            "similarity_matcher": get_similarity_matcher() is not None,
            "pattern_extractor": get_pattern_extractor() is not None,
            "lstm_forecaster": get_lstm_forecaster() is not None,
            "tariff_calculator": get_tariff_calculator() is not None
        }
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(user_input: UserInput, background_tasks: BackgroundTasks):
    # Generate consumption forecast for a user

    try:
        # Convert to dictionary
        user_data = {
            'latitude': user_input.latitude,
            'longitude': user_input.longitude,
            'consumption_months': {m.month: m.consumption for m in user_input.consumption_months},
            'tariff': user_input.tariff,
            'phase': user_input.phase,
            'has_solar': user_input.has_solar,
            'household_size': user_input.household_size
        }

        logger.info(f"Processing forecast for location: {user_input.latitude}, {user_input.longitude}")

        # Find similar households
        similar_households = get_similarity_matcher().find_similar_households_safe(user_data)

        # Engineer features
        features = get_feature_engineer().create_all_features(
            user_data,
            similar_households,
            get_data_loader().df
        )

        # Add weather if enabled
        if get_weather_integrator():
            weather_features = get_weather_integrator().get_features(
                user_input.latitude,
                user_input.longitude,
                user_data['consumption_months']
            )
            features['weather'] = weather_features

        # Generate forecast
        method = 'ensemble' if config['forecasting']['ensemble']['enabled'] else 'pattern'

        if method == 'ensemble':
            forecast_result = get_ensemble_forecaster().forecast(
                user_data, similar_households, features
            )
        else:
            forecast_result = get_pattern_extractor().extract_pattern_with_forecast(
                similar_households, user_data
            )

        # Calculate bills
        monthly_consumption = forecast_result['forecast']['monthly_values']
        annual_bills = get_tariff_calculator().calculate_annual_bills(
            monthly_consumption, user_input.tariff
        )

        # Apply net metering if user has solar
        if user_input.has_solar:
            # This would need generation forecast from Component 1
            # Placeholder for now
            pass

        # Prepare response
        response_data = {
            'forecast': forecast_result['forecast'],
            'billing': annual_bills,
            'similar_households': len(similar_households),
            'method': method,
            'confidence': forecast_result['forecast']['statistics']['overall_confidence']
        }

        # Log request in background
        background_tasks.add_task(
            log_request,
            user_data,
            forecast_result['forecast']['statistics']
        )

        return ForecastResponse(
            status="success",
            message="Forecast generated successfully",
            data=response_data
        )

    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tariff/calculate")
async def calculate_tariff(consumption: float, tariff: str = "D1",
                           generation: Optional[float] = None):
    # Calculate electricity bill with optional net metering

    try:
        if generation and generation > 0:
            # With net metering
            calculator = get_net_metering()
            bill = calculator.calculate_monthly_bill(consumption, generation, tariff)
        else:
            # Without solar
            calculator = get_tariff_calculator()
            bill = calculator.calculate_monthly_bill(consumption, tariff)

        return {
            "status": "success",
            "data": bill
        }

    except Exception as e:
        logger.error(f"Error in tariff calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/accounts")
async def find_similar_accounts(lat: float, lon: float,
                                months: str, tariff: str = "D1"):
    # Find similar accounts for a location
    try:
        # Parse months string (format: "9:350.5,10:420.2,11:380.1")
        consumption_months = {}
        for pair in months.split(','):
            month, value = pair.split(':')
            consumption_months[int(month)] = float(value)

        user_data = {
            'latitude': lat,
            'longitude': lon,
            'consumption_months': consumption_months,
            'tariff': tariff,
            'phase': 'SP',
            'has_solar': 0
        }

        similar = get_similarity_matcher().find_similar_households_safe(user_data)

        # Get details for top matches
        accounts = []
        for account, score in similar[:5]:
            profile = get_data_loader().get_customer_profile(account)
            if profile:
                accounts.append({
                    'account': account[:8] + '...',
                    'similarity': score,
                    'annual_consumption': profile['annual_stats']['total'],
                    'monthly_average': profile['annual_stats']['average']
                })

        return {
            "status": "success",
            "data": {
                "total_found": len(similar),
                "accounts": accounts
            }
        }

    except Exception as e:
        logger.error(f"Error finding similar accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def model_info():
    # Get information about available models
    lstm = get_lstm_forecaster()

    return {
        "models": {
            "lstm": {
                "available": lstm.is_trained if lstm else False,
                "architecture": lstm.architecture if lstm else None,
                "layers": lstm.layers if lstm else None,
                "lookback": lstm.lookback if lstm else None
            },
            "pattern_based": {
                "available": True,
                "description": "Pattern-based fallback method"
            },
            "ensemble": {
                "available": True,
                "weights": config['forecasting']['ensemble']
            }
        },
        "tariffs": list(config['tariff'].keys()) if 'tariff' in config else []
    }


@app.post("/train/trigger")
async def trigger_training(background_tasks: BackgroundTasks):
    # Trigger model retraining (admin only)

    # In production, add authentication
    background_tasks.add_task(train_model_background)

    return {
        "status": "success",
        "message": "Training started in background"
    }


def train_model_background():
    # Background task for model training
    logger.info("Starting background model training")
    try:
        from ..scripts.train_model import train_lstm_model
        train_lstm_model(config)
        logger.info("Background training completed")
    except Exception as e:
        logger.error(f"Background training failed: {e}")

def log_request(user_data: Dict, forecast_stats: Dict):
    # Log request for monitoring
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_location': (user_data['latitude'], user_data['longitude']),
        'n_months': len(user_data['consumption_months']),
        'forecast_annual': forecast_stats['annual_total'],
        'forecast_confidence': forecast_stats['overall_confidence']
    }

    # Append to log file
    log_file = Path("logs/api_requests.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=True
    )