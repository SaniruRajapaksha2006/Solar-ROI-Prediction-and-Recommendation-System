from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
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

from data_loader import ElectricityDataLoader
from similarity_matcher import SimilarityMatcher
from pattern_extractor import ConsumptionPatternExtractor
from models.lstm_model import LSTMForecaster
from forecaster import EnsembleForecaster
from tariff_calculator import PUCsLTariffCalculator, NetMeteringCalculator
from features.feature_engineer import FeatureEngineer
from features.weather_integrator import WeatherIntegrator
from utils import load_config, setup_logging, validate_user_input

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