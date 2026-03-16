import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))
models_dir = current_dir / 'models'
sys.path.insert(0, str(models_dir))
features_dir = current_dir / 'features'
sys.path.insert(0, str(features_dir))
validation_dir = current_dir / 'validation'
sys.path.insert(0, str(validation_dir))

# Import components
try:
    from utils import (
        setup_logging, load_config, save_json, create_results_directory,
        validate_user_input, log_execution_time, Timer
    )
    from data_loader import ElectricityDataLoader
    from data_quality import DataQualityMonitor
    from features.feature_engineer import FeatureEngineer
    from features.cyclical_encoder import CyclicalFeatureEncoder
    from features.weather_integrator import WeatherIntegrator
    from models.lstm_model import LSTMForecaster
    from similarity_matcher import SimilarityMatcher
    from pattern_extractor import ConsumptionPatternExtractor
    from forecaster import EnsembleForecaster
    from tariff_calculator import PUCsLTariffCalculator, NetMeteringCalculator
    from validation.model_validator import ModelValidator
    from validation.time_series_split import TemporalSplitter
    from monitoring.drift_detector import ConceptDriftDetector

    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current Python path: {sys.path}")
    sys.exit(1)


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Component 3: Consumption Forecasting')

    # Input data
    parser.add_argument('--lat', type=float, required=False,
                        help='Latitude of user location')
    parser.add_argument('--lon', type=float, required=False,
                        help='Longitude of user location')
    parser.add_argument('--months', type=str, required=False,
                        help='JSON string of month:consumption pairs')
    parser.add_argument('--bills_file', type=str, required=False,
                        help='Path to file with consumption data')

    # User info
    parser.add_argument('--tariff', type=str, default='D1',
                        choices=['D1', 'GP1', 'GP2'],
                        help='Tariff category')
    parser.add_argument('--phase', type=str, default='SP',
                        choices=['SP', 'TP'],
                        help='Phase (SP=Single, TP=Three)')
    parser.add_argument('--has_solar', type=int, default=0,
                        choices=[0, 1],
                        help='Has solar installed (0/1)')
    parser.add_argument('--household_size', type=int, default=4,
                        help='Number of people in household')

    # Mode selection
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'train', 'evaluate', 'validate'],
                        help='Execution mode')
    parser.add_argument('--input_file', type=str,
                        help='Input file for batch mode')
    parser.add_argument('--output_file', type=str,
                        help='Output file for results')

    # Model options
    parser.add_argument('--method', type=str, default='ensemble',
                        choices=['lstm', 'pattern', 'ensemble'],
                        help='Forecasting method to use')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain LSTM model')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--profile', action='store_true',
                        help='Enable performance profiling')

    return parser.parse_args()


def setup_environment(args):
    # Setup execution environment
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level, log_file="logs/component3.log")
    logger = logging.getLogger(__name__)

    # Load configuration
    config_path = current_dir / 'config' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))

    # Create results directory
    results_dir = create_results_directory("results")

    return logger, config, results_dir


def load_user_input(args, logger):
    # Load user input from arguments or file
    if args.bills_file:
        # Load from file
        with open(args.bills_file, 'r') as f:
            user_data = json.load(f)
        logger.info(f"Loaded user data from {args.bills_file}")

    elif args.months:
        # Parse from command line
        consumption_months = json.loads(args.months)

        # Determine year from months or use default
        # Assuming all months are from the same year
        years = set()
        for month_str in consumption_months.keys():
            if ':' in month_str:  # If format includes year
                month, year = map(int, month_str.split(':'))
                years.add(year)

        if len(years) == 1:
            year = years.pop()
        else:
            year = 2025  # Default year

        user_data = {
            'latitude': args.lat,
            'longitude': args.lon,
            'consumption_months': {int(k.split(':')[0] if ':' in str(k) else k): v
                                   for k, v in consumption_months.items()},
            'year': year,  # ADD THIS
            'tariff': args.tariff,
            'phase': args.phase,
            'has_solar': args.has_solar,
            'household_size': args.household_size
        }
        logger.info(f"User location: {args.lat}, {args.lon}")
        logger.info(f"Consumption months: {len(consumption_months)} from year {year}")

    else:
        # Use default sample (Maharagama)
        user_data = {
            'latitude': 6.9271,
            'longitude': 79.8612,
            'consumption_months': {
                9: 350.5,  # September
                10: 420.2,  # October
                11: 380.1  # November
            },
            'year': 2025,  # ADD THIS
            'tariff': args.tariff,
            'phase': args.phase,
            'has_solar': args.has_solar,
            'household_size': args.household_size
        }
        logger.warning("No input provided, using sample user data")

    # Validate user input
    try:
        user_data = validate_user_input(user_data)
        logger.info("User input validated successfully")
    except ValueError as e:
        logger.error(f"Invalid user input: {e}")
        raise

    return user_data