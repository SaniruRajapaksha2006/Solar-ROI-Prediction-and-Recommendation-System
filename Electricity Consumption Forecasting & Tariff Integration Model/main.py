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
    from src.data_loader import ElectricityDataLoader
    from data_quality import DataQualityMonitor
    from features.feature_engineer import FeatureEngineer
    from features.cyclical_encoder import CyclicalFeatureEncoder
    from features.weather_integrator import WeatherIntegrator
    from models.lstm_model import LSTMForecaster
    from src.similarity_matcher import SimilarityMatcher
    from src.pattern_extractor import ConsumptionPatternExtractor
    from src.forecaster import EnsembleForecaster
    from src.tariff_calculator import PUCsLTariffCalculator, NetMeteringCalculator
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

def run_single_user(args, logger, config, results_dir):
    # Run forecasting for a single user
    logger.info("=" * 80)
    logger.info("COMPONENT 3: Single User Mode")
    logger.info("=" * 80)

    # Load user input
    user_data = load_user_input(args, logger)

    # Initialize components
    data_loader = ElectricityDataLoader(config)
    quality_monitor = DataQualityMonitor(config)
    feature_engineer = FeatureEngineer(config)
    similarity_matcher = SimilarityMatcher(data_loader, config)
    pattern_extractor = ConsumptionPatternExtractor(data_loader, config)
    lstm_forecaster = LSTMForecaster(config)

    model_path = Path("models/saved/lstm_model.h5")
    if model_path.exists():
        try:
            lstm_forecaster.load(model_path)
        except Exception as e:
            print(f"Error loading model normally: {e}")
            print("Attempting to load with compile=False...")

            # Load with compile=False to bypass metric issues
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            lstm_forecaster.model = model
            lstm_forecaster.is_trained = True
            print("Model loaded successfully with compile=False")
        print("LSTM model loaded successfully")
    else:
        print("LSTM model file not found")

    tariff_calculator = PUCsLTariffCalculator(config)
    net_metering = NetMeteringCalculator(tariff_calculator)

    # Initialize ensemble
    ensemble = EnsembleForecaster(config)
    ensemble.add_forecaster('lstm', lstm_forecaster, config['forecasting']['ensemble']['lstm_weight'])
    ensemble.add_forecaster('pattern', pattern_extractor, config['forecasting']['ensemble']['pattern_weight'])

    with Timer() as timer:
        # Step 1: Load dataset
        logger.info("\n Step 1: Loading dataset...")
        df = data_loader.load_dataset()

        # Check data quality
        quality_report = quality_monitor.check_data_quality(df)
        logger.info(f"Data quality: {quality_report.overall_score:.2%}")

        # Step 2: Find similar households (no look-ahead bias)
        logger.info("\nStep 2: Finding similar households...")
        similar_households = similarity_matcher.find_similar_households_safe(user_data)

        if not similar_households:
            logger.warning("No similar households found, using pattern-based fallback")
            method = 'pattern_only'
        else:
            logger.info(f"Found {len(similar_households)} similar households")
            method = args.method

        # Step 3: Engineer features
        logger.info("\nStep 3: Engineering features...")
        features = feature_engineer.create_all_features(
            user_data,
            similar_households,
            df
        )

        # Step 4: Generate forecast
        logger.info(f"\nStep 4: Generating forecast using {method}...")

        if method == 'ensemble':
            forecast_result = ensemble.forecast(user_data, similar_households, features)
        elif method == 'lstm' and len(similar_households) > 5:
            forecast_result = lstm_forecaster.forecast(user_data, features)
        else:
            forecast_result = pattern_extractor.extract_pattern_with_forecast(
                similar_households, user_data
            )

        # Step 5: Calculate bills (without solar)
        logger.info("\nStep 5: Calculating electricity bills...")
        monthly_consumption = forecast_result['forecast']['monthly_values']
        annual_bills = tariff_calculator.calculate_annual_bills(
            monthly_consumption, user_data['tariff']
        )

        # Step 6: Calculate net metering (if solar)
        if user_data['has_solar']:
            logger.info("\nStep 6: Applying net metering...")
            # This would use Component 1's generation forecast
            # Placeholder for now
            pass

        # Step 7: Prepare results
        logger.info("\nStep 7: Preparing final results...")

        final_results = {
            'metadata': {
                'component': 'Component 3 - Electricity Consumption Forecasting',
                'version': '2.0',
                'execution_timestamp': datetime.now().isoformat(),
                'execution_time_seconds': timer.elapsed,
                'method': method,
                # FIX: Remove config_path or use config variable
                'config_used': 'config/config.yaml',  # Just use a string
                'results_directory': str(results_dir)
            },
            'user_input': user_data,
            'data_quality': quality_report,
            'similarity_analysis': {
                'similar_households_found': len(similar_households),
                'top_similar_households': [
                    {'account': acc[:8] + '...', 'similarity': score}
                    for acc, score in (similar_households[:5] if similar_households else [])
                ]
            },
            'forecast': forecast_result,
            'billing': annual_bills
        }

        # Step 8: Export for Component 4
        component4_data = {
            'consumption_forecast': {
                'monthly_kwh': forecast_result['forecast']['monthly_values'],
                'monthly_confidence': forecast_result['forecast']['monthly_confidence'],
                'uncertainty_ranges': forecast_result['forecast'].get('uncertainty_ranges', {}),
                'annual_total_kwh': forecast_result['forecast']['statistics']['annual_total'],
                'overall_confidence': forecast_result['forecast']['statistics']['overall_confidence'],
                'method': method
            },
            'billing_analysis': {
                'monthly_bills': annual_bills['monthly_bills'],
                'annual_summary': annual_bills['annual_summary']
            },
            'user_profile': {
                'location': {
                    'latitude': user_data['latitude'],
                    'longitude': user_data['longitude']
                },
                'tariff': user_data['tariff'],
                'household_size': user_data.get('household_size', 4),
                'has_solar': user_data.get('has_solar', 0)
            },
            'metadata': {
                'forecast_timestamp': forecast_result['metadata']['generated_at'],
                'forecast_confidence': forecast_result['forecast']['statistics']['overall_confidence'],
                'similar_households_used': len(similar_households)
            }
        }

        # Step 9: Save results (already there)
        results_file = results_dir / 'component3_results.json'
        save_json(final_results, str(results_file))

        component4_file = results_dir / 'for_component4.json'
        save_json(component4_data, str(component4_file))

        # Step 9.5: Generate and save text files for easy viewing
        logger.info("\nStep 9.5: Saving human-readable text files...")

        try:
            # Pattern analysis text
            if hasattr(pattern_extractor, 'format_pattern_for_display'):
                pattern_display = pattern_extractor.format_pattern_for_display(
                    final_results['forecast']
                )
                with open(results_dir / 'pattern_analysis.txt', 'w') as f:
                    f.write(pattern_display)
                logger.info("  Saved pattern_analysis.txt")
        except Exception as e:
            logger.warning(f"  Could not save pattern_analysis.txt: {e}")

        try:
            # Consumption forecast text
            if hasattr(ensemble, 'format_forecast_for_display') or hasattr(pattern_extractor,
                                                                           'format_forecast_for_display'):
                # Try to get from ensemble first, then pattern
                if hasattr(ensemble, 'format_forecast_for_display'):
                    forecast_display = ensemble.format_forecast_for_display(forecast_result)
                else:
                    forecast_display = pattern_extractor.format_forecast_for_display(forecast_result)

                with open(results_dir / 'consumption_forecast.txt', 'w') as f:
                    f.write(forecast_display)
                logger.info("  Saved consumption_forecast.txt")
        except Exception as e:
            logger.warning(f"  Could not save consumption_forecast.txt: {e}")

        try:
            # Electricity bills text
            if hasattr(tariff_calculator, 'format_annual_bills_for_display'):
                bills_display = tariff_calculator.format_annual_bills_for_display(annual_bills)
                with open(results_dir / 'electricity_bills.txt', 'w') as f:
                    f.write(bills_display)
                logger.info("  Saved electricity_bills.txt")
        except Exception as e:
            logger.warning(f"  Could not save electricity_bills.txt: {e}")

        display_summary(final_results, results_file, component4_file)

    return final_results, component4_data

def run_batch_mode(args, logger, config, results_dir):
    # Run forecasting for multiple users from file
    logger.info("=" * 80)
    logger.info("COMPONENT 3: Batch Mode")
    logger.info("=" * 80)

    with open(args.input_file, 'r') as f:
        users = json.load(f)

    logger.info(f"Processing {len(users)} users")

    all_results = []
    for i, user_data in enumerate(users):
        logger.info(f"\n--- User {i+1}/{len(users)} ---")

        # Validate user data
        try:
            user_data = validate_user_input(user_data)
        except ValueError as e:
            logger.error(f"Invalid user {i}: {e}")
            all_results.append({'error': str(e), 'user': user_data})
            continue

        # Process single user (reusing functions)
        args.lat = user_data['latitude']
        args.lon = user_data['longitude']
        args.months = json.dumps(user_data['consumption_months'])
        args.tariff = user_data.get('tariff', 'D1')
        args.phase = user_data.get('phase', 'SP')
        args.has_solar = user_data.get('has_solar', 0)
        args.household_size = user_data.get('household_size', 4)

        try:
            result, _ = run_single_user(args, logger, config, results_dir)
            all_results.append({
                'user': user_data,
                'success': True,
                'forecast': result['forecast']['forecast']['statistics']
            })
        except Exception as e:
            logger.error(f"Error processing user {i}: {e}")
            all_results.append({
                'user': user_data,
                'success': False,
                'error': str(e)
            })

    # Save batch results
    batch_file = results_dir / 'batch_results.json'
    save_json({
        'total_users': len(users),
        'successful': sum(1 for r in all_results if r.get('success')),
        'failed': sum(1 for r in all_results if not r.get('success')),
        'results': all_results
    }, str(batch_file))

    logger.info(f"\nBatch complete: {sum(1 for r in all_results if r.get('success'))}/{len(users)} successful")

    return all_results


def run_training_mode(args, logger, config, results_dir):
    # Train LSTM model
    logger.info("=" * 80)
    logger.info("COMPONENT 3: Training Mode")
    logger.info("=" * 80)

    # Load data
    data_loader = ElectricityDataLoader(config)
    df = data_loader.load_dataset()

    # Create temporal split
    splitter = TemporalSplitter(config)
    train_df, val_df, test_df = splitter.split(df)

    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    X_train, y_train = feature_engineer.prepare_for_training(train_df)
    X_val, y_val = feature_engineer.prepare_for_training(val_df)

    logger.info(f"Features shape: {X_train.shape}")

    # Train LSTM
    lstm = LSTMForecaster(config)
    history = lstm.train(X_train, y_train, X_val, y_val)

    # Save model
    model_path = results_dir / 'lstm_model.h5'
    lstm.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Plot training history
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Train')
    axes[1].plot(history.history['val_mae'], label='Validation')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (kWh)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png')

    return history


def run_evaluation_mode(args, logger, config, results_dir):
    # Evaluate model performance
    logger.info("=" * 80)
    logger.info("COMPONENT 3: Evaluation Mode")
    logger.info("=" * 80)

    # Load data
    data_loader = ElectricityDataLoader(config)
    df = data_loader.load_dataset()

    # Create validator
    validator = ModelValidator(config)

    # Run cross-validation
    cv_results = validator.cross_validate(df)

    # Compare with baselines
    baseline_results = validator.compare_with_baselines(df)

    # Test statistical significance
    significance = validator.diebold_mariano_test(df)

    # Save results
    eval_file = results_dir / 'evaluation_results.json'
    save_json({
        'cross_validation': cv_results,
        'baselines': baseline_results,
        'significance': significance
    }, str(eval_file))

    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nCross-Validation:")
    for metric, value in cv_results.items():
        if isinstance(value, dict):
            print(f"  {metric}: mean={value['mean']:.2f}, std={value['std']:.2f}")
        else:
            print(f"  {metric}: {value:.2f}")

    print("\n📈 Baseline Comparison:")
    for model, metrics in baseline_results.items():
        print(f"  {model}: MAPE={metrics.get('mape', 0):.2f}%, RMSE={metrics.get('rmse', 0):.2f} kWh")

    print("\n" + "=" * 60)

    return cv_results


def run_validation_mode(args, logger, config, results_dir):
    # Validate against actual CEB bills
    logger.info("=" * 80)
    logger.info("COMPONENT 3: Validation Mode")
    logger.info("=" * 80)

    # This would use actual CEB bill data
    # Placeholder implementation
    logger.info("Validation against actual CEB bills - requires bill data")

    return {"status": "validation_complete"}


def display_summary(final_results, results_file, component4_file):
    # Display execution summary
    forecast_stats = final_results['forecast']['forecast']['statistics']
    billing = final_results['billing']['annual_summary']
    similar_count = final_results['similarity_analysis']['similar_households_found']

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Consumption Forecast:")
    print(f"   • Annual Consumption: {forecast_stats['annual_total']:,.0f} kWh")
    print(f"   • Monthly Average: {forecast_stats['annual_average']:.1f} kWh")
    print(f"   • Peak Month: {forecast_stats['peak_month']} ({forecast_stats['peak_consumption']:.1f} kWh)")
    print(f"   • Method: {final_results['metadata']['method']}")
    print(f"   • Confidence: {forecast_stats['overall_confidence']:.1%}")

    print(f"\nFinancial Analysis:")
    print(f"   • Annual Electricity Bill: Rs. {billing['total_bill_lkr']:,.0f}")
    print(f"   • Monthly Average: Rs. {billing['monthly_average_bill_lkr']:,.0f}")
    print(f"   • Effective Rate: Rs. {billing['effective_rate_lkr_per_kwh']:.2f}/kWh")

    print(f"\nSimilarity Analysis:")
    print(f"   • Similar Households Found: {similar_count}")

    print(f"\nPerformance:")
    exec_time = final_results['metadata'].get('execution_time_seconds', 0)
    if exec_time is None:
        exec_time = 0
    print(f"   • Execution Time: {exec_time:.2f} seconds")

    print(f"\nOutput Files:")
    print(f"   • Full Results: {results_file}")
    print(f"   • Component 4 Data: {component4_file}")

def main():
    # Main execution function
    # Parse arguments
    args = parse_arguments()

    # Setup environment
    logger, config, results_dir = setup_environment(args)

    try:
        # Execute based on mode
        if args.mode == 'single':
            results, component4 = run_single_user(args, logger, config, results_dir)

        elif args.mode == 'batch':
            results = run_batch_mode(args, logger, config, results_dir)

        elif args.mode == 'train':
            results = run_training_mode(args, logger, config, results_dir)

        elif args.mode == 'evaluate':
            results = run_evaluation_mode(args, logger, config, results_dir)

        elif args.mode == 'validate':
            results = run_validation_mode(args, logger, config, results_dir)

        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

        logger.info(f"Component 3 execution completed successfully in {args.mode} mode")
        return 0

    except Exception as e:
        logger.error(f"Error in Component 3 execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())