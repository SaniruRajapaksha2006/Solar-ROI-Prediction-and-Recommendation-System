import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from datetime import datetime

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
from models.lstm_model import LSTMForecaster
from features.feature_engineer import FeatureEngineer
from validation.time_series_split import TemporalSplitter
from validation.model_validator import ModelValidator
from utils import load_config, setup_logging, save_json, create_results_directory

logger = logging.getLogger(__name__)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM model')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after training')

    return parser.parse_args()


def train_lstm_model(config, args=None):
    logger.info("=" * 60)
    logger.info("LSTM MODEL TRAINING")
    logger.info("=" * 60)

    # Override config with command line args
    if args:
        if args.epochs:
            config['forecasting']['lstm']['epochs'] = args.epochs
        if args.batch_size:
            config['forecasting']['lstm']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['forecasting']['lstm']['learning_rate'] = args.learning_rate

    # Load data
    logger.info("\nLoading dataset...")
    data_loader = ElectricityDataLoader(config)
    df = data_loader.load_dataset()

    logger.info(f"Loaded {len(df)} records for {df['ACCOUNT_NO'].nunique()} accounts")

    # Create temporal split
    logger.info("\n🔀 Creating train/val/test split...")
    splitter = TemporalSplitter(config)
    train_df, val_df, test_df = splitter.split(df)

    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Validation: {len(val_df)} rows")
    logger.info(f"Test: {len(test_df)} rows")

    # Engineer features
    logger.info("\nEngineering features...")
    feature_engineer = FeatureEngineer(config)

    # Prepare training data
    X_train, y_train = feature_engineer.prepare_for_training(train_df)
    X_val, y_val = feature_engineer.prepare_for_training(val_df)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Create and train model
    logger.info("\nCreating LSTM model...")
    lstm = LSTMForecaster(config)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm.build_model(input_shape)

    logger.info(f"Model parameters: {lstm.model.count_params():,}")

    # Train
    logger.info("\nTraining model...")
    history = lstm.train(X_train, y_train, X_val, y_val)

    # Log training results
    final_loss = history['loss'][-1]
    final_val_loss = history.get('val_loss', [None])[-1]

    logger.info(f"Final training loss: {final_loss:.4f}")
    if final_val_loss:
        logger.info(f"Final validation loss: {final_val_loss:.4f}")

    # Evaluate on test set if available
    if args and args.evaluate and len(test_df) > 0:
        logger.info("\nEvaluating on test set...")
        X_test, y_test = feature_engineer.prepare_for_training(test_df)

        y_pred = lstm.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))

        logger.info(f"Test MAE: {mae:.2f} kWh")
        logger.info(f"Test RMSE: {rmse:.2f} kWh")

    # Save model
    if not (args and args.no_save):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path("models/saved")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"lstm_model_{timestamp}.h5"
        lstm.save(str(model_path))

        logger.info(f"Model saved to {model_path}")

    return lstm, history


def main():
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/training.log")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Create results directory
    results_dir = create_results_directory("training_results")

    try:
        # Train model
        model, history = train_lstm_model(config, args)

        # Save training history
        history_file = results_dir / "training_history.json"
        save_json(history, str(history_file))

        logger.info(f"\nTraining completed successfully")
        logger.info(f"Results saved to {results_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()