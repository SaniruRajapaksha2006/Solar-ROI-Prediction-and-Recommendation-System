import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import json


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
from ..models.lstm_model import LSTMForecaster
from ..features.feature_engineer import FeatureEngineer
from ..validation.model_validator import ModelValidator
from ..validation.time_series_split import TemporalSplitter
from ..src.utils import load_config, setup_logging, save_json, create_results_directory

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LSTM model')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation')
    parser.add_argument('--baselines', action='store_true',
                        help='Compare with baselines')
    parser.add_argument('--by_season', action='store_true',
                        help='Evaluate by season')
    parser.add_argument('--by_household', action='store_true',
                        help='Evaluate by household type')

    return parser.parse_args()


def load_model(model_path: str, config: dict) -> LSTMForecaster:
    lstm = LSTMForecaster(config)
    lstm.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return lstm


def evaluate_model(model: LSTMForecaster, config: dict, args):
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading dataset...")
    data_loader = ElectricityDataLoader(config)
    df = data_loader.load_dataset()

    # Create validator
    validator = ModelValidator(config)
    splitter = TemporalSplitter(config)

    results = {}

    # 1. Basic evaluation on test set
    logger.info("\nEvaluating on test set...")
    train_df, val_df, test_df = splitter.split(df)

    feature_engineer = FeatureEngineer(config)
    X_test, y_test = feature_engineer.prepare_for_training(test_df)

    y_pred = model.predict(X_test)

    # Flatten for metrics
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    metrics = validator.evaluate_forecast(y_test_flat, y_pred_flat)
    results['test_set'] = metrics

    logger.info(f"Test Set Metrics:")
    logger.info(f"  MAE:  {metrics['mae']:.2f} kWh")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} kWh")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R²:   {metrics['r2']:.4f}")

    # 2. Cross-validation
    if args.cv:
        logger.info("\nRunning cross-validation...")
        cv_results = validator.cross_validate(df, n_folds=5)
        results['cross_validation'] = cv_results

        logger.info(f"Cross-validation results:")
        for metric, value in cv_results['average_metrics'].items():
            logger.info(f"  {metric}: {value:.2f} (±{cv_results['std_metrics'].get(metric, 0):.2f})")

    # 3. Compare with baselines
    if args.baselines:
        logger.info("\nComparing with baselines...")
        baseline_results = validator.compare_with_baselines(df)
        results['baselines'] = baseline_results

        logger.info(f"Baseline comparison (MAE):")
        for name, bl_results in baseline_results.items():
            if 'mae' in bl_results:
                logger.info(f"  {name}: {bl_results['mae']:.2f} kWh")

    # 4. Evaluate by season
    if args.by_season:
        logger.info("\nEvaluating by season...")

        # Get predictions for each account
        account_predictions = {}

        for account in test_df['ACCOUNT_NO'].unique():
            account_test = test_df[test_df['ACCOUNT_NO'] == account]
            if len(account_test) >= 12:  # Need enough data
                X_acc, y_acc = feature_engineer.prepare_for_training(account_test)
                if len(X_acc) > 0:
                    y_pred_acc = model.predict(X_acc)

                    # Store predictions for this account
                    account_predictions[account] = {
                        'actual': y_acc.flatten().tolist(),
                        'predicted': y_pred_acc.flatten().tolist(),
                        'errors': np.abs(y_acc.flatten() - y_pred_acc.flatten()).tolist()
                    }

        # Use validator to analyze by season
        if account_predictions:
            seasonal_results = validator.validate_by_season(df, account_predictions)
            results['by_season'] = seasonal_results

            logger.info(f"Seasonal Performance:")
            for season, perf in seasonal_results.items():
                logger.info(f"  {season}: MAE={perf['mae']:.2f} kWh ({perf['n_samples']} samples)")
        else:
            logger.warning("  No account-level predictions available for season analysis")

    # 5. Evaluate by household type
    if args.by_household:
        logger.info("\nEvaluating by household type...")

        # Get predictions for each account (reuse from above if already computed)
        if not args.by_season or 'account_predictions' not in locals():
            account_predictions = {}
            for account in test_df['ACCOUNT_NO'].unique():
                account_test = test_df[test_df['ACCOUNT_NO'] == account]
                if len(account_test) >= 12:
                    X_acc, y_acc = feature_engineer.prepare_for_training(account_test)
                    if len(X_acc) > 0:
                        y_pred_acc = model.predict(X_acc)
                        account_predictions[account] = {
                            'errors': np.abs(y_acc.flatten() - y_pred_acc.flatten()).tolist()
                        }

        # Use validator to analyze by household type
        if account_predictions:
            household_results = validator.validate_on_household_types(df, account_predictions)
            results['by_household'] = household_results

            logger.info(f"Household Type Performance:")
            for hh_type, perf in household_results.items():
                if perf['mae']:
                    logger.info(f"  {hh_type}: MAE={perf['mae']:.2f} kWh ({perf['n_accounts']} accounts)")
                else:
                    logger.info(f"  {hh_type}: No data")
        else:
            logger.warning("  No account-level predictions available for household analysis")

    # 6. Residual analysis
    logger.info("\nAnalyzing residuals...")
    residual_analysis = validator.analyze_residuals(y_test_flat, y_pred_flat)
    results['residuals'] = residual_analysis

    logger.info(f"Residual analysis:")
    logger.info(f"  Mean: {residual_analysis['mean']:.2f}")
    logger.info(f"  Std:  {residual_analysis['std']:.2f}")
    logger.info(f"  Skewness: {residual_analysis['skewness']:.3f}")
    logger.info(f"  Normal: {residual_analysis.get('is_normal', False)}")

    # 7. Diebold-Mariano test against persistence baseline
    if args.baselines:
        logger.info("\nRunning Diebold-Mariano test...")

        # Get persistence predictions (simple baseline)
        persistence_errors = []
        for account in test_df['ACCOUNT_NO'].unique():
            account_data = test_df[test_df['ACCOUNT_NO'] == account].sort_values(['YEAR', 'MONTH'])
            values = account_data['NET_CONSUMPTION_kWh'].values
            if len(values) > 1:
                # Persistence: next = last
                persistence_errors.extend(np.abs(values[1:] - values[:-1]))

        if persistence_errors and len(persistence_errors) == len(y_test_flat):
            persistence_pred = y_test_flat - np.array(persistence_errors)  # Approximate
            dm_result = validator.diebold_mariano_test(
                y_test_flat,
                y_pred_flat,
                persistence_pred
            )
            results['diebold_mariano'] = dm_result

            logger.info(f"Diebold-Mariano test:")
            logger.info(f"  DM Statistic: {dm_result['dm_statistic']:.4f}")
            logger.info(f"  P-value: {dm_result['p_value']:.4f}")
            logger.info(f"  Significant: {dm_result['significant']}")
            logger.info(f"  Better model: {dm_result['better_model']}")

    return results


def main():
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/evaluation.log")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))

    # Create results directory
    results_dir = create_results_directory("evaluation_results")

    try:
        # Load model
        model = load_model(args.model_path, config)

        # Run evaluation
        results = evaluate_model(model, config, args)

        # Save results
        results_file = results_dir / "evaluation_results.json"
        save_json(results, str(results_file))

        logger.info(f"\n✅ Evaluation completed successfully")
        logger.info(f"Results saved to {results_dir}")
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())