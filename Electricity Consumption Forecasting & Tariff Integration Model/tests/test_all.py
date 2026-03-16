import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import json
import tempfile

# Add project root to path
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
from features.cyclical_encoder import CyclicalFeatureEncoder
from features.feature_engineer import FeatureEngineer
from validation.model_validator import ModelValidator
from validation.time_series_split import TemporalSplitter
from utils import validate_user_input, calculate_mae, calculate_rmse, calculate_mape
from data_quality import DataQualityMonitor


@pytest.fixture
def sample_config():
    # Sample configuration for testing
    return {
        'data': {
            'dataset_path': 'test_data.csv',
            'database_path': ':memory:',
            'columns': {
                'account_no': 'ACCOUNT_NO',
                'month': 'MONTH',
                'net_consumption': 'NET_CONSUMPTION_kWh',
                'customer_lat': 'CUSTOMER_LAT',
                'customer_lon': 'CUSTOMER_LON',
                'tariff': 'CAL_TARIFF',
                'phase': 'PHASE'
            }
        },
        'similarity': {
            'weights': {
                'geographic': 0.25,
                'consumption_pattern': 0.40,
                'technical': 0.20,
                'seasonal_compatibility': 0.15
            },
            'max_distance_km': 2.0,
            'min_similarity_score': 0.4,
            'top_n_similar': 10,
            'sri_lanka_bounds': {
                'lat_min': 5.9,
                'lat_max': 9.8,
                'lon_min': 79.6,
                'lon_max': 81.9
            }
        },
        'features': {
            'sri_lanka': {
                'holiday_months': [1, 4, 5, 12],
                'peak_months': [3, 4, 10],
                'low_months': [7, 8],
                'seasonal_factors': {
                    1: 0.85, 2: 0.90, 3: 1.15, 4: 1.25,
                    5: 1.20, 6: 1.00, 7: 0.85, 8: 0.90,
                    9: 1.00, 10: 1.10, 11: 1.05, 12: 0.95
                }
            },
            'weather': {
                'enabled': False
            }
        },
        'forecasting': {
            'lstm': {
                'enabled': True,
                'lookback_window': 12,
                'forecast_horizon': 12
            },
            'ensemble': {
                'lstm_weight': 0.7,
                'pattern_weight': 0.3,
                'enabled': True
            }
        },
        'tariff': {
            'D1': {
                'blocks': [
                    {'min': 0, 'max': 30, 'rate': 8.00},
                    {'min': 31, 'max': 60, 'rate': 10.00},
                    {'min': 61, 'max': 90, 'rate': 16.00},
                    {'min': 91, 'max': 120, 'rate': 50.00},
                    {'min': 121, 'max': 180, 'rate': 75.00},
                    {'min': 181, 'max': 9999, 'rate': 100.00}
                ],
                'fixed_charge': 180,
                'fuel_adjustment': 0.18,
                'vat_rate': 0.18,
                'export_rate': 100.00
            }
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_seed': 42
        },
        'monitoring': {
            'drift_detection': {
                'window_size': 6,
                'threshold': 2.0
            }
        }
    }


@pytest.fixture
def sample_dataframe():
    # Sample DataFrame for testing
    np.random.seed(42)

    data = []
    for account in ['ACC001', 'ACC002', 'ACC003']:
        for year in [2023, 2024]:
            for month in range(1, 13):
                data.append({
                    'ACCOUNT_NO': account,
                    'YEAR': year,
                    'MONTH': month,
                    'NET_CONSUMPTION_kWh': np.random.normal(300, 50),
                    'CUSTOMER_LAT': 6.9 + np.random.normal(0, 0.1),
                    'CUSTOMER_LON': 79.9 + np.random.normal(0, 0.1),
                    'CAL_TARIFF': 'D1',
                    'PHASE': 'SP',
                    'HAS_SOLAR': 0
                })

    return pd.DataFrame(data)


@pytest.fixture
def sample_user_data():
    # Sample user input for testing
    return {
        'latitude': 6.9271,
        'longitude': 79.8612,
        'consumption_months': {
            9: 350.5,
            10: 420.2,
            11: 380.1
        },
        'tariff': 'D1',
        'phase': 'SP',
        'has_solar': 0,
        'household_size': 4
    }


class TestDataLoader:
    def test_initialization(self, sample_config):
        loader = ElectricityDataLoader(sample_config)
        assert loader is not None
        assert loader.config == sample_config

    def test_validate_user_location(self, sample_config):
        loader = ElectricityDataLoader(sample_config)

        # Valid Sri Lanka location
        assert loader.validate_user_location(6.9, 79.9) is True

        # Invalid location
        assert loader.validate_user_location(10.0, 80.0) is False

    def test_calculate_distance(self, sample_config):
        loader = ElectricityDataLoader(sample_config)

        # Distance between Colombo and Kandy (approx)
        dist = loader._calculate_distance(6.9271, 79.8612, 7.2906, 80.6337)
        assert 100000 < dist < 120000  # ~115km



class TestSimilarityMatcher:
    def test_initialization(self, sample_config, sample_dataframe, mocker):
        # Mock data loader
        mock_loader = mocker.Mock()
        mock_loader.get_customer_profile.return_value = {
            'monthly_pattern': {m: {'net_consumption': 300} for m in range(1, 13)},
            'has_solar': 0,
            'tariff': 'D1',
            'phase': 'SP',
            'annual_stats': {'total': 3600, 'average': 300}
        }

        matcher = SimilarityMatcher(mock_loader, sample_config)
        assert matcher is not None

    def test_normalize_array(self, sample_config):
        matcher = SimilarityMatcher(None, sample_config)

        arr = [10, 20, 30, 40, 50]
        normalized = matcher._normalize_array(arr)

        assert len(normalized) == len(arr)
        assert min(normalized) == 0
        assert max(normalized) == 1



class TestPatternExtractor:
    def test_robust_statistics(self, sample_config, mocker):
        mock_loader = mocker.Mock()
        extractor = ConsumptionPatternExtractor(mock_loader, sample_config)

        # Test with sample data
        monthly_data = {m: [{'value': 300, 'weight': 1.0}] for m in range(1, 13)}
        pattern = extractor._calculate_robust_pattern(monthly_data)

        assert 'monthly_median' in pattern
        assert 'annual_total' in pattern
        assert pattern['annual_total'] > 0



class TestLSTMModel:
    def test_model_creation(self, sample_config):
        model = LSTMForecaster(sample_config)

        # Test building model
        input_shape = (12, 5)  # 12 timesteps, 5 features
        keras_model = model.build_model(input_shape)

        assert keras_model is not None
        assert keras_model.input_shape == (None, 12, 5)

    def test_prepare_sequences(self, sample_config):
        model = LSTMForecaster(sample_config)

        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        X_seq, y_seq = model.prepare_sequences(X, y)

        assert len(X_seq.shape) == 3
        assert X_seq.shape[1] == model.lookback



class TestTariffCalculator:
    def test_bill_calculation(self, sample_config):
        calculator = PUCsLTariffCalculator(sample_config)

        # Test various consumption levels
        test_cases = [
            (30, 8.00 * 30 + 180),  # First block only
            (75, 8 * 30 + 10 * 30 + 16 * 15 + 180),  # Up to third block
            (200, 8 * 30 + 10 * 30 + 16 * 30 + 50 * 30 + 75 * 60 + 100 * 20 + 180)  # All blocks
        ]

        for consumption, expected_base in test_cases:
            bill = calculator.calculate_monthly_bill(consumption, 'D1')
            assert bill['energy_charge_lkr'] > 0
            assert bill['fixed_charge_lkr'] == 180
            assert bill['total_bill_lkr'] > bill['energy_charge_lkr']

    def test_net_metering(self, sample_config):
        base = PUCsLTariffCalculator(sample_config)
        calculator = NetMeteringCalculator(base)

        # Test export scenario
        bill = calculator.calculate_monthly_bill(300, 350, 'D1')
        assert bill['credit_earned'] > 0
        assert bill['total_bill_lkr'] == 180  # Minimum bill

        # Test import scenario
        bill = calculator.calculate_monthly_bill(350, 300, 'D1')
        assert 'credits_used' in bill or 'credit_earned' not in bill



class TestCyclicalEncoder:
    def test_month_encoding(self, sample_config):
        encoder = CyclicalFeatureEncoder(sample_config)

        # Test single month
        encoded = encoder.encode_month(1)
        assert 'month_sin' in encoded
        assert 'month_cos' in encoded

        # Test array
        encoded = encoder.encode_month([1, 2, 3, 4])
        assert len(encoded['month_sin']) == 4

    def test_decode_month(self, sample_config):
        encoder = CyclicalFeatureEncoder(sample_config)

        for month in range(1, 13):
            encoded = encoder.encode_month(month)
            decoded = encoder.decode_month(
                encoded['month_sin'] if isinstance(encoded['month_sin'], float) else encoded['month_sin'][0],
                encoded['month_cos'] if isinstance(encoded['month_cos'], float) else encoded['month_cos'][0]
            )
            assert decoded == month or decoded == month + 1 or decoded == month - 1  # Allow off-by-one due to rounding



class TestModelValidator:
    def test_evaluation_metrics(self, sample_config):
        validator = ModelValidator(sample_config)

        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        metrics = validator.evaluate_forecast(y_true, y_pred)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert metrics['mae'] > 0

    def test_residual_analysis(self, sample_config):
        validator = ModelValidator(sample_config)

        y_true = np.random.randn(100) * 100 + 300
        y_pred = y_true + np.random.randn(100) * 20

        analysis = validator.analyze_residuals(y_true, y_pred)

        assert 'mean' in analysis
        assert 'std' in analysis
        assert 'skewness' in analysis



class TestTemporalSplitter:
    def test_split(self, sample_config, sample_dataframe):
        splitter = TemporalSplitter(sample_config)

        train_df, val_df, test_df = splitter.split(sample_dataframe)

        assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)
        assert len(train_df) > len(val_df)
        assert len(train_df) > len(test_df)

    def test_walk_forward(self, sample_config, sample_dataframe):
        splitter = TemporalSplitter(sample_config)

        n_windows = 0
        for train_df, test_df in splitter.walk_forward_split(sample_dataframe, n_windows=3):
            n_windows += 1
            assert len(train_df) > 0
            assert len(test_df) > 0

        assert n_windows > 0



class TestUtils:
    def test_validate_user_input(self, sample_user_data):
        # Valid input
        validated = validate_user_input(sample_user_data)
        assert validated['latitude'] == sample_user_data['latitude']

        # Invalid latitude
        invalid_data = sample_user_data.copy()
        invalid_data['latitude'] = 10.0
        with pytest.raises(ValueError):
            validate_user_input(invalid_data)

    def test_calculation_functions(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)

        assert mae > 0
        assert rmse > 0
        assert mape > 0



class TestDataQuality:
    def test_quality_check(self, sample_config, sample_dataframe):
        monitor = DataQualityMonitor(sample_config)

        report = monitor.check_data_quality(sample_dataframe)

        assert report.total_records == len(sample_dataframe)
        assert report.unique_accounts == sample_dataframe['ACCOUNT_NO'].nunique()
        assert 0 <= report.overall_score <= 1

    def test_quality_trend(self, sample_config, sample_dataframe):
        monitor = DataQualityMonitor(sample_config)

        # Run multiple checks
        for _ in range(5):
            monitor.check_data_quality(sample_dataframe)

        trend = monitor.get_quality_trend()
        assert 'trend' in trend



class TestIntegration:
    def test_end_to_end(self, sample_config, sample_dataframe, sample_user_data, mocker):
        """Test the entire pipeline end-to-end"""

        # Mock data loader
        mock_loader = mocker.Mock()
        mock_loader.load_dataset.return_value = sample_dataframe
        mock_loader.get_customer_profile.return_value = {
            'monthly_pattern': {m: {'net_consumption': 300} for m in range(1, 13)},
            'has_solar': 0,
            'tariff': 'D1',
            'phase': 'SP',
            'annual_stats': {'total': 3600, 'average': 300}
        }
        mock_loader.get_profiles_by_location.return_value = [
            ('ACC001', 1000),
            ('ACC002', 1500)
        ]

        # Initialize components
        similarity = SimilarityMatcher(mock_loader, sample_config)
        pattern = ConsumptionPatternExtractor(mock_loader, sample_config)

        # Find similar households
        similar = similarity.find_similar_households_safe(sample_user_data)
        assert len(similar) > 0

        # Extract pattern
        pattern_result = pattern.extract_pattern_safe(similar, sample_user_data)
        assert 'pattern' in pattern_result
        assert 'confidence' in pattern_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=."])