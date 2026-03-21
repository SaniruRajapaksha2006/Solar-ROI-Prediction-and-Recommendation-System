"""
Integrated Solar ROI Prediction System
Combines all 4 components into a single workflow
- Component 1: Solar Generation (imported)
- Component 2: Geospatial (imported)
- Component 3: Consumption (called via subprocess - works with terminal)
- Component 4: ROI (imported)
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATH SETUP
# ============================================================================

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add all component paths for direct imports (Components 1, 2, 4)
sys.path.insert(0, str(PROJECT_ROOT))

# Component 1 - Solar Generation
sys.path.insert(0, str(PROJECT_ROOT / "Solar Generation Forecasting Model"))
sys.path.insert(0, str(PROJECT_ROOT / "Solar Generation Forecasting Model" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Solar Generation Forecasting Model" / "utils"))

# Component 2 - Geospatial
sys.path.insert(0, str(PROJECT_ROOT / "Geospatial Mapping & Transformer Suitability Model"))
sys.path.insert(0, str(PROJECT_ROOT / "Geospatial Mapping & Transformer Suitability Model" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "Geospatial Mapping & Transformer Suitability Model" / "backend" / "utils"))

# Component 4 - ROI
sys.path.insert(0, str(PROJECT_ROOT / "Return of Investment, Payback and Risk-Aware Financial Model"))
sys.path.insert(0, str(PROJECT_ROOT / "Return of Investment, Payback and Risk-Aware Financial Model" / "Backend"))

# Import shared models
from shared.models import (
    UserInput, SolarGenerationForecast, TransformerInfo,
    ConsumptionForecast, ROIAnalysis, IntegratedResult
)

# ============================================================================
# COMPONENT 1 WRAPPER - Solar Generation Forecasting (Direct Import)
# ============================================================================

class SolarGenerationWrapper:
    """Wrapper for Component 1 - Solar Generation Forecasting"""

    def __init__(self):
        self.initialized = False
        self._init_model()

    def _init_model(self):
        """Initialize the solar generation model"""
        try:
            # Add the correct paths - clean approach
            solar_root = PROJECT_ROOT / "Solar Generation Forecasting Model"

            # Print debug info
            logger.info(f"Checking Component 1 paths:")
            logger.info(f"  solar_root exists: {solar_root.exists()}")

            # CRITICAL: Only add solar_root, NOT utils or src directly
            # Remove any existing paths to avoid conflicts
            paths_to_remove = [p for p in sys.path if 'Solar Generation' in p]
            for p in paths_to_remove:
                if p in sys.path:
                    sys.path.remove(p)
                    logger.debug(f"Removed old path: {p}")

            # Add only the root directory
            if str(solar_root) not in sys.path:
                sys.path.insert(0, str(solar_root))
                logger.info(f"Added to path: {solar_root}")

            # Now check what's in sys.path
            logger.info(f"Component 1 paths in sys.path: {[p for p in sys.path if 'Solar Generation' in p]}")

            # Change working directory temporarily? No - better to use absolute imports
            # Let's check if we can import utils_config
            try:
                # This should work now because utils is a subfolder of solar_root
                from utils.utils_config import load_config
                logger.info("✅ Successfully imported utils_config")
            except ImportError as e:
                logger.error(f"Cannot import utils_config: {e}")
                logger.error(f"Current working directory: {os.getcwd()}")
                # List what's in the utils folder
                utils_path = solar_root / "utils"
                if utils_path.exists():
                    logger.info(f"Files in utils: {[f.name for f in utils_path.iterdir() if f.is_file()]}")
                raise

            # Now try to import predict
            from predict import predict_monthly
            self.predict_func = predict_monthly
            self.initialized = True
            logger.info("✅ Component 1 (Solar Generation) initialized successfully")

            # Quick test with try-except to handle any runtime errors
            try:
                test_df = self.predict_func(5.0)
                if test_df is not None and not test_df.empty:
                    logger.info(f"   Test successful - generated forecast for 5kW system")
            except Exception as test_e:
                logger.warning(f"Test prediction failed: {test_e}")
                logger.info("   Model loaded but test failed - will still use real model")

        except ImportError as e:
            logger.error(f"Failed to import Component 1: {e}")
            logger.info(f"Python path: {sys.path[:5]}")
            self.initialized = False
            self.predict_func = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Component 1: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.initialized = False
            self.predict_func = None

    def predict(self, panel_size_kw: float, custom_tariff: float = None) -> SolarGenerationForecast:
        """Generate solar generation forecast"""
        if not self.initialized or self.predict_func is None:
            logger.warning("Component 1 not available, using mock data")
            return self._mock_forecast(panel_size_kw)

        try:
            # Call the real predict_monthly function
            df = self.predict_func(panel_size_kw, custom_tariff)

            # Check if we got valid data
            if df is None or df.empty:
                logger.warning("predict_monthly returned empty data, using mock")
                return self._mock_forecast(panel_size_kw)

            monthly_export = {}
            monthly_income = {}
            monthly_confidence = {}

            for _, row in df.iterrows():
                month = int(row['Month'])
                monthly_export[month] = float(row['Predicted_Export_kWh'])
                monthly_income[month] = float(row['Estimated_Income_LKR'])
                monthly_confidence[month] = float(row['Confidence_Pct']) / 100

            annual_total = sum(monthly_export.values())
            annual_income = sum(monthly_income.values())
            avg_confidence = np.mean(list(monthly_confidence.values()))

            logger.info(f"   Real model output: {annual_total:.0f} kWh/year, {annual_income:.0f} LKR/year")

            return SolarGenerationForecast(
                monthly_export_kwh=monthly_export,
                monthly_income_lkr=monthly_income,
                monthly_confidence_pct=monthly_confidence,
                annual_total_kwh=annual_total,
                annual_income_lkr=annual_income,
                avg_confidence_pct=avg_confidence,
                panel_size_kw=panel_size_kw
            )
        except Exception as e:
            logger.error(f"Error in Component 1 prediction: {e}")
            logger.info(f"Falling back to mock data")
            return self._mock_forecast(panel_size_kw)

    def _mock_forecast(self, panel_size_kw: float) -> SolarGenerationForecast:
        monthly_export = {m: panel_size_kw * 120 * (0.8 + 0.4 * np.sin(m * np.pi / 6)) for m in range(1, 13)}
        monthly_income = {m: val * 27.06 for m, val in monthly_export.items()}
        monthly_confidence = {m: 0.85 for m in range(1, 13)}

        return SolarGenerationForecast(
            monthly_export_kwh=monthly_export,
            monthly_income_lkr=monthly_income,
            monthly_confidence_pct=monthly_confidence,
            annual_total_kwh=sum(monthly_export.values()),
            annual_income_lkr=sum(monthly_income.values()),
            avg_confidence_pct=0.85,
            panel_size_kw=panel_size_kw
        )


# ============================================================================
# COMPONENT 2 WRAPPER - Geospatial (Direct Import)
# ============================================================================

class GeospatialWrapper:
    """Wrapper for Component 2 - Geospatial Mapping & Transformer Suitability"""

    def __init__(self):
        self.initialized = False
        self._init_model()

    def _init_model(self):
        try:
            from backend.utils.data_preprocessor import DataPreprocessor
            from backend.utils.geo_utils import filter_nearby_transformers  # ← Store this
            from backend.models.ml_models import SolarSuitabilityMLModel

            self.filter_func = filter_nearby_transformers  # ← Add this line
            self.preprocessor = DataPreprocessor

            csv_path = PROJECT_ROOT / "processed" / "MASTER_DATASET_RESIDENTIAL_ONLY.csv"
            self.transformer_data = DataPreprocessor.load_and_prepare_data(str(csv_path))

            self.initialized = True
            logger.info("✅ Component 2 (Geospatial) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Component 2: {e}")
            self.initialized = False

    def assess_transformer(self, user_lat: float, user_lon: float,
                           solar_kw: float, radius_m: float = 2000) -> Optional[TransformerInfo]:
        if not self.initialized:
            return self._mock_assessment()

        try:
            # Use the imported filter function from __init__, not re-import
            # Remove this line: from utils.geo_utils import filter_nearby_transformers
            # Instead, use self.filter_func which we already set in __init__

            nearby = self.filter_func(
                self.transformer_data, user_lat, user_lon, radius_m
            )

            if nearby.empty:
                return None

            closest = nearby.iloc[0]
            cap = closest['ESTIMATED_CAPACITY_kW']
            load = closest['current_load_kW']
            existing_solar = closest['total_solar_capacity']
            available = cap - load - existing_solar

            can_support = available >= solar_kw
            utilization_after = (load + existing_solar + solar_kw) / cap

            return TransformerInfo(
                transformer_id=str(closest['TRANSFORMER_CODE']),
                distance_m=float(closest['DISTANCE_M']),
                suitability_score=float(closest.get('score', 70.0)),
                suitability_label="Good" if can_support else "Poor",
                capacity_kw=float(cap),
                current_load_kw=float(load),
                available_headroom_kw=float(available),
                can_support=can_support,
                curtailment_risk=utilization_after > 0.75,
                recommendation="Proceed with connection" if can_support else "Insufficient capacity"
            )
        except Exception as e:
            logger.error(f"Error in Component 2: {e}")
            return self._mock_assessment()

    def _mock_assessment(self) -> TransformerInfo:
        return TransformerInfo(
            transformer_id="TF-MOCK-001",
            distance_m=450.0,
            suitability_score=78.5,
            suitability_label="Good",
            capacity_kw=100.0,
            current_load_kw=45.2,
            available_headroom_kw=35.0,
            can_support=True,
            curtailment_risk=False,
            recommendation="Transformer has sufficient capacity"
        )


# ============================================================================
# COMPONENT 3 WRAPPER - Consumption Forecasting (SUBPROCESS - CRITICAL)
# ============================================================================

class ConsumptionWrapper:
    """
    Wrapper for Component 3 - Electricity Consumption Forecasting
    Calls main.py via subprocess because it's designed to run from terminal
    """

    def __init__(self):
        self.component3_path = PROJECT_ROOT / "Electricity Consumption Forecasting & Tariff Integration Model"
        self.results_dir = self.component3_path / "results"
        self.initialized = True
        logger.info("✅ Component 3 (Consumption) ready - will use subprocess")

    def forecast(self, user_data: UserInput) -> ConsumptionForecast:
        """
        Run Component 3 via subprocess and parse results
        This mimics the terminal command:
        python main.py --mode single --lat X --lon Y --months '{"9":350.5,"10":420.2,"11":380.1}'
        """
        logger.info(f"Running Component 3 for location: {user_data.latitude}, {user_data.longitude}")

        # Prepare the months JSON
        months_json = json.dumps(user_data.consumption_months)

        # Build the command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(self.component3_path / "main.py"),
            "--mode", "single",
            "--lat", str(user_data.latitude),
            "--lon", str(user_data.longitude),
            "--months", months_json,
            "--tariff", user_data.tariff,
            "--phase", user_data.phase,
            "--household_size", str(user_data.household_size)
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            # Run the command and wait for completion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.component3_path),
                timeout=120  # 2 minutes timeout
            )

            if result.returncode != 0:
                logger.error(f"Component 3 failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return self._mock_forecast(user_data)

            # Wait a moment for files to be written
            time.sleep(1)

            # Find the latest results file
            latest_results = self._find_latest_results()

            if latest_results:
                logger.info(f"Found results file: {latest_results}")
                return self._parse_results(latest_results, user_data)
            else:
                logger.error("No results file found")
                return self._mock_forecast(user_data)

        except subprocess.TimeoutExpired:
            logger.error("Component 3 timed out after 120 seconds")
            return self._mock_forecast(user_data)
        except Exception as e:
            logger.error(f"Error running Component 3: {e}")
            return self._mock_forecast(user_data)

    def _find_latest_results(self) -> Optional[Path]:
        """Find the most recent component3_results.json file"""
        if not self.results_dir.exists():
            return None

        # Find all timestamped directories
        result_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not result_dirs:
            return None

        # Sort by creation time (newest first)
        result_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        # Check for component3_results.json in the latest directory
        for dir_path in result_dirs:
            results_file = dir_path / "component3_results.json"
            if results_file.exists():
                return results_file

        return None

    def _parse_results(self, results_file: Path, user_data: UserInput) -> ConsumptionForecast:
        """Parse the JSON results from Component 3"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            # Extract forecast data
            forecast = data.get('forecast', {}).get('forecast', {})
            billing = data.get('billing', {})

            monthly_consumption = {}
            monthly_bills = {}

            # Get monthly values
            monthly_values = forecast.get('monthly_values', {})
            for month in range(1, 13):
                month_str = str(month)
                if month_str in monthly_values:
                    monthly_consumption[month] = float(monthly_values[month_str])
                elif month in monthly_values:
                    monthly_consumption[month] = float(monthly_values[month])
                else:
                    monthly_consumption[month] = 350.0  # fallback

            # Get monthly bills
            monthly_bills_data = billing.get('monthly_bills', {})
            for month in range(1, 13):
                month_str = str(month)
                if month_str in monthly_bills_data:
                    bill_data = monthly_bills_data[month_str]
                    if isinstance(bill_data, dict):
                        monthly_bills[month] = float(bill_data.get('total_bill_lkr', 0))
                    else:
                        monthly_bills[month] = float(bill_data)
                else:
                    monthly_bills[month] = 0

            stats = forecast.get('statistics', {})
            metadata = data.get('metadata', {})

            return ConsumptionForecast(
                monthly_consumption_kwh=monthly_consumption,
                monthly_bills_lkr=monthly_bills,
                annual_total_kwh=float(stats.get('annual_total', sum(monthly_consumption.values()))),
                annual_total_bill_lkr=float(billing.get('annual_summary', {}).get('total_bill_lkr', 0)),
                avg_confidence=float(stats.get('overall_confidence', 0.75)),
                method=metadata.get('method', 'pattern'),
                similar_households_count=data.get('similarity_analysis', {}).get('similar_households_found', 0)
            )

        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            return self._mock_forecast(user_data)

    def _mock_forecast(self, user_data: UserInput) -> ConsumptionForecast:
        """Return mock forecast for testing"""
        seasonal = {1: 0.85, 2: 0.9, 3: 1.15, 4: 1.25, 5: 1.2, 6: 1.0,
                    7: 0.85, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.05, 12: 0.95}

        base = 350
        monthly_consumption = {}
        monthly_bills = {}

        for month in range(1, 13):
            monthly_consumption[month] = base * seasonal[month]
            monthly_bills[month] = monthly_consumption[month] * 45.0

        return ConsumptionForecast(
            monthly_consumption_kwh=monthly_consumption,
            monthly_bills_lkr=monthly_bills,
            annual_total_kwh=sum(monthly_consumption.values()),
            annual_total_bill_lkr=sum(monthly_bills.values()),
            avg_confidence=0.75,
            method="mock_fallback",
            similar_households_count=0
        )


# ============================================================================
# COMPONENT 4 WRAPPER - ROI Financial Model (Direct Import)
# ============================================================================

class ROIWrapper:
    """Wrapper for Component 4 - ROI & Risk-Aware Financial Model"""

    def __init__(self):
        self.initialized = False
        self._init_model()

    def _init_model(self):
        try:
            from financeml import SolarFinancialModel
            self.model = SolarFinancialModel()
            self.initialized = True
            logger.info("✅ Component 4 (ROI) initialized")
        except ImportError as e:
            logger.error(f"Failed to import Component 4: {e}")
            self.initialized = False

    def calculate(self, panel_size_kw: float, annual_generation_kwh: float,
                  annual_consumption_kwh: float) -> ROIAnalysis:
        if not self.initialized:
            return self._mock_analysis(panel_size_kw, annual_generation_kwh, annual_consumption_kwh)

        try:
            result = self.model.calculate_financial_report(
                panel_size_kw, annual_generation_kwh, annual_consumption_kwh
            )

            return ROIAnalysis(
                total_investment_lkr=float(result['Total_Investment_LKR']),
                expected_roi_percent=float(result['Expected_ROI_Percent']),
                expected_payback_years=float(result['Expected_Payback_Years']),
                recommendation=result.get('Recommendation', 'Review options')
            )
        except Exception as e:
            logger.error(f"Error in Component 4: {e}")
            return self._mock_analysis(panel_size_kw, annual_generation_kwh, annual_consumption_kwh)

    def _mock_analysis(self, panel_size_kw: float, annual_gen: float,
                       annual_cons: float) -> ROIAnalysis:
        investment = panel_size_kw * 160000
        annual_savings = min(annual_gen, annual_cons) * 45.0
        net_profit = annual_savings * 20 - investment
        roi = (net_profit / investment) * 100 if investment > 0 else 0
        payback = investment / annual_savings if annual_savings > 0 else 20

        return ROIAnalysis(
            total_investment_lkr=investment,
            expected_roi_percent=roi,
            expected_payback_years=payback,
            recommendation="Good investment" if roi > 0 else "Review sizing"
        )


# ============================================================================
# MAIN INTEGRATED SYSTEM
# ============================================================================

class IntegratedSolarSystem:
    """Main integrated system combining all 4 components"""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing Integrated Solar ROI System")
        logger.info("=" * 60)

        self.solar_gen = SolarGenerationWrapper()
        self.geospatial = GeospatialWrapper()
        self.consumption = ConsumptionWrapper()
        self.roi = ROIWrapper()

        logger.info("System initialization complete")

    def analyze(self, user_input: UserInput, panel_size_kw: float = None) -> IntegratedResult:
        """
        Run complete analysis for a user
        """
        logger.info(f"Starting analysis for location: {user_input.latitude}, {user_input.longitude}")

        # Step 1: Get consumption forecast (via subprocess)
        logger.info("Step 1: Running consumption forecast (Component 3)...")
        consumption = self.consumption.forecast(user_input)
        logger.info(f"   Annual consumption: {consumption.annual_total_kwh:.0f} kWh")

        # Step 2: Determine recommended panel size
        if panel_size_kw is None:
            annual_consumption = consumption.annual_total_kwh
            panel_size_kw = max(1.5, min(20, round(annual_consumption / 1200, 1)))
            logger.info(f"Step 2: Recommended panel size: {panel_size_kw} kW")

        # Step 3: Get solar generation forecast
        logger.info("Step 3: Running solar generation forecast (Component 1)...")
        solar = self.solar_gen.predict(panel_size_kw)
        logger.info(f"   Annual generation: {solar.annual_total_kwh:.0f} kWh")

        # Step 4: Get transformer assessment
        logger.info("Step 4: Running transformer assessment (Component 2)...")
        transformer = self.geospatial.assess_transformer(
            user_input.latitude, user_input.longitude,
            panel_size_kw, user_input.search_radius_m
        )
        if transformer:
            logger.info(f"   Transformer: {transformer.transformer_id}, Score: {transformer.suitability_score:.0f}")

        # Step 5: Calculate ROI
        logger.info("Step 5: Running financial analysis (Component 4)...")
        roi_result = self.roi.calculate(
            panel_size_kw,
            solar.annual_total_kwh,
            consumption.annual_total_kwh
        )
        logger.info(f"   ROI: {roi_result.expected_roi_percent:.1f}%, Payback: {roi_result.expected_payback_years:.1f} years")

        logger.info("Analysis complete!")

        return IntegratedResult(
            user_input=user_input,
            solar_forecast=solar,
            transformer_info=transformer,
            consumption_forecast=consumption,
            roi_analysis=roi_result,
            recommended_panel_size_kw=panel_size_kw
        )

    def print_summary(self, result: IntegratedResult):
        """Print a formatted summary"""
        print("\n" + "=" * 70)
        print("  SOLAR ROI ANALYSIS REPORT")
        print("=" * 70)

        print(f"\n📍 LOCATION: {result.user_input.latitude}, {result.user_input.longitude}")
        print(f"📊 TARIFF: {result.user_input.tariff}")

        print(f"\n☀️ SOLAR GENERATION ({result.recommended_panel_size_kw} kW system):")
        print(f"   Annual Generation: {result.solar_forecast.annual_total_kwh:,.0f} kWh")
        print(f"   Annual Income: Rs. {result.solar_forecast.annual_income_lkr:,.0f}")
        print(f"   Forecast Confidence: {result.solar_forecast.avg_confidence_pct:.0%}")

        print(f"\n⚡ CONSUMPTION FORECAST:")
        print(f"   Annual Consumption: {result.consumption_forecast.annual_total_kwh:,.0f} kWh")
        print(f"   Annual Bill (without solar): Rs. {result.consumption_forecast.annual_total_bill_lkr:,.0f}")
        print(f"   Method: {result.consumption_forecast.method}")

        if result.transformer_info:
            print(f"\n🏢 TRANSFORMER SUITABILITY:")
            print(f"   ID: {result.transformer_info.transformer_id}")
            print(f"   Distance: {result.transformer_info.distance_m:.0f} m")
            print(f"   Score: {result.transformer_info.suitability_score:.1f}/100")
            print(f"   Available Headroom: {result.transformer_info.available_headroom_kw:.1f} kW")
            print(f"   Can Support: {'✅ Yes' if result.transformer_info.can_support else '❌ No'}")

        print(f"\n💰 FINANCIAL ANALYSIS:")
        print(f"   Total Investment: Rs. {result.roi_analysis.total_investment_lkr:,.0f}")
        print(f"   Expected ROI: {result.roi_analysis.expected_roi_percent:.1f}%")
        print(f"   Expected Payback: {result.roi_analysis.expected_payback_years:.1f} years")

        print(f"\n💡 RECOMMENDATION:")
        print(f"   {result.roi_analysis.recommendation}")

        print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Example usage"""

    user = UserInput(
        latitude=6.8511,
        longitude=79.9212,
        consumption_months={
            9: 250.5,
            10: 300.2,
            11: 280.1
        },
        tariff="D1",
        phase="SP",
        household_size=4,
        year=2025,
        search_radius_m=2000
    )

    system = IntegratedSolarSystem()
    result = system.analyze(user)
    system.print_summary(result)

    # Save report
    output_file = PROJECT_ROOT / "results" / "integrated_report.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result.model_dump(), f, indent=2, default=str)

    print(f"\n📄 Report saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()