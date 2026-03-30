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
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
import logging
import pandas as pd
import numpy as np
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PATH SETUP

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

# ARGUMENT PARSER
def parse_arguments():
    parser = argparse.ArgumentParser(description='Integrated Solar ROI System')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--months', type=str, help='JSON string of month:consumption pairs')
    parser.add_argument('--solar_kw', type=float, default=None, help='Solar capacity in kW (optional)')
    parser.add_argument('--radius_m', type=float, default=2000, help='Search radius in meters')
    parser.add_argument('--tariff', type=str, default='D1', help='Tariff category')
    parser.add_argument('--phase', type=str, default='SP', help='Phase (SP/TP)')
    parser.add_argument('--household_size', type=int, default=4, help='Household size')
    parser.add_argument('--year', type=int, default=2025, help='Year')
    return parser.parse_args()


# COMPONENT 1 WRAPPER - Solar Generation Forecasting
class SolarGenerationWrapper:
    """Wrapper for Component 1 - Solar Generation Forecasting"""

    def __init__(self):
        self.initialized = False
        self._init_model()

    def _init_model(self):
        """Initialize the solar generation model"""
        try:
            solar_root = PROJECT_ROOT / "Solar Generation Forecasting Model"

            logger.info(f"Checking Component 1 paths:")
            logger.info(f"  solar_root exists: {solar_root.exists()}")

            paths_to_remove = [p for p in sys.path if 'Solar Generation' in p]
            for p in paths_to_remove:
                if p in sys.path:
                    sys.path.remove(p)

            if str(solar_root) not in sys.path:
                sys.path.insert(0, str(solar_root))
                logger.info(f"Added to path: {solar_root}")

            logger.info(f"Component 1 paths in sys.path: {[p for p in sys.path if 'Solar Generation' in p]}")

            try:
                from utils.utils_config import load_config
                logger.info("✅ Successfully imported utils_config")
            except ImportError as e:
                logger.error(f"Cannot import utils_config: {e}")
                utils_path = solar_root / "utils"
                if utils_path.exists():
                    logger.info(f"Files in utils: {[f.name for f in utils_path.iterdir() if f.is_file()]}")
                raise

            from predict import predict_monthly
            self.predict_func = predict_monthly
            self.initialized = True
            logger.info("✅ Component 1 (Solar Generation) initialized successfully")

            try:
                test_df = self.predict_func(5.0)
                if test_df is not None and not test_df.empty:
                    logger.info(f"   Test successful - generated forecast for 5kW system")
            except Exception as test_e:
                logger.warning(f"Test prediction failed: {test_e}")

        except Exception as e:
            logger.error(f"Failed to import Component 1: {e}")
            self.initialized = False
            self.predict_func = None

    def predict(self, panel_size_kw: float, custom_tariff: float = None) -> SolarGenerationForecast:
        """Generate solar generation forecast"""
        if not self.initialized or self.predict_func is None:
            logger.warning("Component 1 not available, using mock data")
            return self._mock_forecast(panel_size_kw)

        try:
            df = self.predict_func(panel_size_kw, custom_tariff)

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


# COMPONENT 2 WRAPPER - Geospatial
class GeospatialWrapper:
    """Wrapper for Component 2 - Geospatial Mapping & Transformer Suitability"""

    def __init__(self):
        self.initialized = False
        self._init_model()

    def _init_model(self):
        try:
            from backend.utils.data_preprocessor import DataPreprocessor
            from backend.utils.geo_utils import filter_nearby_transformers
            from backend.models.ml_models import SolarSuitabilityMLModel

            self.filter_func = filter_nearby_transformers
            self.preprocessor = DataPreprocessor

            csv_path = PROJECT_ROOT / "processed" / "MASTER_DATASET_RESIDENTIAL_ONLY.csv"
            self.transformer_data = DataPreprocessor.load_and_prepare_data(str(csv_path))

            self.initialized = True
            logger.info("✅ Component 2 (Geospatial) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Component 2: {e}")
            self.initialized = False

    def assess_all_transformers(self, user_lat: float, user_lon: float,
                                solar_kw: float, radius_m: float = 2000) -> List[TransformerInfo]:
        """Get all transformers within radius, sorted by suitability"""
        if not self.initialized:
            return []

        try:
            # Get all nearby transformers using the existing filter function
            nearby = self.filter_func(
                self.transformer_data, user_lat, user_lon, radius_m
            )

            if nearby.empty:
                return []

            transformers = []
            for idx, row in nearby.iterrows():
                # Extract data
                cap = float(row['ESTIMATED_CAPACITY_kW'])
                load = float(row['current_load_kW'])
                sol = float(row['total_solar_capacity'])
                dist = float(row.get('DISTANCE_M', 0))
                lat = float(row.get('TRANSFORMER_LAT', 0))
                lon = float(row.get('TRANSFORMER_LON', 0))

                # Skip if missing coordinates
                if lat == 0 or lon == 0:
                    continue

                # Use the same calculation as assess_transformer
                available = cap - load - sol
                can_support = available >= solar_kw
                utilization_after = (load + sol + solar_kw) / cap

                # Use the existing score from the row if available
                score = float(row.get('score', 70.0))

                transformers.append(TransformerInfo(
                    transformer_id=str(row['TRANSFORMER_CODE']),
                    lat=lat,
                    lon=lon,
                    distance_m=float(dist),
                    suitability_score=score,
                    suitability_label="Good" if can_support else "Poor",
                    capacity_kw=cap,
                    current_load_kw=load,
                    available_headroom_kw=available,
                    can_support=can_support,
                    curtailment_risk=utilization_after > 0.75,
                    recommendation="Proceed with connection" if can_support else "Insufficient capacity",
                    utilization_after=utilization_after
                ))

            # Sort by score (highest first)
            transformers.sort(key=lambda x: x.suitability_score, reverse=True)
            return transformers

        except Exception as e:
            logger.error(f"Error getting all transformers: {e}")
            return []

    def assess_transformer(self, user_lat: float, user_lon: float,
                           solar_kw: float, radius_m: float = 2000) -> Optional[TransformerInfo]:
        """Get only the best transformer (for backward compatibility)"""
        all_transformers = self.assess_all_transformers(user_lat, user_lon, solar_kw, radius_m)
        return all_transformers[0] if all_transformers else None


# COMPONENT 3 WRAPPER - Consumption Forecasting
class ConsumptionWrapper:
    """Wrapper for Component 3 - Electricity Consumption Forecasting"""

    def __init__(self):
        self.component3_path = PROJECT_ROOT / "Electricity Consumption Forecasting & Tariff Integration Model"
        self.results_dir = self.component3_path / "results"
        self.initialized = True
        logger.info("✅ Component 3 (Consumption) ready - will use subprocess")

    def forecast(self, user_data: UserInput) -> ConsumptionForecast:
        """Run Component 3 via subprocess and parse results"""
        logger.info(f"Running Component 3 for location: {user_data.latitude}, {user_data.longitude}")

        months_json = json.dumps(user_data.consumption_months)

        cmd = [
            sys.executable,
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.component3_path),
                timeout=120
            )

            if result.returncode != 0:
                logger.error(f"Component 3 failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return self._mock_forecast(user_data)

            time.sleep(1)
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
        if not self.results_dir.exists():
            return None

        result_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not result_dirs:
            return None

        result_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        for dir_path in result_dirs:
            results_file = dir_path / "component3_results.json"
            if results_file.exists():
                return results_file
        return None

    def _parse_results(self, results_file: Path, user_data: UserInput) -> ConsumptionForecast:
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            forecast = data.get('forecast', {}).get('forecast', {})
            billing = data.get('billing', {})

            monthly_consumption = {}
            monthly_bills = {}

            monthly_values = forecast.get('monthly_values', {})
            for month in range(1, 13):
                month_str = str(month)
                if month_str in monthly_values:
                    monthly_consumption[month] = float(monthly_values[month_str])
                elif month in monthly_values:
                    monthly_consumption[month] = float(monthly_values[month])
                else:
                    monthly_consumption[month] = 350.0

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


# COMPONENT 4 WRAPPER - ROI Financial Model
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


# MAIN INTEGRATED SYSTEM
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

        # Step 4: Get ALL transformers assessment
        logger.info("Step 4: Running transformer assessment (Component 2)...")
        all_transformers = self.geospatial.assess_all_transformers(
            user_input.latitude, user_input.longitude,
            panel_size_kw, user_input.search_radius_m
        )

        best_transformer = all_transformers[0] if all_transformers else None

        if best_transformer:
            logger.info(f"   Best transformer: {best_transformer.transformer_id}, Score: {best_transformer.suitability_score:.0f}")
            logger.info(f"   Found {len(all_transformers)} transformers within radius")

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
            transformer_info=best_transformer,
            all_transformers=all_transformers,
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

        if result.all_transformers:
            print(f"\n🏢 TRANSFORMER SUITABILITY (Top 3):")
            for i, tf in enumerate(result.all_transformers[:3]):
                print(f"   {i+1}. {tf.transformer_id} - Score: {tf.suitability_score:.1f}/100, Distance: {tf.distance_m:.0f}m, {'✅ Can Support' if tf.can_support else '❌ Cannot Support'}")

        print(f"\n💰 FINANCIAL ANALYSIS:")
        print(f"   Total Investment: Rs. {result.roi_analysis.total_investment_lkr:,.0f}")
        print(f"   Expected ROI: {result.roi_analysis.expected_roi_percent:.1f}%")
        print(f"   Expected Payback: {result.roi_analysis.expected_payback_years:.1f} years")

        print(f"\n💡 RECOMMENDATION:")
        print(f"   {result.roi_analysis.recommendation}")

        print("\n" + "=" * 70)


# MAIN EXECUTION
def main():
    """Main execution with command line arguments"""
    args = parse_arguments()

    # If arguments provided, use them
    if args.lat and args.lon and args.months:
        try:
            consumption_months = json.loads(args.months)
        except json.JSONDecodeError:
            logger.error(f"Invalid months JSON: {args.months}")
            return 1

        user = UserInput(
            latitude=args.lat,
            longitude=args.lon,
            consumption_months=consumption_months,
            tariff=args.tariff,
            phase=args.phase,
            household_size=args.household_size,
            year=args.year,
            search_radius_m=args.radius_m
        )

        panel_size_kw = args.solar_kw if args.solar_kw else None
    else:
        # Use default example
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
        panel_size_kw = None
        logger.info("No command line args provided, using default example")

    system = IntegratedSolarSystem()
    result = system.analyze(user, panel_size_kw)
    system.print_summary(result)

    # Save report
    output_file = PROJECT_ROOT / "results" / "integrated_report.json"
    output_file.parent.mkdir(exist_ok=True)

    # Convert to dict for JSON serialization
    result_dict = result.model_dump()

    # Convert all_transformers to list of dicts
    if 'all_transformers' in result_dict and result_dict['all_transformers']:
        result_dict['all_transformers'] = [tf.model_dump() if hasattr(tf, 'model_dump') else tf for tf in result.all_transformers]

    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info(f"\n📄 Report saved to: {output_file}")

    return 0


if __name__ == "__main__":
    main()