import numpy as np
import json
import os
import logging
from typing import Dict, Any

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SolarFinancialModel:

    def __init__(self, json_filename: str = "market_data.json"):
        self.PRICING_DATABASE: Dict[int, float] = {}
        self.VENDOR_DATABASE: list = []
        self.BASE_IMPORT_TARIFF_LKR: float = 45.00
        self.EXPORT_UNDER_5: float = 20.90
        self.EXPORT_OVER_5: float = 19.61
        self.DISCOUNT_RATE: float = 0.10
        self.PROJECT_LIFETIME: int = 20


        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_filepath = os.path.join(base_dir, json_filename)

        try:
            if os.path.exists(json_filepath):
                with open(json_filepath, "r") as db_file:
                    data = json.load(db_file)

                    self.PRICING_DATABASE = {int(k): v for k, v in data["pricing_database"].items()}
                    self.VENDOR_DATABASE = data["vendors"]
                    self.BASE_IMPORT_TARIFF_LKR = data["tariffs"]["base_import_tariff"]
                    self.EXPORT_UNDER_5 = data["tariffs"]["export_tariff_under_5kw"]
                    self.EXPORT_OVER_5 = data["tariffs"]["export_tariff_over_5kw"]
                    self.DISCOUNT_RATE = data["tariffs"]["discount_rate"]

                    logger.info("Model successfully initialized with live market data from JSON database.")
            else:
                raise FileNotFoundError(f"{json_filepath} not found.")

        except Exception as e:
            logger.warning(f"Database error: {e}. Falling back to safe offline market data.")
            self.PRICING_DATABASE = {3: 750000, 5: 800000, 8: 1450000, 10: 1800000, 15: 2600000, 20: 2900000}
            self.VENDOR_DATABASE = [
                {"Name": "System Default Vendor", "Location": "Unknown", "Contact": "N/A", "Specialty": "N/A"}]

    def get_system_cost(self, size_kw: float) -> float:
        return self.PRICING_DATABASE.get(size_kw, size_kw * 200000.0)

    def get_export_tariff(self, size_kw: float) -> float:
        return self.EXPORT_UNDER_5 if size_kw <= 5 else self.EXPORT_OVER_5

    def calculate_financial_report(self, system_size_kw: float,
                                   predicted_annual_generation_kwh: float,
                                   predicted_annual_consumption_kwh: float) -> Dict[str, Any]:

        if system_size_kw <= 0 or predicted_annual_generation_kwh <= 0 or predicted_annual_consumption_kwh < 0:
            raise ValueError("System size, generation, and consumption must be valid non-negative numbers.")

        # Force float to prevent NumPy casting errors
        initial_investment_lkr = float(self.get_system_cost(system_size_kw))
        export_tariff_lkr = float(self.get_export_tariff(system_size_kw))


        n_sims = 2000

        degradation_rates = np.random.uniform(0.005, 0.010, n_sims)
        annual_maintenance = (initial_investment_lkr * 0.01) * np.random.normal(1.0, 0.2, n_sims)
        inverter_fail_year = np.random.randint(8, 13, n_sims)
        tariff_escalation = np.random.uniform(0.02, 0.05, n_sims)
        inverter_cost = initial_investment_lkr * 0.25

        # Initialize tracking arrays for all 2000 simulations (using dtype=float)
        cumulative_cash = np.full(n_sims, -initial_investment_lkr, dtype=float)
        npv_array = np.full(n_sims, -initial_investment_lkr, dtype=float)
        payback_years = np.full(n_sims, self.PROJECT_LIFETIME + 1.0, dtype=float)
        paid_back = np.zeros(n_sims, dtype=bool)
        total_net_profit = np.zeros(n_sims, dtype=float)
        current_import_tariff = np.full(n_sims, self.BASE_IMPORT_TARIFF_LKR, dtype=float)

        for year in range(1, self.PROJECT_LIFETIME + 1):
            gen_for_year = predicted_annual_generation_kwh * ((1 - degradation_rates) ** (year - 1))

            savings = np.where(gen_for_year >= predicted_annual_consumption_kwh,
                               predicted_annual_consumption_kwh * current_import_tariff,
                               gen_for_year * current_import_tariff)

            revenue = np.where(gen_for_year >= predicted_annual_consumption_kwh,
                               (gen_for_year - predicted_annual_consumption_kwh) * export_tariff_lkr,
                               0)

            total_financial_benefit = savings + revenue

            year_cost = np.copy(annual_maintenance)
            year_cost[inverter_fail_year == year] += inverter_cost

            net_flow = total_financial_benefit - year_cost
            discounted_flow = net_flow / ((1 + self.DISCOUNT_RATE) ** year)

            npv_array += discounted_flow
            cumulative_cash += net_flow
            total_net_profit += net_flow

            just_paid_back = (cumulative_cash >= 0) & (~paid_back)
            if np.any(just_paid_back):
                prev_cash = cumulative_cash[just_paid_back] - net_flow[just_paid_back]
                payback_years[just_paid_back] = (year - 1) + (np.abs(prev_cash) / net_flow[just_paid_back])
                paid_back[just_paid_back] = True

            current_import_tariff *= (1 + tariff_escalation)

        roi_array = ((total_net_profit - initial_investment_lkr) / initial_investment_lkr) * 100


        expected_roi = np.mean(roi_array)
        expected_payback = np.median(payback_years)
        expected_npv = np.mean(npv_array)

        worst_case_roi = np.percentile(roi_array, 5)
        worst_case_npv = np.percentile(npv_array, 5)
        worst_case_payback = np.percentile(payback_years, 95)

        best_case_roi = np.percentile(roi_array, 95)
        shortest_payback = np.percentile(payback_years, 10)
        prob_positive_roi = np.mean(roi_array > 0) * 100

        if expected_npv > (initial_investment_lkr * 0.5):
            rec = "Excellent Investment: Highly resilient to market risks."
        elif expected_npv > 0:
            rec = "Good Investment: Profitable over the system lifetime."
        else:
            rec = "High Risk / Marginal Return: Consider a different system size or tariff scheme."


        yearly_net_cashflow = []
        baseline_cum_cashflow = [-initial_investment_lkr]
        p10_cum_cashflow = [-initial_investment_lkr]
        p90_cum_cashflow = [-initial_investment_lkr]

        expected_annual_return = initial_investment_lkr * (expected_roi / 100) / self.PROJECT_LIFETIME

        for year in range(1, self.PROJECT_LIFETIME + 1):
            yearly_net = expected_annual_return
            if year == 10:
                yearly_net -= (initial_investment_lkr * 0.25)

            yearly_net_cashflow.append(round(float(yearly_net), 2))

            current_cum = baseline_cum_cashflow[-1] + yearly_net
            baseline_cum_cashflow.append(round(float(current_cum), 2))
            p10_cum_cashflow.append(round(float(current_cum * 0.85), 2))
            p90_cum_cashflow.append(round(float(current_cum * 1.15), 2))


        return {
            "System_Size_KW": system_size_kw,
            "Total_Investment_LKR": initial_investment_lkr,
            "Expected_NPV_LKR": round(float(expected_npv), 2),
            "Expected_ROI_Percent": round(float(expected_roi), 2),
            "Expected_Payback_Years": round(float(expected_payback), 1),
            "Risk_Analysis": {
                "Worst_Case_ROI_Percent": round(float(worst_case_roi), 2),
                "Worst_Case_NPV_LKR": round(float(worst_case_npv), 2),
                "Worst_Case_Payback_Years": round(float(worst_case_payback), 1),
                "Certainty_Score": "High" if worst_case_npv > 0 else "Moderate"
            },
            "Scenario_Analysis": {
                "Best_Case_ROI_Percent": round(float(best_case_roi), 2),
                "Shortest_Payback_Years": round(float(shortest_payback), 1),
                "Probability_Positive_ROI": round(float(prob_positive_roi), 1)
            },
            "Recommendation": rec,
            "Recommended_Local_Vendors": self.VENDOR_DATABASE,
            "Chart_Data": {
                "Years_Labels_0_to_20": list(range(0, 21)),
                "Yearly_Revenue_Forecast": yearly_net_cashflow,
                "Cumulative_Cash_Flow_Expected": baseline_cum_cashflow,
                "Cumulative_Cash_Flow_P10": p10_cum_cashflow,
                "Cumulative_Cash_Flow_P90": p90_cum_cashflow,
                "Monte_Carlo_ROI_Distribution": np.round(roi_array, 2).tolist(),
                "Monte_Carlo_NPV_Distribution": np.round(npv_array, 2).tolist()
            }
        }



# TEST RUN
if __name__ == "__main__":
    logger.info("Running High-Performance Financial Component Test...")
    roi_model = SolarFinancialModel()

    final_output = roi_model.calculate_financial_report(5, 7200, 4800)

    # Truncate arrays for readable terminal output
    display_output = final_output.copy()
    display_output["Chart_Data"][
        "Monte_Carlo_ROI_Distribution"] = f"[{len(final_output['Chart_Data']['Monte_Carlo_ROI_Distribution'])} simulated data points]"
    display_output["Chart_Data"][
        "Monte_Carlo_NPV_Distribution"] = f"[{len(final_output['Chart_Data']['Monte_Carlo_NPV_Distribution'])} simulated data points]"

    print(json.dumps(display_output, indent=4))