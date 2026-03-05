import numpy as np
import pandas as pd
import json


class SolarFinancialModel:
    def __init__(self):
        # =========================================================================
        # 1. MARKET DATA (Source: Local Vendor Analysis 2024-2025)
        # =========================================================================
        self.PRICING_DATABASE = {
            3: 750000,  # 3kW Single Phase
            5: 1050000,  # 5kW Single/Three Phase
            8: 1450000,  # 8kW
            10: 1800000,  # 10kW
            15: 2600000,  # 15kW
            20: 2900000  # 20kW
        }

        # =========================================================================
        # 2. POLICY & TARIFF DATA (Source: PUCSL / CEB Revisions)
        # =========================================================================
        self.BASE_IMPORT_TARIFF_LKR = 45.00
        self.PROJECT_LIFETIME = 20
        self.DISCOUNT_RATE = 0.10

    def get_system_cost(self, size_kw):
        if size_kw in self.PRICING_DATABASE:
            return self.PRICING_DATABASE[size_kw]
        else:
            return size_kw * 200000

    def get_export_tariff(self, size_kw):
        if size_kw <= 5:
            return 20.90
        else:
            return 19.61

    def calculate_financial_report(self, system_size_kw, predicted_annual_generation_kwh,
                                   predicted_annual_consumption_kwh):
        """
        MAIN COMPONENT FUNCTION
        """
        # 1. Deterministic Cost Estimation
        initial_investment_lkr = self.get_system_cost(system_size_kw)
        export_tariff_lkr = self.get_export_tariff(system_size_kw)

        # 2. Monte Carlo Simulation (Risk Analysis)
        n_simulations = 2000
        sim_results = []

        for _ in range(n_simulations):
            degradation_rate = np.random.uniform(0.005, 0.010)
            base_maintenance = initial_investment_lkr * 0.01
            annual_maintenance = base_maintenance * np.random.normal(1.0, 0.2)
            inverter_fail_year = np.random.randint(8, 13)
            inverter_cost = initial_investment_lkr * 0.25
            tariff_escalation = np.random.uniform(0.02, 0.05)

            # --- Cash Flow Projection (20 Years) ---
            cumulative_cash = -initial_investment_lkr
            payback_year = self.PROJECT_LIFETIME + 1
            paid_back = False
            total_net_profit = 0
            npv = -initial_investment_lkr

            current_import_tariff = self.BASE_IMPORT_TARIFF_LKR

            for year in range(1, self.PROJECT_LIFETIME + 1):
                gen_for_year = predicted_annual_generation_kwh * ((1 - degradation_rate) ** (year - 1))

                # Net Accounting Logic
                if gen_for_year >= predicted_annual_consumption_kwh:
                    savings = predicted_annual_consumption_kwh * current_import_tariff
                    excess_exported = gen_for_year - predicted_annual_consumption_kwh
                    revenue = excess_exported * export_tariff_lkr
                else:
                    savings = gen_for_year * current_import_tariff
                    revenue = 0

                total_financial_benefit = savings + revenue

                # Costs
                year_cost = annual_maintenance
                if year == inverter_fail_year:
                    year_cost += inverter_cost

                # Net Cash Flow
                net_flow = total_financial_benefit - year_cost
                cumulative_cash += net_flow
                total_net_profit += net_flow

                # Discounted Cash Flow for NPV
                discounted_flow = net_flow / ((1 + self.DISCOUNT_RATE) ** year)
                npv += discounted_flow

                # Precise Fractional Payback Calculation
                if cumulative_cash >= 0 and not paid_back:
                    prev_cash = cumulative_cash - net_flow
                    payback_year = (year - 1) + (abs(prev_cash) / net_flow)
                    paid_back = True

                # Escalate CEB import tariff ONLY
                current_import_tariff *= (1 + tariff_escalation)

            roi_percent = (total_net_profit / initial_investment_lkr) * 100
            sim_results.append({
                "ROI": roi_percent,
                "Payback": payback_year,
                "NPV": npv
            })

        # 3. Aggregating Results
        df_sim = pd.DataFrame(sim_results)
        expected_roi = df_sim["ROI"].mean()
        expected_payback = df_sim["Payback"].median()
        expected_npv = df_sim["NPV"].mean()

        # NEW: Comprehensive Worst-Case Risk Metrics
        worst_case_roi = df_sim["ROI"].quantile(0.05)
        worst_case_npv = df_sim["NPV"].quantile(0.05)
        worst_case_payback = df_sim["Payback"].quantile(0.95)  # 95th percentile for longest payback

        # 4. Generate Recommendation
        if expected_npv > (initial_investment_lkr * 0.5):
            rec = "Excellent Investment: Highly resilient to market risks."
        elif expected_npv > 0:
            rec = "Good Investment: Profitable over the system lifetime."
        else:
            rec = "High Risk / Marginal Return: Consider a different system size or tariff scheme."

        return {
            "System_Size_KW": system_size_kw,
            "Total_Investment_LKR": initial_investment_lkr,
            "Expected_NPV_LKR": round(expected_npv, 2),
            "Expected_ROI_Percent": round(expected_roi, 2),
            "Payback_Period_Years": round(expected_payback, 1),
            "Risk_Analysis": {
                "Worst_Case_ROI_Percent": round(worst_case_roi, 2),
                "Worst_Case_NPV_LKR": round(worst_case_npv, 2),
                "Worst_Case_Payback_Years": round(worst_case_payback, 1),
                "Certainty_Score": "High" if worst_case_npv > 0 else "Moderate"  # Stricter certainty check
            },
            "Recommendation": rec
        }


# =========================================================
# TEST RUN
# =========================================================
roi_model = SolarFinancialModel()

input_size = 5
input_gen = 7200
input_consumption = 4800

final_output = roi_model.calculate_financial_report(input_size, input_gen, input_consumption)
print(json.dumps(final_output, indent=4))