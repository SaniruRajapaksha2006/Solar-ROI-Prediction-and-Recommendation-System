import numpy as np
import pandas as pd
import json


class SolarFinancialModel:
    def __init__(self):
        # =========================================================================
        # 1. MARKET DATA (Source: Local Vendor Analysis 2024-2025)
        # =========================================================================
        self.PRICING_DATABASE = {
            3: 750000,  # 3kW Single Phase ~750k - 900k
            5: 1050000,  # 5kW Single/Three Phase ~975k - 1.2M
            8: 1450000,  # 8kW ~1.4M - 1.5M
            10: 1800000,  # 10kW ~1.7M - 1.9M
            15: 2600000,  # 15kW ~2.5M - 2.8M
            20: 2900000  # 20kW ~2.8M - 3.0M
        }

        # =========================================================================
        # 2. POLICY & TARIFF DATA (Source: PUCSL / CEB Revisions)
        # =========================================================================
        self.EXPORT_TARIFF_LKR = 27.06

        # NEW: Average blended import tariff (what the user saves by generating their own power)
        self.BASE_IMPORT_TARIFF_LKR = 45.00

        self.PROJECT_LIFETIME = 20  # Years
        self.DISCOUNT_RATE = 0.10  # 10% (Avg lending rate / inflation adjust)

    def get_system_cost(self, size_kw):
        if size_kw in self.PRICING_DATABASE:
            return self.PRICING_DATABASE[size_kw]
        else:
            return size_kw * 200000

    # NEW: Added predicted_annual_consumption_kwh parameter
    def calculate_financial_report(self, system_size_kw, predicted_annual_generation_kwh,
                                   predicted_annual_consumption_kwh):
        """
        MAIN COMPONENT FUNCTION
        """
        # 1. Deterministic Cost Estimation
        initial_investment_lkr = self.get_system_cost(system_size_kw)

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
            payback_year = 22
            paid_back = False
            total_net_profit = 0

            # Temporarily keeping escalation on export for this step (will fix in next commit)
            current_export_tariff = self.EXPORT_TARIFF_LKR
            current_import_tariff = self.BASE_IMPORT_TARIFF_LKR

            for year in range(1, self.PROJECT_LIFETIME + 1):
                gen_for_year = predicted_annual_generation_kwh * ((1 - degradation_rate) ** (year - 1))

                # NEW: Net Accounting Logic
                if gen_for_year >= predicted_annual_consumption_kwh:
                    savings = predicted_annual_consumption_kwh * current_import_tariff
                    excess_exported = gen_for_year - predicted_annual_consumption_kwh
                    revenue = excess_exported * current_export_tariff
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

                # Check Payback
                if cumulative_cash >= 0 and not paid_back:
                    payback_year = year + (cumulative_cash - net_flow) / net_flow
                    paid_back = True

                # Update Tariffs for next year (Will fix export flat-rate bug in Commit 2)
                current_export_tariff *= (1 + tariff_escalation)
                current_import_tariff *= (1 + tariff_escalation)

            roi_percent = (total_net_profit / initial_investment_lkr) * 100
            sim_results.append({
                "ROI": roi_percent,
                "Payback": payback_year
            })

        # 3. Aggregating Results
        df_sim = pd.DataFrame(sim_results)
        expected_roi = df_sim["ROI"].mean()
        expected_payback = df_sim["Payback"].median()
        worst_case_roi = df_sim["ROI"].quantile(0.05)

        # 4. Generate Recommendation
        if expected_roi > 150:
            rec = "Excellent Investment"
        elif expected_roi > 50:
            rec = "Good Investment"
        else:
            rec = "High Risk / Marginal Return"

        return {
            "System_Size_KW": system_size_kw,
            "Total_Investment_LKR": initial_investment_lkr,
            "Expected_ROI_Percent": round(expected_roi, 2),
            "Payback_Period_Years": round(expected_payback, 1),
            "Risk_Analysis": {
                "Worst_Case_ROI": round(worst_case_roi, 2),
                "Certainty": "High" if expected_roi > 50 else "Moderate"
            },
            "Recommendation": rec
        }


# =========================================================
# TEST RUN
# =========================================================
roi_model = SolarFinancialModel()

input_size = 5
input_gen = 7200
input_consumption = 4800  # NEW: Added mock consumption from Component 3

final_output = roi_model.calculate_financial_report(input_size, input_gen, input_consumption)
print(json.dumps(final_output, indent=4))