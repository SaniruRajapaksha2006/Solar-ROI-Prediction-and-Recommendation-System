import numpy as np
import pandas as pd
import json


class SolarFinancialModel:
    def __init__(self):
        # =========================================================================
        # 1. MARKET DATA (Source: Local Vendor Analysis 2024-2026)
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
        # 2. LOCAL VENDOR DATABASE (Maharagama Area)
        # =========================================================================
        self.VENDOR_DATABASE = [
            {
                "Name": "Genso Power Technologies",
                "Location": "Maharagama",
                "Contact": "011 2 000 000",
                "Specialty": "Residential & Commercial Solar"
            },
            {
                "Name": "Mega Solar (Pvt) Ltd",
                "Location": "Maharagama",
                "Contact": "011 2 111 111",
                "Specialty": "Net Accounting & Hybrid Systems"
            },
            {
                "Name": "Growatt Lanka",
                "Location": "Maharagama",
                "Contact": "011 2 222 222",
                "Specialty": "Inverters & Turnkey Solar Solutions"
            }
        ]

        # =========================================================================
        # 3. POLICY & TARIFF DATA (Sri Lanka CEB/PUCSL)
        # =========================================================================
        self.BASE_IMPORT_TARIFF_LKR = 45.00
        self.PROJECT_LIFETIME = 20  # Years
        self.DISCOUNT_RATE = 0.10  # 10% Discount rate for NPV

    def get_system_cost(self, size_kw):
        """Estimates total system cost based on size."""
        if size_kw in self.PRICING_DATABASE:
            return self.PRICING_DATABASE[size_kw]
        else:
            return size_kw * 200000

    def get_export_tariff(self, size_kw):
        """Returns the fixed export tariff based on system size"""
        if size_kw <= 5:
            return 20.90  # LKR per kWh for 0-5 kW
        else:
            return 19.61  # LKR per kWh for 5-20 kW

    def calculate_financial_report(self, system_size_kw, predicted_annual_generation_kwh,
                                   predicted_annual_consumption_kwh):
        """
        MAIN COMPONENT FUNCTION
        """
        initial_investment_lkr = self.get_system_cost(system_size_kw)
        export_tariff_lkr = self.get_export_tariff(system_size_kw)

        # ---------------------------------------------------------
        # 1. MONTE CARLO SIMULATION
        # ---------------------------------------------------------
        n_simulations = 2000
        sim_results = []

        for _ in range(n_simulations):
            # --- Randomize Uncertain Risk Variables ---
            degradation_rate = np.random.uniform(0.005, 0.010)
            base_maintenance = initial_investment_lkr * 0.01
            annual_maintenance = base_maintenance * np.random.normal(1.0, 0.2)
            inverter_fail_year = np.random.randint(8, 13)
            inverter_cost = initial_investment_lkr * 0.25
            import_tariff_escalation = np.random.uniform(0.02, 0.05)

            # --- Cash Flow Projection ---
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

                # Costs for the year
                year_cost = annual_maintenance
                if year == inverter_fail_year:
                    year_cost += inverter_cost

                # Net Cash Flow & NPV
                net_flow = total_financial_benefit - year_cost
                discounted_flow = net_flow / ((1 + self.DISCOUNT_RATE) ** year)

                npv += discounted_flow
                cumulative_cash += net_flow
                total_net_profit += net_flow

                # Precise Fractional Payback Calculation
                if cumulative_cash >= 0 and not paid_back:
                    prev_cash = cumulative_cash - net_flow
                    payback_year = (year - 1) + (abs(prev_cash) / net_flow)
                    paid_back = True

                # Escalate CEB import tariff ONLY
                current_import_tariff *= (1 + import_tariff_escalation)

            roi_percent = (total_net_profit / initial_investment_lkr) * 100
            sim_results.append({
                "ROI": roi_percent,
                "Payback": payback_year,
                "NPV": npv
            })

        # ---------------------------------------------------------
        # 2. AGGREGATING RESULTS
        # ---------------------------------------------------------
        df_sim = pd.DataFrame(sim_results)

        expected_roi = df_sim["ROI"].mean()
        expected_payback = df_sim["Payback"].median()
        expected_npv = df_sim["NPV"].mean()

        # Risk metrics
        worst_case_roi = df_sim["ROI"].quantile(0.05)
        worst_case_npv = df_sim["NPV"].quantile(0.05)
        worst_case_payback = df_sim["Payback"].quantile(0.95)

        # Advanced Scenario Analysis Metrics for Frontend
        best_case_roi = df_sim["ROI"].quantile(0.95)
        shortest_payback = df_sim["Payback"].quantile(0.10)
        prob_positive_roi = (df_sim["ROI"] > 0).mean() * 100

        # Recommendation based on NPV
        if expected_npv > (initial_investment_lkr * 0.5):
            rec = "Excellent Investment: Highly resilient to market risks."
        elif expected_npv > 0:
            rec = "Good Investment: Profitable over the system lifetime."
        else:
            rec = "High Risk / Marginal Return: Consider a different system size or tariff scheme."

        # ---------------------------------------------------------
        # 3. GENERATING DATA FOR FRONTEND CHARTS (NEW FOR COMMIT 10)
        # ---------------------------------------------------------
        yearly_net_cashflow = []

        # Initialize cumulative tracking arrays for Year 0
        baseline_cum_cashflow = [-initial_investment_lkr]
        p10_cum_cashflow = [-initial_investment_lkr]
        p90_cum_cashflow = [-initial_investment_lkr]

        expected_annual_return = initial_investment_lkr * (expected_roi / 100) / self.PROJECT_LIFETIME

        for year in range(1, self.PROJECT_LIFETIME + 1):
            yearly_net = expected_annual_return
            if year == 10:  # Inverter Replacement Dip
                yearly_net -= (initial_investment_lkr * 0.25)

            yearly_net_cashflow.append(round(yearly_net, 2))

            # Step the cumulative totals forward
            current_cum = baseline_cum_cashflow[-1] + yearly_net
            baseline_cum_cashflow.append(round(current_cum, 2))

            # Synthetic confidence bounds (P10 = 15% worse, P90 = 15% better) for shaded line chart
            p10_cum_cashflow.append(round(current_cum * 0.85, 2))
            p90_cum_cashflow.append(round(current_cum * 1.15, 2))

        # Extract Monte Carlo arrays for the Histograms
        roi_distribution = df_sim["ROI"].round(2).tolist()
        npv_distribution = df_sim["NPV"].round(2).tolist()

        # ---------------------------------------------------------
        # 4. RETURN FINAL JSON
        # ---------------------------------------------------------
        return {
            "System_Size_KW": system_size_kw,
            "Total_Investment_LKR": initial_investment_lkr,
            "Expected_NPV_LKR": round(expected_npv, 2),
            "Expected_ROI_Percent": round(expected_roi, 2),
            "Expected_Payback_Years": round(expected_payback, 1),
            "Risk_Analysis": {
                "Worst_Case_ROI_Percent": round(worst_case_roi, 2),
                "Worst_Case_NPV_LKR": round(worst_case_npv, 2),
                "Worst_Case_Payback_Years": round(worst_case_payback, 1),
                "Certainty_Score": "High" if worst_case_npv > 0 else "Moderate"
            },
            "Scenario_Analysis": {
                "Best_Case_ROI_Percent": round(best_case_roi, 2),
                "Shortest_Payback_Years": round(shortest_payback, 1),
                "Probability_Positive_ROI": round(prob_positive_roi, 1)
            },
            "Recommendation": rec,
            "Recommended_Local_Vendors": self.VENDOR_DATABASE,

            # --- THE EXACT DATA YOUR HTML CHARTS NEED ---
            "Chart_Data": {
                "Years_Labels_0_to_20": list(range(0, 21)),
                "Yearly_Revenue_Forecast": yearly_net_cashflow,
                "Cumulative_Cash_Flow_Expected": baseline_cum_cashflow,
                "Cumulative_Cash_Flow_P10": p10_cum_cashflow,
                "Cumulative_Cash_Flow_P90": p90_cum_cashflow,
                "Monte_Carlo_ROI_Distribution": roi_distribution,
                "Monte_Carlo_NPV_Distribution": npv_distribution
            }
        }


# =========================================================
# TEST RUN
# =========================================================
roi_model = SolarFinancialModel()

input_size = 5
input_gen = 7200
input_consumption = 4800

final_output = roi_model.calculate_financial_report(input_size, input_gen, input_consumption)

# Temporarily truncate the massive arrays for terminal viewing readability
display_output = final_output.copy()
display_output["Chart_Data"][
    "Monte_Carlo_ROI_Distribution"] = f"[{len(final_output['Chart_Data']['Monte_Carlo_ROI_Distribution'])} simulated data points...]"
display_output["Chart_Data"][
    "Monte_Carlo_NPV_Distribution"] = f"[{len(final_output['Chart_Data']['Monte_Carlo_NPV_Distribution'])} simulated data points...]"

print(json.dumps(display_output, indent=4))