import numpy as np
import pandas as pd


class SolarFinancialModel:
    def __init__(self):
        # 1. MARKET DATA (Source: Local Vendor Analysis 2024-2025)

        # Approximate turnkey prices (Panels + Inverter + Installation) in LKR
        # Source: Market averages (e.g., Dinapala, Singer Solar, Solitra - early 2025)
        self.PRICING_DATABASE = {
            3: 750000,  # 3kW Single Phase ~750k - 900k
            5: 1050000,  # 5kW Single/Three Phase ~975k - 1.2M
            8: 1450000,  # 8kW ~1.4M - 1.5M
            10: 1800000,  # 10kW ~1.7M - 1.9M
            15: 2600000,  # 15kW ~2.5M - 2.8M
            20: 2900000  # 20kW ~2.8M - 3.0M
        }

        # 2. POLICY & TARIFF DATA (Source: PUCSL / CEB Revisions)

        self.EXPORT_TARIFF_LKR = 27.06

        self.PROJECT_LIFETIME = 20  # Years
        self.DISCOUNT_RATE = 0.10  # 10% (Avg lending rate / inflation adjust)

    def get_system_cost(self, size_kw):
        """
        Estimates total system cost based on size.
        Interpolates price if exact size (e.g., 6kW) is not in database.
        """
        if size_kw in self.PRICING_DATABASE:
            return self.PRICING_DATABASE[size_kw]
        else:
            # Fallback estimation: Approx 200,000 LKR per kW for sizes between gaps
            return size_kw * 200000

    def calculate_financial_report(self, system_size_kw, predicted_annual_generation_kwh):
        """
        MAIN COMPONENT FUNCTION

        Inputs:
          system_size_kw (float): From Grid/Map Component
          predicted_annual_generation_kwh (float): From Solar Forecasting Component

        Returns:
            Dictionary containing ROI, Payback, and Risk Assessment.
        """

        # 1. Deterministic Cost Estimation
        initial_investment_lkr = self.get_system_cost(system_size_kw)

        # 2. Monte Carlo Simulation (Risk Analysis)
        # We simulate 2,000 possible futures to handle uncertainty
        n_simulations = 2000
        sim_results = []

        for _ in range(n_simulations):
            #A. Randomize Uncertain Variables (Risk Factors)

            # Risk 1: Panel Degradation (0.5% to 1.0% per year)
            degradation_rate = np.random.uniform(0.005, 0.010)

            # Risk 2: Maintenance Cost Volatility (Base: 1% of CAPEX, varies +/- 20%)
            base_maintenance = initial_investment_lkr * 0.01
            annual_maintenance = base_maintenance * np.random.normal(1.0, 0.2)

            # Risk 3: Inverter Failure (Occurs between year 8 and 12)
            inverter_fail_year = np.random.randint(8, 13)
            inverter_cost = initial_investment_lkr * 0.25  # Approx 25% of system cost

            # Risk 4: Tariff/Policy Fluctuation (2% - 5% annual increase in value)
            tariff_escalation = np.random.uniform(0.02, 0.05)

            #B. Cash Flow Projection (20 Years)
            cumulative_cash = -initial_investment_lkr
            payback_year = 22  # Default if never pays back
            paid_back = False
            total_net_profit = 0

            current_tariff = self.EXPORT_TARIFF_LKR

            for year in range(1, self.PROJECT_LIFETIME + 1):
                # Energy Generation (Declines with age)
                gen_for_year = predicted_annual_generation_kwh * ((1 - degradation_rate) ** (year - 1))

                # Revenue (Net Plus Scheme)
                revenue = gen_for_year * current_tariff

                # Costs
                year_cost = annual_maintenance
                if year == inverter_fail_year:
                    year_cost += inverter_cost

                # Net Cash Flow
                net_flow = revenue - year_cost

                # Financial Tracking
                cumulative_cash += net_flow
                total_net_profit += net_flow

                # Check Payback
                if cumulative_cash >= 0 and not paid_back:
                    payback_year = year + (cumulative_cash - net_flow) / net_flow  # Fractional year
                    paid_back = True

                # Update Tariff for next year
                current_tariff = current_tariff * (1 + tariff_escalation)

            # Calculate Metrics for this simulation
            roi_percent = (total_net_profit / initial_investment_lkr) * 100
            sim_results.append({
                "ROI": roi_percent,
                "Payback": payback_year
            })

        # 3. Aggregating Results
        df_sim = pd.DataFrame(sim_results)

        expected_roi = df_sim["ROI"].mean()
        expected_payback = df_sim["Payback"].median()
        worst_case_roi = df_sim["ROI"].quantile(0.05)  # 5th percentile

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


# TEST RUN


# 1. Initialize the model
roi_model = SolarFinancialModel()

# 2. Simulate inputs coming from group members
input_size = 5  # From Component 2
input_gen = 7200  # From Component 1 (5kW * 120kWh * 12 months approx)

# 3. Run the calculation
final_output = roi_model.calculate_financial_report(input_size, input_gen)

# 4. Print the result
import json

print(json.dumps(final_output, indent=4))