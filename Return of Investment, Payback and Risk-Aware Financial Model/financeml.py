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

