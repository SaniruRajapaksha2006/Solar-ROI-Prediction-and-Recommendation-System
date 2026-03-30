"""
Shared data models for all components
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

class UserInput(BaseModel):
    # User input shared across all components
    latitude: float
    longitude: float
    consumption_months: Dict[int, float]  # month -> kWh
    tariff: str = "D1"
    phase: str = "SP"
    household_size: int = 4
    year: int = 2025
    search_radius_m: float = 2000.0  # For transformer search

class SolarGenerationForecast(BaseModel):
    # Output from Component 1
    monthly_export_kwh: Dict[int, float]
    monthly_income_lkr: Dict[int, float]
    monthly_confidence_pct: Dict[int, float]
    annual_total_kwh: float
    annual_income_lkr: float
    avg_confidence_pct: float
    panel_size_kw: float

class TransformerInfo(BaseModel):
    # Output from Component 2
    transformer_id: str
    lat: float = 0.0  # Add latitude for map
    lon: float = 0.0  # Add longitude for map
    distance_m: float
    suitability_score: float
    suitability_label: str
    capacity_kw: float
    current_load_kw: float
    available_headroom_kw: float
    can_support: bool
    curtailment_risk: bool
    recommendation: str
    utilization_after: float = 0.0  # Add for map tooltip

class ConsumptionForecast(BaseModel):
    # Output from Component 3
    monthly_consumption_kwh: Dict[int, float]
    monthly_bills_lkr: Dict[int, float]
    annual_total_kwh: float
    annual_total_bill_lkr: float
    avg_confidence: float
    method: str
    similar_households_count: int

class ROIAnalysis(BaseModel):
    # Output from Component 4
    total_investment_lkr: float
    expected_roi_percent: float
    expected_payback_years: float
    recommendation: str

class IntegratedResult(BaseModel):
    user_input: UserInput
    solar_forecast: SolarGenerationForecast
    transformer_info: Optional[TransformerInfo]
    all_transformers: List[TransformerInfo] = []
    consumption_forecast: ConsumptionForecast
    roi_analysis: ROIAnalysis
    recommended_panel_size_kw: float
    timestamp: datetime = datetime.now()