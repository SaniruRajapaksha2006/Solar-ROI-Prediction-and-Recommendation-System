"""
suitability_engine.py
Rule-based transformer suitability scoring engine.

Scoring dimensions (weighted):
  - Headroom capacity   40 %
  - Distance proximity  30 %
  - Grid stability      30 %

Outputs a 0–100 blended score plus detailed headroom metrics and a curtailment
risk flag.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Configuration ────────────────────────────────────────────────────────────

SAFETY_MARGIN          = 0.80   # maximum safe loading fraction
CURTAILMENT_THRESHOLD  = 0.75   # utilisation above which curtailment is likely

# Score weights
W_HEADROOM   = 0.40
W_DISTANCE   = 0.30
W_STABILITY  = 0.30


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class HeadroomAnalysis:
    transformer_capacity_kW: float
    current_load_kW: float
    existing_solar_kW: float
    total_before_new_kW: float
    available_headroom_kW: float
    safe_headroom_kW: float
    new_solar_request_kW: float
    total_after_new_kW: float
    can_support: bool
    curtailment_risk: bool
    utilization_before: float   # fraction 0-1
    utilization_after: float    # fraction 0-1
    headroom_margin_pct: float  # how much buffer remains after solar, as % of capacity


@dataclass
class SuitabilityScore:
    overall_score: float        # 0–100
    headroom_score: float
    distance_score: float
    stability_score: float
    headroom_analysis: HeadroomAnalysis
    suitability_label: str      # "IDEAL" | "GOOD" | "FAIR" | "POOR"


# ─── Engine ──────────────────────────────────────────────────────────────────

class TransformerSuitabilityEngine:
    """
    Evaluates a single transformer row against a proposed solar installation.

    Usage:
        engine = TransformerSuitabilityEngine(solar_kW=5.0)
        result = engine.evaluate(transformer_row, distance_m=240.5)
    """

    def __init__(self, solar_kW: float):
        if solar_kW <= 0:
            raise ValueError("solar_kW must be positive")
        self.solar_kW = solar_kW

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, transformer_row, distance_m: float) -> SuitabilityScore:
        headroom = self._headroom_analysis(transformer_row)
        h_score  = self._headroom_score(headroom)
        d_score  = self._distance_score(distance_m)
        s_score  = self._stability_score(headroom)

        overall = (
            h_score * W_HEADROOM
            + d_score * W_DISTANCE
            + s_score * W_STABILITY
        )

        return SuitabilityScore(
            overall_score     = round(overall, 2),
            headroom_score    = round(h_score, 2),
            distance_score    = round(d_score, 2),
            stability_score   = round(s_score, 2),
            headroom_analysis = headroom,
            suitability_label = self._label(overall),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _headroom_analysis(self, row) -> HeadroomAnalysis:
        cap           = float(row['ESTIMATED_CAPACITY_kW'])
        current_load  = float(row['current_load_kW'])
        existing_solar = float(row['total_solar_capacity'])

        total_before    = current_load + existing_solar
        available       = cap - total_before
        safe_headroom   = cap * SAFETY_MARGIN - total_before
        total_after     = total_before + self.solar_kW

        can_support     = available >= self.solar_kW
        curtailment     = (total_after / cap) > CURTAILMENT_THRESHOLD

        headroom_margin_pct = max(0.0, (available - self.solar_kW) / cap * 100)

        return HeadroomAnalysis(
            transformer_capacity_kW = cap,
            current_load_kW         = current_load,
            existing_solar_kW       = existing_solar,
            total_before_new_kW     = total_before,
            available_headroom_kW   = available,
            safe_headroom_kW        = safe_headroom,
            new_solar_request_kW    = self.solar_kW,
            total_after_new_kW      = total_after,
            can_support             = can_support,
            curtailment_risk        = curtailment,
            utilization_before      = total_before / cap,
            utilization_after       = total_after / cap,
            headroom_margin_pct     = headroom_margin_pct,
        )

    @staticmethod
    def _headroom_score(h: HeadroomAnalysis) -> float:
        """Score 0-100 based on safe headroom vs requested solar."""
        ratio = h.safe_headroom_kW / max(h.new_solar_request_kW, 0.001)
        if ratio >= 1.5:   return 100.0
        if ratio >= 1.0:   return 80.0
        # fall back to available (not safe) headroom
        avail_ratio = h.available_headroom_kW / max(h.new_solar_request_kW, 0.001)
        if avail_ratio >= 1.0: return 50.0
        return 0.0

    @staticmethod
    def _distance_score(distance_m: float) -> float:
        """Exponential decay: 100 at 0 m, ~37 at 1 000 m."""
        return max(0.0, 100 * math.exp(-distance_m / 1000))

    @staticmethod
    def _stability_score(h: HeadroomAnalysis) -> float:
        """Score 0-100 based on post-connection utilisation."""
        u = h.utilization_after
        if u <= 0.70:   return 100.0
        if u <= 0.85:   return 75.0
        if u <= 0.95:   return 40.0
        return 0.0

    @staticmethod
    def _label(score: float) -> str:
        if score >= 80: return "IDEAL"
        if score >= 60: return "GOOD"
        if score >= 40: return "FAIR"
        return "POOR"


# ─── Recommendation generator ─────────────────────────────────────────────────

def generate_recommendation(score: float, headroom: HeadroomAnalysis) -> str:
    if score >= 80:
        return (
            f"Highly suitable. Transformer has {headroom.available_headroom_kW:.1f} kW "
            f"of available headroom. Proceed with grid connection."
        )
    if score >= 60:
        return (
            f"Conditionally suitable. Post-connection utilisation will be "
            f"{headroom.utilization_after * 100:.1f}%. Review with your utility provider."
        )
    if score >= 40:
        if headroom.curtailment_risk:
            return (
                "Fair option but curtailment risk is elevated. "
                "Consider scheduling generation during off-peak hours or choosing an alternative transformer."
            )
        return (
            "Marginal suitability due to limited headroom. "
            "An upgrade assessment is recommended before connecting."
        )
    return (
        f"Not suitable. Only {headroom.available_headroom_kW:.1f} kW headroom available "
        f"against {headroom.new_solar_request_kW:.1f} kW requested. "
        "Select a different transformer."
    )