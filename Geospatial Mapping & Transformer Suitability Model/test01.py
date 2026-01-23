import warnings
from typing import Tuple, List, Dict

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

warnings.filterwarnings('ignore')


# ============================================================================
# 1. COORDINATE EXTRACTION & GEOCODING
# ============================================================================

class CoordinateExtractor:
    """Extract user coordinates from address or return existing lat/lon"""

    def __init__(self):
        self.geocoder = Nominatim(user_agent="solar_transformer_mapper")

    def get_coordinates(self, location_input: str) -> Tuple[float, float]:
        """
        Get coordinates from address or validate existing coordinates
        Args:
            location_input: Address string or "lat,lon" format
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            # Try parsing as coordinates
            if ',' in location_input:
                parts = location_input.split(',')
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

            # Geocode address
            location = self.geocoder.geocode(location_input)
            if location:
                return (location.latitude, location.longitude)
            else:
                raise ValueError(f"Could not find coordinates for: {location_input}")
        except Exception as e:
            raise ValueError(f"Coordinate extraction error: {str(e)}")

    def reverse_geocode(self, lat: float, lon: float) -> str:
        """Get address from coordinates"""
        try:
            location = self.geocoder.reverse(f"{lat}, {lon}")
            return location.address
        except:
            return "Address not found"


# ============================================================================
# 2. TRANSFORMER MAPPING & DATA PROCESSING
# ============================================================================

class TransformerMapper:
    """Map and analyze transformer data"""

    def __init__(self, transformer_csv_path: str):
        """
        Initialize with transformer dataset
        Args:
            transformer_csv_path: Path to CSV with transformer data
        """
        self.df = pd.read_csv(transformer_csv_path)
        self.transformers = self._process_transformer_data()

    def _process_transformer_data(self) -> pd.DataFrame:
        """Process and aggregate transformer data"""
        # Group by transformer and calculate statistics
        transformer_stats = self.df.groupby('TRANSFORMER_CODE').agg({
            'TRANSFORMER_LAT': 'first',
            'TRANSFORMER_LON': 'first',
            'NET_CONSUMPTION_kWh': 'mean',
            'HAS_SOLAR': 'sum',
            'INV_CAPACITY': 'sum',
            'IMPORT_kWh': 'mean',
            'EXPORT_kWh': 'mean'
        }).reset_index()

        # Estimate transformer capacity (simplified: based on typical distribution transformer)
        # Standard assumption: 50 kVA = 40 kW continuous capacity for residential
        transformer_stats['ESTIMATED_CAPACITY_kW'] = 40

        return transformer_stats

    def find_nearby_transformers(
            self,
            user_lat: float,
            user_lon: float,
            radius_meters: float = 500
    ) -> pd.DataFrame:
        """
        Find transformers within specified radius
        Args:
            user_lat, user_lon: User coordinates
            radius_meters: Search radius
        Returns:
            DataFrame of nearby transformers with distances
        """
        user_coords = (user_lat, user_lon)

        def calc_distance(row):
            tf_coords = (row['TRANSFORMER_LAT'], row['TRANSFORMER_LON'])
            return geodesic(user_coords, tf_coords).meters

        self.transformers['DISTANCE_M'] = self.transformers.apply(calc_distance, axis=1)
        nearby = self.transformers[self.transformers['DISTANCE_M'] <= radius_meters].copy()

        return nearby.sort_values('DISTANCE_M')


# ============================================================================
# 3. TRANSFORMER HEADROOM & SUITABILITY CALCULATION
# ============================================================================

class TransformerSuitability:
    """Assess transformer suitability for solar connection"""

    # Configuration parameters
    SAFETY_MARGIN = 0.8  # Use 80% of capacity as max safe load
    CURTAILMENT_THRESHOLD = 0.75  # Flag if >75% capacity used

    def __init__(self, solar_forecast_kW: float):
        """
        Initialize with expected solar generation
        Args:
            solar_forecast_kW: Predicted peak solar capacity in kW
        """
        self.solar_forecast_kW = solar_forecast_kW

    def calculate_headroom(self, transformer_row: pd.Series) -> Dict:
        """
        Calculate available capacity headroom
        Args:
            transformer_row: Row from transformer DataFrame
        Returns:
            Dictionary with capacity analysis
        """
        capacity = transformer_row['ESTIMATED_CAPACITY_kW']
        current_load = transformer_row['NET_CONSUMPTION_kWh'] * 0.85  # Convert avg consumption to approx load
        existing_solar = transformer_row['INV_CAPACITY']

        available_capacity = capacity - (current_load + existing_solar)
        safe_headroom = capacity * self.SAFETY_MARGIN - (current_load + existing_solar)

        can_support = available_capacity >= self.solar_forecast_kW
        curtailment_risk = (
                                       current_load + existing_solar + self.solar_forecast_kW) / capacity > self.CURTAILMENT_THRESHOLD

        return {
            'transformer_capacity_kW': capacity,
            'current_load_kW': current_load,
            'existing_solar_kW': existing_solar,
            'total_before_new_kW': current_load + existing_solar,
            'available_headroom_kW': available_capacity,
            'safe_headroom_kW': safe_headroom,
            'new_solar_request_kW': self.solar_forecast_kW,
            'total_after_new_kW': current_load + existing_solar + self.solar_forecast_kW,
            'can_support': can_support,
            'curtailment_risk': curtailment_risk,
            'utilization_before': (current_load + existing_solar) / capacity,
            'utilization_after': (current_load + existing_solar + self.solar_forecast_kW) / capacity
        }

    def score_suitability(self, transformer_row: pd.Series, distance_m: float) -> Dict:
        """
        Calculate overall suitability score
        Args:
            transformer_row: Transformer data
            distance_m: Distance from user
        Returns:
            Suitability score and factors
        """
        headroom = self.calculate_headroom(transformer_row)

        # Scoring factors (0-100 scale)
        # 1. Headroom factor (40% weight)
        if headroom['safe_headroom_kW'] >= self.solar_forecast_kW * 1.5:
            headroom_score = 100
        elif headroom['safe_headroom_kW'] >= self.solar_forecast_kW:
            headroom_score = 80
        elif headroom['available_headroom_kW'] >= self.solar_forecast_kW:
            headroom_score = 50
        else:
            headroom_score = 0

        # 2. Distance factor (30% weight) - closer is better
        # 100 at 0m, 50 at 500m, 0 at 1000m+
        distance_score = max(0, 100 - (distance_m / 10))

        # 3. Grid stability factor (30% weight)
        if headroom['utilization_after'] <= 0.7:
            stability_score = 100
        elif headroom['utilization_after'] <= 0.85:
            stability_score = 75
        elif headroom['utilization_after'] <= 0.95:
            stability_score = 40
        else:
            stability_score = 0

        # Weighted overall score
        overall_score = (
                headroom_score * 0.40 +
                distance_score * 0.30 +
                stability_score * 0.30
        )

        return {
            'overall_score': overall_score,
            'headroom_score': headroom_score,
            'distance_score': distance_score,
            'stability_score': stability_score,
            'headroom_analysis': headroom
        }


# ============================================================================
# 4. VISUALIZATION & REPORTING
# ============================================================================

class TransformerMapper_Visualizer:
    """Create maps and reports"""

    @staticmethod
    def create_map(
            user_lat: float,
            user_lon: float,
            nearby_transformers: pd.DataFrame,
            suitability_scores: List[Dict],
            output_file: str = "transformer_map.html"
    ) -> folium.Map:
        """Create interactive map with transformers"""

        # Create base map centered on user
        m = folium.Map(
            location=[user_lat, user_lon],
            zoom_start=16,
            tiles='OpenStreetMap'
        )

        # Add user location
        folium.Marker(
            location=[user_lat, user_lon],
            popup="Your Location",
            icon=folium.Icon(color='green', icon='home'),
            tooltip='Solar Installation Site'
        ).add_to(m)

        # Add transformers
        for idx, (_, transformer) in enumerate(nearby_transformers.iterrows()):
            score_info = suitability_scores[idx] if idx < len(suitability_scores) else {}
            overall_score = score_info.get('overall_score', 0)

            # Color based on suitability
            if overall_score >= 80:
                color = 'green'
            elif overall_score >= 50:
                color = 'orange'
            else:
                color = 'red'

            popup_text = f"""
            <b>Transformer: {transformer['TRANSFORMER_CODE']}</b><br>
            <b>Suitability Score: {overall_score:.1f}/100</b><br>
            Distance: {transformer['DISTANCE_M']:.0f} m<br>
            Capacity: {transformer['ESTIMATED_CAPACITY_kW']:.1f} kW<br>
            Current Load: {score_info.get('headroom_analysis', {}).get('current_load_kW', 0):.1f} kW
            """

            folium.Marker(
                location=[transformer['TRANSFORMER_LAT'], transformer['TRANSFORMER_LON']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='plug'),
                tooltip=f"Transformer {transformer['TRANSFORMER_CODE']}"
            ).add_to(m)

        m.save(output_file)
        return m

    @staticmethod
    def generate_report(
            user_location: str,
            user_coords: Tuple[float, float],
            solar_forecast_kW: float,
            nearby_transformers: pd.DataFrame,
            suitability_results: List[Dict]
    ) -> str:
        """Generate detailed text report"""

        report = f"""
================================================================================
                TRANSFORMER SUITABILITY ASSESSMENT REPORT
================================================================================

USER LOCATION
{'-' * 80}
Address: {user_location}
Coordinates: {user_coords[0]:.6f}, {user_coords[1]:.6f}
Predicted Solar Capacity: {solar_forecast_kW:.2f} kW

NEARBY TRANSFORMERS (Within 500m)
{'-' * 80}
Total transformers found: {len(nearby_transformers)}

"""

        for idx, (_, tf) in enumerate(nearby_transformers.iterrows()):
            if idx >= len(suitability_results):
                break

            result = suitability_results[idx]
            headroom = result['headroom_analysis']

            report += f"""
RANK #{idx + 1}: {tf['TRANSFORMER_CODE']}
{'-' * 80}
Distance: {tf['DISTANCE_M']:.1f} meters
Overall Suitability Score: {result['overall_score']:.1f}/100

Capacity Analysis:
  • Transformer Capacity: {headroom['transformer_capacity_kW']:.1f} kW
  • Current Load: {headroom['current_load_kW']:.1f} kW
  • Existing Solar: {headroom['existing_solar_kW']:.1f} kW
  • Your New Solar: {headroom['new_solar_request_kW']:.1f} kW

Load Before Your Installation: {headroom['utilization_before'] * 100:.1f}%
Load After Your Installation: {headroom['utilization_after'] * 100:.1f}%
Available Headroom: {headroom['available_headroom_kW']:.1f} kW
Safe Headroom (80% rule): {headroom['safe_headroom_kW']:.1f} kW

Can Support Connection: {'✓ YES' if headroom['can_support'] else '✗ NO'}
Curtailment Risk: {'⚠ HIGH RISK' if headroom['curtailment_risk'] else '✓ LOW RISK'}

Scoring Breakdown:
  • Headroom Score: {result['headroom_score']:.1f}/100 (40% weight)
  • Distance Score: {result['distance_score']:.1f}/100 (30% weight)
  • Grid Stability Score: {result['stability_score']:.1f}/100 (30% weight)

Recommendation:
"""
            if result['overall_score'] >= 80:
                report += "  ✓ HIGHLY SUITABLE - Proceed with connection\n"
            elif result['overall_score'] >= 50:
                report += "  ⚠ CONDITIONALLY SUITABLE - Review with utility\n"
            else:
                report += "  ✗ NOT SUITABLE - Consider alternative transformers\n"

        report += f"\n{'=' * 80}\n"
        return report


# ============================================================================
# 5. MAIN EXECUTION FUNCTION
# ============================================================================

def assess_transformer_suitability(
        user_location: str,
        solar_forecast_kW: float,
        transformer_csv_path: str,
        search_radius_m: float = 500,
        output_map_file: str = "transformer_map.html"
) -> Dict:
    """
    Main function to run complete assessment
    Args:
        user_location: Address or "lat,lon" format
        solar_forecast_kW: Predicted solar capacity
        transformer_csv_path: Path to transformer CSV
        search_radius_m: Search radius for transformers
        output_map_file: Output HTML map file
    Returns:
        Dictionary with results
    """

    print("🔍 STEP 1: Extracting user coordinates...")
    extractor = CoordinateExtractor()
    user_coords = extractor.get_coordinates(user_location)
    user_address = extractor.reverse_geocode(user_coords[0], user_coords[1])
    print(f"   ✓ User location: {user_address}")
    print(f"   ✓ Coordinates: {user_coords}")

    print("\n📡 STEP 2: Loading transformer data...")
    mapper = TransformerMapper(transformer_csv_path)
    print(f"   ✓ Loaded {len(mapper.transformers)} unique transformers")

    print(f"\n🎯 STEP 3: Finding nearby transformers (within {search_radius_m}m)...")
    nearby = mapper.find_nearby_transformers(user_coords[0], user_coords[1], search_radius_m)
    print(f"   ✓ Found {len(nearby)} nearby transformers")

    print("\n⚡ STEP 4: Assessing suitability...")
    suitability_assessor = TransformerSuitability(solar_forecast_kW)
    suitability_results = []

    for _, tf_row in nearby.iterrows():
        score = suitability_assessor.score_suitability(tf_row, tf_row['DISTANCE_M'])
        suitability_results.append(score)

    # Sort by score descending
    sorted_indices = np.argsort([r['overall_score'] for r in suitability_results])[::-1]
    nearby_sorted = nearby.iloc[sorted_indices].reset_index(drop=True)
    suitability_results_sorted = [suitability_results[i] for i in sorted_indices]

    print(
        f"   ✓ Top transformer: {nearby_sorted.iloc[0]['TRANSFORMER_CODE']} (Score: {suitability_results_sorted[0]['overall_score']:.1f}/100)")

    print("\n🗺️  STEP 5: Creating visualization...")
    visualizer = TransformerMapper_Visualizer()
    visualizer.create_map(
        user_coords[0],
        user_coords[1],
        nearby_sorted,
        suitability_results_sorted,
        output_map_file
    )
    print(f"   ✓ Map saved to: {output_map_file}")

    print("\n📋 STEP 6: Generating report...")
    report = visualizer.generate_report(
        user_address,
        user_coords,
        solar_forecast_kW,
        nearby_sorted,
        suitability_results_sorted
    )
    print(report)

    return {
        'user_coords': user_coords,
        'user_address': user_address,
        'nearby_transformers': nearby_sorted,
        'suitability_scores': suitability_results_sorted,
        'report': report,
        'map_file': output_map_file
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    results = assess_transformer_suitability(
        user_location="6.848917321,79.92456012",  # Colombo, Sri Lanka
        solar_forecast_kW=5.0,  # 5 kW rooftop solar system
        transformer_csv_path=r"C:\Users\dewmi\OneDrive\Documents\IIT\2nd Year\DSGP\MASTER_DATASET_ALL_10TRANSFORMERS.csv",        search_radius_m=500,
        output_map_file="transformer_suitability_map.html"
    )

    print("\n✅ Assessment complete!")