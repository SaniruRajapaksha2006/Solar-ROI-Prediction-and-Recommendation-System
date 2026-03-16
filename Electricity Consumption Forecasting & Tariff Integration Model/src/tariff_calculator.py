from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PUCsLTariffCalculator:
    """
    Calculates electricity bills using PUCSL tariff structure
    Supports multiple tariff categories and net metering
    """

    def __init__(self, config: Dict):
        self.config = config
        self.tariff_structures = self._load_tariff_structures()

    def _load_tariff_structures(self) -> Dict[str, Dict]:
        # Load tariff structures from config
        tariff_config = self.config['tariff']
        structures = {}

        # D1 Tariff (Domestic)
        if 'D1' in tariff_config:
            structures['D1'] = tariff_config['D1']
            logger.info("Loaded D1 tariff structure")

        # GP1 Tariff (General Purpose)
        if 'GP1' in tariff_config:
            structures['GP1'] = tariff_config['GP1']
            logger.info("Loaded GP1 tariff structure")

        return structures

    def _get_tariff(self, tariff_code: str) -> Dict:
        # Get tariff structure with fallback to D1
        tariff = self.tariff_structures.get(tariff_code.upper())
        if not tariff:
            logger.warning(f"Tariff {tariff_code} not found, using D1")
            tariff = self.tariff_structures.get('D1', {})
        return tariff

    def _calculate_energy_charge(self, consumption_kwh: float,
                                 blocks: List[Dict]) -> tuple:
        # Calculate energy charge using increasing block rates
        total_charge = 0
        block_details = []
        remaining = consumption_kwh

        for block in blocks:
            if remaining <= 0:
                break

            block_min = block['min']
            block_max = block['max']
            block_rate = block['rate']

            block_size = block_max - block_min + 1
            consumption_in_block = min(remaining, block_size)

            if consumption_in_block > 0:
                block_charge = consumption_in_block * block_rate
                total_charge += block_charge

                block_details.append({
                    'block_range': f"{block_min}-{block_max}",
                    'kwh_in_block': round(consumption_in_block, 2),
                    'rate_lkr_per_kwh': block_rate,
                    'charge_lkr': round(block_charge, 2)
                })

                remaining -= consumption_in_block

        return total_charge, block_details