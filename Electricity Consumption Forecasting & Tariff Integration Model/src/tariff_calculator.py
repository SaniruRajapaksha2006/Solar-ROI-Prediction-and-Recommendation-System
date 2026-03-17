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

    def calculate_monthly_bill(self, consumption_kwh: float,
                               tariff_code: str = 'D1') -> Dict:
        # Calculate monthly electricity bill WITHOUT solar

        # Get tariff structure
        tariff = self._get_tariff(tariff_code)

        # Step 1: Calculate energy charge by blocks
        energy_charge, block_details = self._calculate_energy_charge(
            consumption_kwh, tariff['blocks']
        )

        # Step 2: Fixed charge
        fixed_charge = tariff['fixed_charge']

        # Step 3: Fuel adjustment charge
        fuel_adjustment = energy_charge * tariff['fuel_adjustment']

        # Step 4: Subtotal before VAT
        subtotal = energy_charge + fixed_charge + fuel_adjustment

        # Step 5: VAT
        vat = subtotal * tariff['vat_rate']

        # Step 6: Total bill
        total_bill = subtotal + vat

        # Step 7: Effective rate
        effective_rate = total_bill / consumption_kwh if consumption_kwh > 0 else 0

        result = {
            'consumption_kwh': round(consumption_kwh, 2),
            'tariff_category': tariff_code.upper(),
            'energy_charge_lkr': round(energy_charge, 2),
            'fixed_charge_lkr': round(fixed_charge, 2),
            'fuel_adjustment_lkr': round(fuel_adjustment, 2),
            'subtotal_lkr': round(subtotal, 2),
            'vat_lkr': round(vat, 2),
            'total_bill_lkr': round(total_bill, 2),
            'effective_rate_lkr_per_kwh': round(effective_rate, 2),
            'block_details': block_details
        }

        return result

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

    def calculate_annual_bills(self, monthly_consumption: Dict[int, float],
                               tariff_code: str = 'D1') -> Dict:
        # Calculate bills for all 12 months
        monthly_bills = {}
        annual_summary = {
            'total_consumption_kwh': 0,
            'total_energy_charge_lkr': 0,
            'total_fixed_charge_lkr': 0,
            'total_fuel_adjustment_lkr': 0,
            'total_vat_lkr': 0,
            'total_bill_lkr': 0
        }

        for month in range(1, 13):
            consumption = monthly_consumption.get(month, 0)
            bill = self.calculate_monthly_bill(consumption, tariff_code)
            monthly_bills[month] = bill

            # Update annual summary
            annual_summary['total_consumption_kwh'] += consumption
            annual_summary['total_energy_charge_lkr'] += bill['energy_charge_lkr']
            annual_summary['total_fixed_charge_lkr'] += bill['fixed_charge_lkr']
            annual_summary['total_fuel_adjustment_lkr'] += bill['fuel_adjustment_lkr']
            annual_summary['total_vat_lkr'] += bill['vat_lkr']
            annual_summary['total_bill_lkr'] += bill['total_bill_lkr']

        # Calculate averages
        annual_summary['monthly_average_consumption_kwh'] = \
            annual_summary['total_consumption_kwh'] / 12
        annual_summary['monthly_average_bill_lkr'] = \
            annual_summary['total_bill_lkr'] / 12

        # Calculate effective annual rate
        if annual_summary['total_consumption_kwh'] > 0:
            annual_summary['effective_rate_lkr_per_kwh'] = \
                annual_summary['total_bill_lkr'] / annual_summary['total_consumption_kwh']
        else:
            annual_summary['effective_rate_lkr_per_kwh'] = 0

        return {
            'monthly_bills': monthly_bills,
            'annual_summary': annual_summary
        }

    def get_highest_block_rate(self, tariff_code: str = 'D1') -> float:
        # Get highest block rate for export credit calculation
        tariff = self._get_tariff(tariff_code)
        blocks = tariff.get('blocks', [])
        return max(block['rate'] for block in blocks) if blocks else 0

    def get_fixed_charge(self, tariff_code: str = 'D1') -> float:
        # Get fixed charge for tariff
        tariff = self._get_tariff(tariff_code)
        return tariff.get('fixed_charge', 180)

    def format_monthly_bill_for_display(self, bill: Dict) -> str:
        # Format monthly bill for display
        lines = []
        lines.append("\n MONTHLY ELECTRICITY BILL")
        lines.append("=" * 60)
        lines.append(f"Consumption: {bill['consumption_kwh']:.1f} kWh")
        lines.append(f"Tariff: {bill['tariff_category']}")
        lines.append("-" * 60)

        if bill['block_details']:
            lines.append("Block-wise Calculation:")
            for block in bill['block_details']:
                lines.append(f"  {block['block_range']:10} kWh: "
                           f"{block['kwh_in_block']:6.1f} kWh × Rs.{block['rate_lkr_per_kwh']:6.2f} "
                           f"= Rs.{block['charge_lkr']:8.2f}")
            lines.append("-" * 60)

        lines.append(f"Energy Charge:        Rs.{bill['energy_charge_lkr']:12,.2f}")
        lines.append(f"Fixed Charge:         Rs.{bill['fixed_charge_lkr']:12,.2f}")
        lines.append(f"Fuel Adjustment:      Rs.{bill['fuel_adjustment_lkr']:12,.2f}")
        lines.append(f"Subtotal:             Rs.{bill['subtotal_lkr']:12,.2f}")
        lines.append(f"VAT:                  Rs.{bill['vat_lkr']:12,.2f}")
        lines.append("=" * 60)
        lines.append(f"TOTAL MONTHLY BILL:   Rs.{bill['total_bill_lkr']:12,.2f}")
        lines.append(f"Effective Rate:       Rs.{bill['effective_rate_lkr_per_kwh']:7.2f}/kWh")

        return "\n".join(lines)

    def format_annual_bills_for_display(self, annual_bills: Dict) -> str:
        # Format annual bills for display
        monthly_bills = annual_bills['monthly_bills']
        annual_summary = annual_bills['annual_summary']

        lines = []
        lines.append("\n ANNUAL ELECTRICITY BILLS SUMMARY")
        lines.append("=" * 70)
        lines.append(f"{'Month':8} {'Consumption':>12} {'Energy':>12} {'Fixed':>10} {'VAT':>10} {'Total':>12}")
        lines.append(f"{'':8} {'(kWh)':>12} {'Charge':>12} {'Charge':>10} {'':>10} {'Bill':>12}")
        lines.append("-" * 70)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for month in range(1, 13):
            bill = monthly_bills[month]
            lines.append(
                f"{month_names[month-1]:8} {bill['consumption_kwh']:8.1f} kWh "
                f"Rs.{bill['energy_charge_lkr']:10,.0f} "
                f"Rs.{bill['fixed_charge_lkr']:8,.0f} "
                f"Rs.{bill['vat_lkr']:8,.0f} "
                f"Rs.{bill['total_bill_lkr']:10,.0f}"
            )

        lines.append("=" * 70)
        lines.append("ANNUAL TOTALS:")
        lines.append(f"  Total Consumption: {annual_summary['total_consumption_kwh']:,.0f} kWh")
        lines.append(f"  Total Energy Charge: Rs.{annual_summary['total_energy_charge_lkr']:,.0f}")
        lines.append(f"  Total Fixed Charge: Rs.{annual_summary['total_fixed_charge_lkr']:,.0f}")
        lines.append(f"  Total VAT: Rs.{annual_summary['total_vat_lkr']:,.0f}")
        lines.append("=" * 70)
        lines.append(f"  ANNUAL ELECTRICITY BILL: Rs.{annual_summary['total_bill_lkr']:,.0f}")
        lines.append(f"  Monthly Average: Rs.{annual_summary['monthly_average_bill_lkr']:,.0f}")
        lines.append(f"  Effective Rate: Rs.{annual_summary['effective_rate_lkr_per_kwh']:.2f}/kWh")

        return "\n".join(lines)

class NetMeteringCalculator:
    """
    Net metering calculator for solar customers
    Handles credit carryover and annual settlements
    """

    def __init__(self, base_calculator: PUCsLTariffCalculator):
        self.base = base_calculator
        self.credit_balance = 0.0
        self.credit_history = []  # Track when credits were earned
        self.monthly_records = []

    def calculate_monthly_bill(self, consumption_kwh: float, generation_kwh: float,
                              tariff_code: str = 'D1', month: int = None,
                              year: int = None) -> Dict:

        net_consumption = consumption_kwh - generation_kwh

        # Get export rate (highest block rate)
        export_rate = self.base.get_highest_block_rate(tariff_code)
        fixed_charge = self.base.get_fixed_charge(tariff_code)

        if net_consumption >= 0:
            # Using grid power
            bill = self.base.calculate_monthly_bill(net_consumption, tariff_code)

            # Apply credits if available
            credits_used = 0
            if self.credit_balance > 0:
                original_bill = bill['total_bill_lkr']
                bill['total_bill_lkr'] = max(
                    fixed_charge,
                    bill['total_bill_lkr'] - self.credit_balance
                )
                credits_used = original_bill - bill['total_bill_lkr']
                self.credit_balance = max(0, self.credit_balance - credits_used)

            result = {
                **bill,
                'net_consumption_kwh': net_consumption,
                'generation_kwh': generation_kwh,
                'credits_used': credits_used,
                'credit_balance': self.credit_balance,
                'note': 'Net consumption billed'
            }

        else:
            # Exporting to grid
            excess_kwh = -net_consumption
            credit_earned = excess_kwh * export_rate
            self.credit_balance += credit_earned

            # Track credit history
            if month and year:
                self.credit_history.append({
                    'month': month,
                    'year': year,
                    'excess_kwh': excess_kwh,
                    'credit_earned': credit_earned,
                    'balance_after': self.credit_balance
                })

            # Minimum bill (fixed charge only)
            result = {
                'consumption_kwh': round(consumption_kwh, 2),
                'generation_kwh': round(generation_kwh, 2),
                'net_consumption_kwh': round(net_consumption, 2),
                'excess_kwh': round(excess_kwh, 2),
                'credit_earned': round(credit_earned, 2),
                'credit_balance': round(self.credit_balance, 2),
                'total_bill_lkr': fixed_charge,
                'tariff_category': tariff_code.upper(),
                'fixed_charge_lkr': fixed_charge,
                'effective_rate_lkr_per_kwh': fixed_charge / consumption_kwh if consumption_kwh > 0 else 0,
                'note': 'Excess generation - credits earned'
            }

        # Store monthly record
        self.monthly_records.append({
            'month': month,
            'year': year,
            'consumption': consumption_kwh,
            'generation': generation_kwh,
            'net': net_consumption,
            'credit_balance': self.credit_balance,
            'bill': result['total_bill_lkr']
        })

        return result

    def calculate_annual_bills(self, monthly_consumption: Dict[int, float],
                              monthly_generation: Dict[int, float],
                              tariff_code: str = 'D1',
                              year: int = None) -> Dict:

        # Reset for new year
        self.credit_balance = 0
        self.credit_history = []
        self.monthly_records = []

        monthly_bills = {}
        annual_summary = {
            'total_consumption_kwh': 0,
            'total_generation_kwh': 0,
            'total_net_kwh': 0,
            'total_bill_lkr': 0,
            'total_fixed_charge_lkr': 0,
            'total_credits_earned': 0,
            'total_credits_used': 0,
            'final_credit_balance': 0
        }

        for month in range(1, 13):
            consumption = monthly_consumption.get(month, 0)
            generation = monthly_generation.get(month, 0)

            bill = self.calculate_monthly_bill(
                consumption, generation, tariff_code,
                month=month, year=year
            )

            monthly_bills[month] = bill

            # Update summary
            annual_summary['total_consumption_kwh'] += consumption
            annual_summary['total_generation_kwh'] += generation
            annual_summary['total_net_kwh'] += (consumption - generation)
            annual_summary['total_bill_lkr'] += bill['total_bill_lkr']
            annual_summary['total_fixed_charge_lkr'] += bill.get('fixed_charge_lkr', 0)
            annual_summary['total_credits_earned'] += bill.get('credit_earned', 0)
            annual_summary['total_credits_used'] += bill.get('credits_used', 0)

        annual_summary['final_credit_balance'] = self.credit_balance

        # Perform annual settlement (credits expire)
        settlement = self.annual_settlement()
        annual_summary['settlement'] = settlement

        return {
            'monthly_bills': monthly_bills,
            'annual_summary': annual_summary,
            'credit_history': self.credit_history
        }

    def annual_settlement(self) -> Dict:

        # Handle annual settlement - credits expire after 12 months
        expired_credits = self.credit_balance
        self.credit_balance = 0

        logger.info(f"Annual settlement: Rs. {expired_credits:.2f} in credits expired")

        return {
            'expired_credits': expired_credits,
            'date': datetime.now().isoformat(),
            'note': 'Credits expired after 12 months per PUCSL rules'
        }

    def get_credit_summary(self) -> Dict:
        # Get summary of credit activity
        if not self.credit_history:
            return {'total_credits_earned': 0, 'current_balance': 0}

        total_earned = sum(c['credit_earned'] for c in self.credit_history)
        return {
            'total_credits_earned': total_earned,
            'current_balance': self.credit_balance,
            'months_with_excess': len(self.credit_history),
            'average_monthly_credit': total_earned / len(self.credit_history) if self.credit_history else 0
        }

    def reset(self):
        # Reset calculator for new simulation
        self.credit_balance = 0
        self.credit_history = []
        self.monthly_records = []
        logger.info("Net metering calculator reset")