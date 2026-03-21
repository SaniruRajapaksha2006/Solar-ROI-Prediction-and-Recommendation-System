import pytest
from unittest.mock import patch, mock_open, MagicMock
import json

from financeml import SolarFinancialModel
from update_live_data import MarketDataUpdater

# Use the JSON file as a string to mock file reads
MOCK_JSON_DATA = {
    "pricing_database": {"3": 655000, "5": 800000, "8": 1450000, "10": 1800000, "15": 2600000, "20": 2190000},
    "vendors": [{"Name": "Genso Power Technologies", "Location": "Maharagama", "Contact": "011 2 000 000",
                 "Specialty": "Residential & Commercial Solar"}],
    "tariffs": {"base_import_tariff": 45.0, "export_tariff_under_5kw": 20.9, "export_tariff_over_5kw": 19.61,
                "discount_rate": 0.041},
    "last_updated": "2026-03-13 19:26:47"
}


# Tests for SolarFinancialModel

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_financial_model_init_success(mock_file, mock_exists):
    """Test initialization when market_data.json exists."""
    model = SolarFinancialModel("dummy.json")

    assert model.BASE_IMPORT_TARIFF_LKR == 45.0
    assert model.DISCOUNT_RATE == 0.041
    assert model.PRICING_DATABASE[5] == 800000


@patch('os.path.exists', return_value=False)
def test_financial_model_init_fallback(mock_exists):
    """Test fallback data when JSON is missing."""
    model = SolarFinancialModel("missing.json")
    assert model.PRICING_DATABASE[3] == 750000  # Fallback data
    assert model.VENDOR_DATABASE[0]["Name"] == "System Default Vendor"


@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_financial_model_get_system_cost(mock_file, mock_exists):
    model = SolarFinancialModel("dummy.json")
    assert model.get_system_cost(5) == 800000  # From JSON
    assert model.get_system_cost(6) == 1200000.0  # Fallback calculation (6 * 200000.0)


@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_financial_model_report_invalid_inputs(mock_file, mock_exists):
    model = SolarFinancialModel("dummy.json")


    with pytest.raises(ValueError, match="System size, generation, and consumption must be valid non-negative numbers."):
        model.calculate_financial_report(0, 7200, 4800)  # Size 0 should fail


@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_financial_model_report_success(mock_file, mock_exists):
    model = SolarFinancialModel("dummy.json")
    report = model.calculate_financial_report(5, 7200, 4800)

    # Verify the structure of the returned report
    assert report["System_Size_KW"] == 5
    assert report["Total_Investment_LKR"] == 800000
    assert "Expected_NPV_LKR" in report
    assert "Chart_Data" in report



# Tests for MarketDataUpdater

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_market_updater_init_success(mock_file, mock_exists):
    updater = MarketDataUpdater("dummy.json")
    assert updater.db is not None
    assert updater.db["tariffs"]["base_import_tariff"] == 45.0


@patch('os.path.exists', return_value=False)
def test_market_updater_init_missing_file(mock_exists):
    updater = MarketDataUpdater("missing.json")
    assert updater.db is None


@patch('requests.get')
@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_market_updater_fetch_inflation(mock_file, mock_exists, mock_get):
    # Create a fake API response
    mock_response = MagicMock()
    mock_response.json.return_value = [{}, [{"value": 6.5}]]  # 6.5% inflation
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    updater = MarketDataUpdater("dummy.json")
    updater.fetch_inflation_api()

    # Discount rate should be updated to (6.5 / 100) + 0.045 = 0.11
    assert pytest.approx(updater.db["tariffs"]["discount_rate"]) == 0.11


@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps(MOCK_JSON_DATA))
def test_market_updater_scrape_ceb_tariffs(mock_file, mock_exists):
    updater = MarketDataUpdater("dummy.json")
    updater.scrape_ceb_tariffs()
    assert updater.db["tariffs"]["export_tariff_under_5kw"] == 20.90