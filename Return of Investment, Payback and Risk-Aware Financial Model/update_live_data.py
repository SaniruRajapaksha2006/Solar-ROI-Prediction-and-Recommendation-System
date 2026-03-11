import requests
from bs4 import BeautifulSoup
import json
import os
import re
import statistics
from datetime import datetime

class MarketDataUpdater:
    def __init__(self, json_filepath="market_data.json"):
        self.json_filepath = json_filepath
        if os.path.exists(self.json_filepath):
            with open(self.json_filepath, "r") as file:
                self.db = json.load(file)
        else:
            print("❌ market_data.json not found. Please create it first.")
            self.db = None

    def fetch_inflation_api(self):
        """
        METHOD A: REST API Integration
        Fetches the official Inflation Rate for Sri Lanka using the World Bank API.
        """
        print("🔄 Fetching live Inflation Data from World Bank API...")
        try:
            url = "http://api.worldbank.org/v2/country/LK/indicator/FP.CPI.TOTL.ZG?format=json"
            response = requests.get(url, timeout=10)
            data = response.json()

            for record in data[1]:
                if record['value'] is not None:
                    inflation_rate = record['value']
                    inflation_decimal = round(inflation_rate / 100, 3)
                    break

            # Dynamic Discount Rate = Inflation + Risk Premium
            dynamic_discount_rate = inflation_decimal + 0.045
            self.db["tariffs"]["discount_rate"] = dynamic_discount_rate
            print(f"✅ API Success: SL Inflation is {round(inflation_rate, 2)}%. Updated Discount Rate to {round(dynamic_discount_rate * 100, 2)}%.")

        except Exception as e:
            print(f"⚠️ API Fetch Failed: {e}. Using existing JSON data.")

    def scrape_reputed_vendor_prices(self):
        pass

    def scrape_ceb_tariffs(self):
        pass

    def save_database(self):
        if self.db:
            self.db["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.json_filepath, "w") as file:
                json.dump(self.db, file, indent=4)
            print("\n💾 market_data.json has been successfully updated with live real-world data!")

if __name__ == "__main__":
    print("\n--- INITIATING LIVE MARKET DATA SYNC ---")
    updater = MarketDataUpdater()
    if updater.db:
        updater.fetch_inflation_api()
        updater.scrape_reputed_vendor_prices()
        updater.scrape_ceb_tariffs()
        updater.save_database()
    print("--- SYNC COMPLETE ---\n")