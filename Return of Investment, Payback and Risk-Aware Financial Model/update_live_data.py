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
            dynamic_discount_rate = inflation_decimal + 0.045
            self.db["tariffs"]["discount_rate"] = dynamic_discount_rate
            print(f"✅ API Success: SL Inflation is {round(inflation_rate, 2)}%. Updated Discount Rate to {round(dynamic_discount_rate * 100, 2)}%.")
        except Exception as e:
            print(f"⚠️ API Fetch Failed: {e}. Using existing JSON data.")

    def scrape_reputed_vendor_prices(self):
        """
        METHOD B.1: Advanced Web Scraping (Multiple Reputed Vendors)
        """
        print("\n🔄 Scraping reputed solar vendors for live 5kW system prices...")
        prices_found = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        # 1. Dinapala Group
        try:
            url = "https://dinapalagroup.lk/product/5kw-on-grid-solar-system-with-complete-installation/"
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            price_elem = soup.find('p', class_='price')
            if price_elem:
                price_text = price_elem.get_text()
                if "." in price_text:
                    price_text = price_text.split(".")[0]
                price_val = int(''.join(filter(str.isdigit, price_text)))
                if 500000 < price_val < 3000000:
                    prices_found.append(price_val)
                    print(f"   ✔️ Dinapala Group: {price_val:,} LKR")
        except Exception as e:
            print(f"   ⚠️ Dinapala Scraping Failed: {e}")

        # 2. Solitra Power
        try:
            url = "https://www.solitrapower.com/pricing/"
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            text_blocks = soup.get_text(separator=' ')
            match = re.search(r'5kW.*?starting from.*?Rs\.?\s*([\d,]+)', text_blocks, re.IGNORECASE | re.DOTALL)
            if match:
                price_val = int(''.join(filter(str.isdigit, match.group(1))))
                if 500000 < price_val < 3000000:
                    prices_found.append(price_val)
                    print(f"   ✔️ Solitra Power: {price_val:,} LKR")
        except Exception as e:
            print(f"   ⚠️ Solitra Scraping Failed: {e}")

        # 3. Golden Rays Solar
        try:
            url = "https://goldenrayssolar.lk/faq/"
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            text_blocks = soup.get_text(separator=' ')
            match = re.search(r'5kW system.*?LKR\s*([\d,]+)\s*[-–]\s*([\d,]+)', text_blocks, re.IGNORECASE)
            if match:
                price_min = int(''.join(filter(str.isdigit, match.group(1))))
                price_max = int(''.join(filter(str.isdigit, match.group(2))))
                avg_golden = int((price_min + price_max) / 2)
                if 500000 < avg_golden < 3000000:
                    prices_found.append(avg_golden)
                    print(f"   ✔️ Golden Rays Solar: {avg_golden:,} LKR (Average of range)")
        except Exception as e:
            print(f"   ⚠️ Golden Rays Scraping Failed: {e}")

        # Median Calculation
        if prices_found:
            most_likely_price = int(statistics.median(prices_found))
            self.db["pricing_database"]["5"] = most_likely_price
            print(f"✅ Scraping Success: Most likely 5kW System price is {most_likely_price:,} LKR (Aggregated from {len(prices_found)} vendors).")
        else:
            print("⚠️ Could not retrieve live prices. Using existing JSON data.")

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