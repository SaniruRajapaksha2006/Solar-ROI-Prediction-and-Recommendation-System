import pandas as pd
import numpy as np
import random
import math
import os
import warnings

warnings.filterwarnings('ignore')


# CONFIGURATION

RAW_DIR = '../raw'
PROCESSED_DIR = '../processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("FIXED DATA EXPANSION PIPELINE - WITH LOCATION DATA\n")

# LOCATION DATA CONFIGURATION

def get_transformer_locations():
    """Return fixed transformer locations in Maharagama area"""
    return {
        'AZ0001': {  # Maharagama Town Center
            'lat': 6.8514, 'lon': 79.9211,
            'description': 'Maharagama Town Center',
            'max_capacity': 160
        },
        'AZ0002': {  # National Institute of Education
            'lat': 6.8433, 'lon': 79.9322,
            'description': 'National Institute of Education Area',
            'max_capacity': 160
        },
        'AZ0003': {  # High Level Road Residential
            'lat': 6.8502, 'lon': 79.9238,
            'description': 'High Level Road Residential Area',
            'max_capacity': 160
        },
        'SYN1000': {  # Borelesgamuwa Residential Area
            'lat': 6.8422, 'lon': 79.9133,
            'description': 'Borelesgamuwa Residential Area',
            'max_capacity': 160
        },
        'SYN1001': {  # Wijerama Residential
            'lat': 6.8530, 'lon': 79.9100,
            'description': 'Wijerama Residential Area',
            'max_capacity': 160
        },
        'SYN1002': {  # Wattegedara Residential Area
            'lat': 6.8472, 'lon': 79.9177,
            'description': 'Wattegedara Residential Area',
            'max_capacity': 160
        },
        'SYN1003': {  # Nawinna Residential Area
            'lat': 6.8536, 'lon': 79.9125,
            'description': 'Nawinna Residential Area',
            'max_capacity': 160
        },
        'SYN1004': {  # Cancer Hospital Area
            'lat': 6.8372, 'lon': 79.9213,
            'description': 'Cancer Hospital Maharagama Area',
            'max_capacity': 160
        },
        'SYN1005': {  # Arawwala Residential Area
            'lat': 6.8386, 'lon': 79.9380,
            'description': 'Arawwala Residential Area',
            'max_capacity': 160
        },
        'SYN1006': {  # Dehiwala-Maharagama Road Area
            'lat': 6.8427, 'lon': 79.9183,
            'description': 'Dehiwala-Maharagama Road Area',
            'max_capacity': 160
        }
    }


def generate_customer_locations(transformer_code, account_list):
    """
    Generate realistic customer locations around a transformer in urban Maharagama

    Urban distribution pattern:
    - 20% within 50m (same street/commercial area)
    - 50% within 50-200m (immediate neighborhood)
    - 25% within 200-350m (nearby area)
    - 5% within 350-500m (edge cases)
    """
    transformer_locations = get_transformer_locations()
    tf_lat = transformer_locations[transformer_code]['lat']
    tf_lon = transformer_locations[transformer_code]['lon']

    # Conversion factors: meters to degrees
    meter_to_deg_lat = 1 / 111000  # 1 meter in degrees latitude
    meter_to_deg_lon = 1 / (111000 * math.cos(math.radians(tf_lat)))  # 1 meter in degrees longitude

    customer_locations = {}
    n_customers = len(account_list)

    # Create 3-4 neighborhood clusters around transformer
    n_clusters = min(4, max(2, n_customers // 15))
    clusters = []

    for _ in range(n_clusters):
        # Cluster center is 50-150m from transformer
        cluster_dist = random.uniform(50, 150)
        cluster_dir = random.uniform(0, 2 * math.pi)

        cluster_lat = tf_lat + cluster_dist * meter_to_deg_lat * math.sin(cluster_dir)
        cluster_lon = tf_lon + cluster_dist * meter_to_deg_lon * math.cos(cluster_dir)
        clusters.append((cluster_lat, cluster_lon, cluster_dist))

    # Assign accounts to distance zones
    zones = {
        'very_close': int(n_customers * 0.20),  # 0-50m
        'close': int(n_customers * 0.50),  # 50-200m
        'medium': int(n_customers * 0.25),  # 200-350m
        'far': int(n_customers * 0.05)  # 350-500m
    }

    # Adjust to ensure total matches n_customers
    total_assigned = sum(zones.values())
    if total_assigned < n_customers:
        zones['close'] += n_customers - total_assigned

    zone_start_idx = 0

    # Generate locations for each zone
    for zone_name, zone_count in zones.items():
        if zone_count == 0:
            continue

        zone_accounts = account_list[zone_start_idx:zone_start_idx + zone_count]
        zone_start_idx += zone_count

        for account in zone_accounts:
            # Determine distance based on zone
            if zone_name == 'very_close':
                distance_m = random.uniform(10, 50)
            elif zone_name == 'close':
                distance_m = random.uniform(50, 200)
            elif zone_name == 'medium':
                distance_m = random.uniform(200, 350)
            else:  # 'far'
                distance_m = random.uniform(350, 500)

            # Choose whether to base location on transformer or cluster
            if random.random() < 0.7 and zone_name != 'very_close':  # 70% in clusters
                cluster_idx = random.randint(0, len(clusters) - 1)
                cluster_lat, cluster_lon, cluster_dist = clusters[cluster_idx]

                # Distance from cluster center (10-50m for urban density)
                cluster_offset = random.uniform(10, 50)
                direction = random.uniform(0, 2 * math.pi)

                customer_lat = cluster_lat + cluster_offset * meter_to_deg_lat * math.sin(direction)
                customer_lon = cluster_lon + cluster_offset * meter_to_deg_lon * math.cos(direction)
            else:  # Direct from transformer
                direction = random.uniform(0, 2 * math.pi)
                customer_lat = tf_lat + distance_m * meter_to_deg_lat * math.sin(direction)
                customer_lon = tf_lon + distance_m * meter_to_deg_lon * math.cos(direction)

            # Ensure location is within 500m
            distance = calculate_distance_m(customer_lat, customer_lon, tf_lat, tf_lon)
            if distance > 500:
                # Scale back if too far
                scale = 500 / distance
                customer_lat = tf_lat + (customer_lat - tf_lat) * scale
                customer_lon = tf_lon + (customer_lon - tf_lon) * scale

            customer_locations[account] = (customer_lat, customer_lon)

    return customer_locations


def calculate_distance_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two coordinates using Haversine formula"""
    R = 6371000  # Earth's radius in meters

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# STEP 0: LOAD TF_CUSTOMER DATA (CORRECTED)

def load_tf_customer_data_corrected():
    """Load TF_CUSTOMER and ensure proper data types"""
    print("\nLoading TF_CUSTOMER data...")

    try:
        import openpyxl

        # Read Excel file
        file_path = '../raw/TF CUSTOMER.xlsx'
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            return pd.DataFrame()

        tf_customer = pd.read_excel(file_path, sheet_name=None)

        # Combine all sheets
        all_data = []
        for sheet_name, sheet_df in tf_customer.items():
            # Clean the sheet name (remove any whitespace)
            transformer_code = str(sheet_name).strip()
            sheet_df['TRANSFORMER_CODE'] = transformer_code
            all_data.append(sheet_df)

        tf_master = pd.concat(all_data, ignore_index=True)

        # Clean column names
        tf_master.columns = [str(col).strip() for col in tf_master.columns]

        # Ensure ACCOUNT_NO is string
        tf_master['ACCOUNT_NO'] = tf_master['ACCOUNT_NO'].astype(str).str.strip()

        # DEBUG: Show columns
        print(f"  Columns found: {list(tf_master.columns)}")

        # Find INV_CAPACITY column (case insensitive)
        inv_capacity_col = None
        for col in tf_master.columns:
            if 'inv' in col.lower() and 'capacity' in col.lower():
                inv_capacity_col = col
                break

        if inv_capacity_col:
            print(f"  Found capacity column: {inv_capacity_col}")
            # Convert to numeric
            tf_master[inv_capacity_col] = pd.to_numeric(tf_master[inv_capacity_col], errors='coerce').fillna(0)
            # Create HAS_SOLAR column
            tf_master['HAS_SOLAR'] = (tf_master[inv_capacity_col] > 0).astype(int)
            # Rename column
            tf_master = tf_master.rename(columns={inv_capacity_col: 'INV_CAPACITY'})
        else:
            print(f" No INV_CAPACITY column found, checking all numeric columns...")
            # Look for any numeric column that might indicate solar
            numeric_cols = tf_master.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  Using first numeric column: {numeric_cols[0]}")
                tf_master['INV_CAPACITY'] = tf_master[numeric_cols[0]]
                tf_master['HAS_SOLAR'] = (tf_master['INV_CAPACITY'] > 0).astype(int)
            else:
                tf_master['INV_CAPACITY'] = 0
                tf_master['HAS_SOLAR'] = 0

        # Find other columns
        tariff_col = None
        phase_col = None
        for col in tf_master.columns:
            col_lower = col.lower()
            if 'tariff' in col_lower:
                tariff_col = col
            elif 'phase' in col_lower:
                phase_col = col

        if tariff_col:
            tf_master = tf_master.rename(columns={tariff_col: 'CAL_TARIFF'})
        else:
            tf_master['CAL_TARIFF'] = 'D1'

        if phase_col:
            tf_master = tf_master.rename(columns={phase_col: 'PHASE'})
        else:
            tf_master['PHASE'] = 'SP'

        # Keep only essential columns
        essential_cols = ['ACCOUNT_NO', 'TRANSFORMER_CODE', 'HAS_SOLAR',
                          'INV_CAPACITY', 'CAL_TARIFF', 'PHASE']

        # Ensure all columns exist
        for col in essential_cols:
            if col not in tf_master.columns:
                if col == 'HAS_SOLAR':
                    tf_master['HAS_SOLAR'] = 0
                elif col == 'INV_CAPACITY':
                    tf_master['INV_CAPACITY'] = 0
                elif col == 'CAL_TARIFF':
                    tf_master['CAL_TARIFF'] = 'D1'
                elif col == 'PHASE':
                    tf_master['PHASE'] = 'SP'

        result = tf_master[essential_cols]

        # Count solar customers PER TRANSFORMER
        print("\n  Solar customers per transformer:")
        for transformer in result['TRANSFORMER_CODE'].unique():
            tf_data = result[result['TRANSFORMER_CODE'] == transformer]
            solar_count = tf_data['HAS_SOLAR'].sum()
            total_count = len(tf_data)
            print(f"    {transformer}: {solar_count} / {total_count} accounts")

        print(f"\n  Total accounts loaded: {len(result)}")
        print(f"  Total solar customers: {result['HAS_SOLAR'].sum()}")

        return result

    except Exception as e:
        print(f"  Error loading TF_CUSTOMER: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# STEP 1: LOAD MONTHLY DATA (SIMPLE AND CORRECT)

def load_monthly_data_simple(transformer_code, month_files):
    """Simple loading of monthly data - returns ONLY accounts with meter readings"""
    print(f"\nLoading monthly data for {transformer_code}...")

    monthly_data = {}
    all_accounts = set()

    for month_name, filepath in month_files.items():
        try:
            df = pd.read_csv(filepath)

            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]

            # Standardize ACCOUNT_NO
            if 'ACCOUNT_NO' in df.columns:
                df['ACCOUNT_NO'] = df['ACCOUNT_NO'].astype(str).str.strip()
                all_accounts.update(df['ACCOUNT_NO'].unique())

            monthly_data[month_name] = df
            print(f"  {month_name}: {len(df)} accounts")

        except Exception as e:
            print(f"  Error loading {month_name}: {e}")
            monthly_data[month_name] = pd.DataFrame()

    print(f"  Unique accounts across all months: {len(all_accounts)}")
    return monthly_data, sorted(all_accounts)


# STEP 2: PROCESS EACH ACCOUNT (CORRECTED)

def process_account_simple(account, monthly_data, tf_customer_df, transformer_code,
                           customer_locations, transformer_locations):
    """Process a single account - with location data"""

    # Get TF_CUSTOMER info for this account
    has_solar = 0
    inv_capacity = 0
    tariff = 'D1'
    phase = 'SP'

    if not tf_customer_df.empty:
        account_tf_data = tf_customer_df[
            (tf_customer_df['TRANSFORMER_CODE'] == transformer_code) &
            (tf_customer_df['ACCOUNT_NO'] == account)
            ]

        if len(account_tf_data) > 0:
            has_solar = int(account_tf_data['HAS_SOLAR'].iloc[0])
            inv_capacity = float(account_tf_data['INV_CAPACITY'].iloc[0])
            tariff = str(account_tf_data['CAL_TARIFF'].iloc[0])
            phase = str(account_tf_data['PHASE'].iloc[0])

    # Get transformer location
    tf_lat = transformer_locations[transformer_code]['lat']
    tf_lon = transformer_locations[transformer_code]['lon']

    # Get customer location
    cust_lat, cust_lon = customer_locations[account]

    # Month mapping
    month_mapping = {'Sep': 9, 'Oct': 10, 'Nov': 11}
    account_records = []

    # For each month, get or generate data
    for month_name in ['Sep', 'Oct', 'Nov']:
        month_num = month_mapping[month_name]
        df = monthly_data.get(month_name, pd.DataFrame())

        # Check if account exists in this month's data
        account_data = None
        if len(df) > 0 and 'ACCOUNT_NO' in df.columns:
            account_rows = df[df['ACCOUNT_NO'] == account]
            if len(account_rows) > 0:
                row = account_rows.iloc[0]

                # Find import and export columns
                import_val = 0
                export_val = 0

                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'import' in col_lower and 'total' in col_lower:
                        try:
                            import_val = float(row[col])
                            # Cap unrealistic values
                            if import_val > 5000:
                                import_val = random.uniform(300, 1500)
                        except:
                            import_val = random.uniform(300, 800)

                    elif 'export' in col_lower and 'total' in col_lower:
                        try:
                            export_val = float(row[col])
                            # Cap unrealistic values
                            if export_val > 2000:
                                export_val = random.uniform(100, 800)
                            # If TF_CUSTOMER says no solar, export should be 0
                            if has_solar == 0:
                                export_val = 0
                        except:
                            export_val = 0

                net_consumption = import_val - export_val
                account_data = {
                    'import': import_val,
                    'export': export_val,
                    'net': net_consumption,
                    'quality': 'REAL'
                }

        # If no data for this month, generate it
        if account_data is None:
            # Generate realistic values
            base_consumption = random.uniform(200, 800)
            seasonal = get_seasonal_factor(month_num)
            net_consumption = base_consumption * seasonal * random.uniform(0.9, 1.1)

            if has_solar == 1:
                export_kwh = net_consumption * random.uniform(0.3, 0.7)
                import_kwh = net_consumption + export_kwh
            else:
                export_kwh = 0
                import_kwh = net_consumption

            account_data = {
                'import': import_kwh,
                'export': export_kwh,
                'net': net_consumption,
                'quality': 'FILLED'
            }

        # Add to records
        account_records.append({
            'TRANSFORMER_CODE': transformer_code,
            'TRANSFORMER_LAT': tf_lat,
            'TRANSFORMER_LON': tf_lon,
            'ACCOUNT_NO': account,
            'CUSTOMER_LAT': cust_lat,
            'CUSTOMER_LON': cust_lon,
            'MONTH': month_num,
            'YEAR': 2025,
            'IMPORT_kWh': round(account_data['import'], 2),
            'EXPORT_kWh': round(account_data['export'], 2),
            'NET_CONSUMPTION_kWh': round(account_data['net'], 2),
            'HAS_SOLAR': has_solar,
            'INV_CAPACITY': inv_capacity,
            'CAL_TARIFF': tariff,
            'PHASE': phase,
            'DATA_QUALITY': account_data['quality']
        })

    return account_records


# STEP 3: MAIN PROCESSING (CORRECTED) - WITH LOCATIONS

def process_transformer_corrected(transformer_code, month_files, tf_customer_df):
    """Main processing function - CORRECTED with location data"""
    print(f"PROCESSING: {transformer_code}\n")

    # Get transformer locations
    transformer_locations = get_transformer_locations()

    # Step 1: Load monthly data
    monthly_data, all_accounts = load_monthly_data_simple(transformer_code, month_files)

    # Step 1.5: Generate customer locations
    print(f"\nGenerating customer locations for {len(all_accounts)} accounts...")
    customer_locations = generate_customer_locations(transformer_code, all_accounts)

    # Calculate distances
    distances = []
    tf_lat = transformer_locations[transformer_code]['lat']
    tf_lon = transformer_locations[transformer_code]['lon']

    for account, (cust_lat, cust_lon) in customer_locations.items():
        distance = calculate_distance_m(cust_lat, cust_lon, tf_lat, tf_lon)
        distances.append(distance)

    avg_distance = np.mean(distances) if distances else 0
    max_distance = max(distances) if distances else 0

    print(f"  Average distance from transformer: {avg_distance:.1f}m")
    print(f"  Maximum distance: {max_distance:.1f}m")

    # Step 2: Process each account
    all_records = []

    print(f"\nProcessing {len(all_accounts)} accounts...")

    for account in all_accounts:
        account_records = process_account_simple(
            account, monthly_data, tf_customer_df, transformer_code,
            customer_locations, transformer_locations
        )
        all_records.extend(account_records)

    # Create DataFrame
    reconciled_df = pd.DataFrame(all_records)

    # Add distance column
    reconciled_df['DISTANCE_FROM_TF_M'] = reconciled_df.apply(
        lambda row: calculate_distance_m(
            row['CUSTOMER_LAT'], row['CUSTOMER_LON'],
            row['TRANSFORMER_LAT'], row['TRANSFORMER_LON']
        ), axis=1
    )

    # Count properly
    unique_accounts = reconciled_df['ACCOUNT_NO'].nunique()
    solar_accounts = reconciled_df.groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()
    real_records = (reconciled_df['DATA_QUALITY'] == 'REAL').sum()
    filled_records = (reconciled_df['DATA_QUALITY'] == 'FILLED').sum()

    print(f"\n  Summary after reconciliation:")
    print(f"    - Accounts: {unique_accounts}")
    print(f"    - Solar accounts: {solar_accounts}")
    print(f"    - Total records: {len(reconciled_df)}")
    print(f"    - Real records: {real_records}")
    print(f"    - Filled records: {filled_records}")

    # Save reconciled data
    reconciled_path = os.path.join(PROCESSED_DIR, f'{transformer_code}_RECONCILED_3months.csv')
    reconciled_df.to_csv(reconciled_path, index=False)
    print(f"  ✓ Saved: {reconciled_path}")

    # Step 3: Expand to 12 months
    print(f"\nExpanding to 12 months...")
    expanded_df = expand_to_12_months_corrected(reconciled_df, transformer_code)

    # Save expanded data
    expanded_path = os.path.join(PROCESSED_DIR, f'{transformer_code}_EXPANDED_12months.csv')
    expanded_df.to_csv(expanded_path, index=False)
    print(f"  ✓ Saved: {expanded_path}")

    # Final summary
    solar_accounts_expanded = expanded_df.groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()

    print(f"\n  Final summary for {transformer_code}:")
    print(f"    - Accounts: {expanded_df['ACCOUNT_NO'].nunique()}")
    print(f"    - Solar ACCOUNTS (not records!): {solar_accounts_expanded}")
    print(f"    - Total records: {len(expanded_df)}")

    # Distance statistics for expanded data
    expanded_avg_distance = expanded_df['DISTANCE_FROM_TF_M'].mean()
    expanded_max_distance = expanded_df['DISTANCE_FROM_TF_M'].max()
    print(f"    - Avg distance from transformer: {expanded_avg_distance:.1f}m")
    print(f"    - Max distance from transformer: {expanded_max_distance:.1f}m")

    # Count by quality
    quality_counts = expanded_df['DATA_QUALITY'].value_counts()
    for quality, count in quality_counts.items():
        percentage = count / len(expanded_df) * 100
        print(f"    - {quality} records: {count} ({percentage:.1f}%)")

    return expanded_df


def expand_to_12_months_corrected(reconciled_df, transformer_code):
    """Expand 3 months to 12 months - CORRECTED"""
    expanded_records = []

    # Get transformer location from first row
    tf_lat = reconciled_df['TRANSFORMER_LAT'].iloc[0]
    tf_lon = reconciled_df['TRANSFORMER_LON'].iloc[0]

    # Get unique accounts
    accounts = reconciled_df['ACCOUNT_NO'].unique()

    for account in accounts:
        # Get account data
        account_data = reconciled_df[reconciled_df['ACCOUNT_NO'] == account]

        # Account properties (same for all months)
        has_solar = account_data['HAS_SOLAR'].iloc[0]
        inv_capacity = account_data['INV_CAPACITY'].iloc[0]
        tariff = account_data['CAL_TARIFF'].iloc[0]
        phase = account_data['PHASE'].iloc[0]
        cust_lat = account_data['CUSTOMER_LAT'].iloc[0]
        cust_lon = account_data['CUSTOMER_LON'].iloc[0]

        # Get values for Sep, Oct, Nov
        real_data = {}
        for month in [9, 10, 11]:
            month_df = account_data[account_data['MONTH'] == month]
            if len(month_df) > 0:
                real_data[month] = {
                    'import': month_df['IMPORT_kWh'].iloc[0],
                    'export': month_df['EXPORT_kWh'].iloc[0],
                    'net': month_df['NET_CONSUMPTION_kWh'].iloc[0],
                    'quality': month_df['DATA_QUALITY'].iloc[0]
                }

        # Generate 12 months
        for month in range(1, 13):
            if month in real_data:
                # Use existing data
                data = real_data[month]
                expanded_records.append({
                    'TRANSFORMER_CODE': transformer_code,
                    'TRANSFORMER_LAT': tf_lat,
                    'TRANSFORMER_LON': tf_lon,
                    'ACCOUNT_NO': account,
                    'CUSTOMER_LAT': cust_lat,
                    'CUSTOMER_LON': cust_lon,
                    'MONTH': month,
                    'YEAR': 2025,
                    'IMPORT_kWh': data['import'],
                    'EXPORT_kWh': data['export'],
                    'NET_CONSUMPTION_kWh': data['net'],
                    'HAS_SOLAR': has_solar,
                    'INV_CAPACITY': inv_capacity,
                    'CAL_TARIFF': tariff,
                    'PHASE': phase,
                    'DATA_QUALITY': data['quality']
                })
            else:
                # Generate synthetic data
                # Calculate baseline from real months
                if real_data:
                    avg_net = np.mean([d['net'] for d in real_data.values()])
                    if has_solar == 1:
                        avg_export = np.mean([d['export'] for d in real_data.values()])
                    else:
                        avg_export = 0
                else:
                    avg_net = random.uniform(200, 800)
                    avg_export = avg_net * random.uniform(0.3, 0.7) if has_solar == 1 else 0

                # Apply seasonal factor
                seasonal = get_seasonal_factor(month)
                net_consumption = avg_net * seasonal * random.uniform(0.9, 1.1)

                # Ensure realistic
                net_consumption = max(50, min(net_consumption, 2000))

                if has_solar == 1:
                    solar_factor = get_solar_factor(month)
                    export_kwh = avg_export * solar_factor * random.uniform(0.8, 1.2)
                    export_kwh = min(export_kwh, 1000)
                    import_kwh = net_consumption + export_kwh
                else:
                    export_kwh = 0
                    import_kwh = net_consumption

                import_kwh = min(import_kwh, 2500)

                expanded_records.append({
                    'TRANSFORMER_CODE': transformer_code,
                    'TRANSFORMER_LAT': tf_lat,
                    'TRANSFORMER_LON': tf_lon,
                    'ACCOUNT_NO': account,
                    'CUSTOMER_LAT': cust_lat,
                    'CUSTOMER_LON': cust_lon,
                    'MONTH': month,
                    'YEAR': 2025,
                    'IMPORT_kWh': round(import_kwh, 2),
                    'EXPORT_kWh': round(export_kwh, 2),
                    'NET_CONSUMPTION_kWh': round(net_consumption, 2),
                    'HAS_SOLAR': has_solar,
                    'INV_CAPACITY': inv_capacity,
                    'CAL_TARIFF': tariff,
                    'PHASE': phase,
                    'DATA_QUALITY': 'SYNTHETIC'
                })

    expanded_df = pd.DataFrame(expanded_records)

    # Add distance column
    expanded_df['DISTANCE_FROM_TF_M'] = expanded_df.apply(
        lambda row: calculate_distance_m(
            row['CUSTOMER_LAT'], row['CUSTOMER_LON'],
            row['TRANSFORMER_LAT'], row['TRANSFORMER_LON']
        ), axis=1
    )

    # Verify
    expected_records = len(accounts) * 12
    actual_records = len(expanded_df)

    if expected_records == actual_records:
        print(f"  All {len(accounts)} accounts have 12 months ({actual_records} records)")
    else:
        print(f"  Record count mismatch: Expected {expected_records}, got {actual_records}")

    return expanded_df


# HELPER FUNCTIONS

def get_seasonal_factor(month):
    factors = {
        1: 0.90, 2: 0.95, 3: 1.15, 4: 1.25, 5: 1.20,
        6: 1.00, 7: 0.95, 8: 0.90, 9: 1.00, 10: 1.10,
        11: 1.05, 12: 0.95
    }
    return factors.get(month, 1.0)


def get_solar_factor(month):
    factors = {
        1: 1.10, 2: 1.15, 3: 1.20, 4: 1.15, 5: 1.00,
        6: 0.60, 7: 0.65, 8: 0.70, 9: 0.95, 10: 1.05,
        11: 1.00, 12: 1.10
    }
    return factors.get(month, 1.0)


# GENERATE SYNTHETIC TRANSFORMERS (65+ ACCOUNTS EACH) - WITH LOCATIONS

def generate_synthetic_transformers(real_expanded_data, n_synthetic=7):
    """Generate synthetic transformers with 65+ accounts each - with location data"""
    print(f"GENERATING {n_synthetic} SYNTHETIC TRANSFORMERS\n")

    # Analyze real patterns
    real_patterns = {}
    for tf_code, df in real_expanded_data.items():
        real_patterns[tf_code] = {
            'solar_percentage': df['HAS_SOLAR'].mean() * 100,
            'avg_consumption': df['NET_CONSUMPTION_kWh'].mean(),
            'std_consumption': df['NET_CONSUMPTION_kWh'].std(),
            'avg_export': df[df['HAS_SOLAR'] == 1]['EXPORT_kWh'].mean() if df['HAS_SOLAR'].sum() > 0 else 0,
        }

    # Get transformer locations
    transformer_locations = get_transformer_locations()
    synthetic_transformers = {}

    for i in range(n_synthetic):
        tf_code = f'SYN{1000 + i:04d}'
        print(f"\n  Creating {tf_code}...")

        # Get transformer location
        tf_location = transformer_locations[tf_code]
        tf_lat = tf_location['lat']
        tf_lon = tf_location['lon']

        # Choose pattern
        pattern_tf = random.choice(list(real_patterns.keys()))
        pattern = real_patterns[pattern_tf]

        # Create 65-85 accounts
        n_accounts = random.randint(65, 85)

        # Generate customer locations first
        account_numbers = []
        account_prefix = f'7{random.randint(10000000, 99999999):08d}'[:3]

        for acc_idx in range(1, n_accounts + 1):
            account_no = f'{account_prefix}{acc_idx:06d}'
            account_numbers.append(account_no)

        # Generate locations for all accounts
        customer_locations = generate_customer_locations(tf_code, account_numbers)

        # Calculate distance statistics
        distances = []
        for account, (cust_lat, cust_lon) in customer_locations.items():
            distance = calculate_distance_m(cust_lat, cust_lon, tf_lat, tf_lon)
            distances.append(distance)

        avg_distance = np.mean(distances) if distances else 0
        print(f"    Generated {n_accounts} accounts")
        print(f"    Average distance from transformer: {avg_distance:.1f}m")

        synthetic_data = []

        for account_no in account_numbers:
            # Get customer location
            cust_lat, cust_lon = customer_locations[account_no]

            # Solar status (ensure at least 10% solar)
            solar_prob = max(0.1, pattern['solar_percentage'] / 100)
            has_solar = 1 if random.random() < solar_prob else 0

            tariff = random.choice(['D1', 'GP11', 'GP12'])
            phase = 'SP' if random.random() < 0.8 else 'TP'
            inv_capacity = random.choice([3, 5, 7, 10]) if has_solar else 0

            # Realistic consumption
            base_consumption = random.normalvariate(
                min(800, pattern['avg_consumption']),
                min(300, pattern['std_consumption'] * 0.5)
            )
            base_consumption = max(100, min(base_consumption, 1500))

            # Base export if solar
            if has_solar:
                base_export = pattern['avg_export'] * random.uniform(0.7, 1.3)
                base_export = max(50, min(base_export, 800))
            else:
                base_export = 0

            # Generate 12 months
            for month in range(1, 13):
                seasonal = get_seasonal_factor(month)
                net_consumption = base_consumption * seasonal * random.uniform(0.9, 1.1)
                net_consumption = max(50, min(net_consumption, 2000))

                if has_solar:
                    solar_factor = get_solar_factor(month)
                    export_kwh = base_export * solar_factor * random.uniform(0.8, 1.2)
                    export_kwh = min(export_kwh, 1000)
                    import_kwh = net_consumption + export_kwh
                else:
                    export_kwh = 0
                    import_kwh = net_consumption

                import_kwh = min(import_kwh, 2500)

                synthetic_data.append({
                    'TRANSFORMER_CODE': tf_code,
                    'TRANSFORMER_LAT': tf_lat,
                    'TRANSFORMER_LON': tf_lon,
                    'ACCOUNT_NO': account_no,
                    'CUSTOMER_LAT': cust_lat,
                    'CUSTOMER_LON': cust_lon,
                    'MONTH': month,
                    'YEAR': 2025,
                    'IMPORT_kWh': round(import_kwh, 2),
                    'EXPORT_kWh': round(export_kwh, 2),
                    'NET_CONSUMPTION_kWh': round(net_consumption, 2),
                    'HAS_SOLAR': has_solar,
                    'INV_CAPACITY': inv_capacity,
                    'CAL_TARIFF': tariff,
                    'PHASE': phase,
                    'DATA_QUALITY': 'SYNTHETIC',
                    'SOURCE': 'SYNTHETIC'
                })

        synthetic_df = pd.DataFrame(synthetic_data)

        # Add distance column
        synthetic_df['DISTANCE_FROM_TF_M'] = synthetic_df.apply(
            lambda row: calculate_distance_m(
                row['CUSTOMER_LAT'], row['CUSTOMER_LON'],
                row['TRANSFORMER_LAT'], row['TRANSFORMER_LON']
            ), axis=1
        )

        synthetic_transformers[tf_code] = synthetic_df

        # Save
        synth_path = os.path.join(PROCESSED_DIR, f'{tf_code}_SYNTHETIC_12months.csv')
        synthetic_df.to_csv(synth_path, index=False)
        print(f"  Saved: {synth_path}")
        print(f"  Solar accounts: {synthetic_df['HAS_SOLAR'].sum()}")

    return synthetic_transformers


# MAIN EXECUTION

def main():
    print("CORRECTED DATA PIPELINE - WITH LOCATION DATA\n")

    # Load TF_CUSTOMER data
    tf_customer_df = load_tf_customer_data_corrected()

    # File mappings for real transformers
    transformer_files = {
        'AZ0001': {
            'Sep': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (8).csv',
            'Oct': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (7).csv',
            'Nov': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (6).csv'
        },
        'AZ0002': {
            'Sep': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (10).csv',
            'Oct': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (1).csv',
            'Nov': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (2).csv'
        },
        'AZ0003': {
            'Sep': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (9).csv',
            'Oct': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (3).csv',
            'Nov': '../raw/HISTORICAL_READINGS_INVENTORY_02_12_2025 (4).csv'
        }
    }

    # Process each real transformer
    real_expanded_data = {}

    print("PROCESSING REAL TRANSFORMERS\n")


    for transformer_code, files in transformer_files.items():
        expanded_df = process_transformer_corrected(transformer_code, files, tf_customer_df)
        real_expanded_data[transformer_code] = expanded_df

    # Generate synthetic transformers (7 more to make 10 total)
    print("GENERATING SYNTHETIC TRANSFORMERS\n")

    synthetic_transformers = generate_synthetic_transformers(real_expanded_data, n_synthetic=7)

    print("CREATING COMBINED MASTER DATASET\n")

    # Create master dataset with ALL transformers (3 real + 7 synthetic)
    master_list = []

    # Add real transformers
    for tf_code, df in real_expanded_data.items():
        df['SOURCE'] = 'REAL'
        master_list.append(df)

    # Add synthetic transformers
    for tf_code, df in synthetic_transformers.items():
        master_list.append(df)

    # Combine everything
    master_df = pd.concat(master_list, ignore_index=True)

    # Save master dataset
    master_path = os.path.join(PROCESSED_DIR, 'MASTER_DATASET_ALL_10TRANSFORMERS.csv')
    master_df.to_csv(master_path, index=False)
    print(f"\n✓ Saved master dataset: {master_path}")

    # Save separate real and synthetic datasets
    real_df = master_df[master_df['SOURCE'] == 'REAL']
    synth_df = master_df[master_df['SOURCE'] == 'SYNTHETIC']

    real_path = os.path.join(PROCESSED_DIR, 'MASTER_DATASET_REAL_ONLY.csv')
    synth_path = os.path.join(PROCESSED_DIR, 'MASTER_DATASET_SYNTHETIC_ONLY.csv')

    real_df.to_csv(real_path, index=False)
    synth_df.to_csv(synth_path, index=False)

    print(f"✓ Saved real-only dataset: {real_path}")
    print(f"✓ Saved synthetic-only dataset: {synth_path}")

    # Create transformer summary file with capacity information
    print("CREATING TRANSFORMER SUMMARY\n")

    transformer_locations = get_transformer_locations()
    transformer_summary = []

    for tf_code in transformer_locations:
        tf_data = transformer_locations[tf_code]

        # Count accounts for this transformer
        accounts_in_tf = master_df[master_df['TRANSFORMER_CODE'] == tf_code]['ACCOUNT_NO'].nunique()

        # Count solar accounts
        solar_accounts = 0
        if tf_code in real_expanded_data:
            solar_accounts = real_expanded_data[tf_code].groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()
        elif tf_code in synthetic_transformers:
            solar_accounts = synthetic_transformers[tf_code].groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()

        # Calculate headroom
        headroom = tf_data['max_capacity'] - accounts_in_tf

        transformer_summary.append({
            'TRANSFORMER_CODE': tf_code,
            'LATITUDE': tf_data['lat'],
            'LONGITUDE': tf_data['lon'],
            'DESCRIPTION': tf_data['description'],
            'MAX_CAPACITY': tf_data['max_capacity'],
            'CURRENT_CONNECTIONS': accounts_in_tf,
            'HEADROOM': headroom,
            'SOLAR_ACCOUNTS': solar_accounts,
            'SOLAR_PERCENTAGE': round(solar_accounts / accounts_in_tf * 100, 1) if accounts_in_tf > 0 else 0
        })

    # Save transformer summary
    transformer_summary_df = pd.DataFrame(transformer_summary)
    transformer_summary_path = os.path.join(PROCESSED_DIR, 'TRANSFORMERS_SUMMARY.csv')
    transformer_summary_df.to_csv(transformer_summary_path, index=False)
    print(f"✓ Saved transformer summary: {transformer_summary_path}")

    # Final comprehensive summary
    print("FINAL COMPREHENSIVE SUMMARY\n")

    # By source type
    real_tf_count = len(real_expanded_data)
    synth_tf_count = len(synthetic_transformers)

    print(f"\nTransformers by type:")
    print(f"  - Real transformers: {real_tf_count}")
    print(f"  - Synthetic transformers: {synth_tf_count}")
    print(f"  - TOTAL transformers: {real_tf_count + synth_tf_count}")

    # Overall counts
    total_accounts = master_df['ACCOUNT_NO'].nunique()
    total_records = len(master_df)

    # By source
    real_accounts = real_df['ACCOUNT_NO'].nunique()
    synth_accounts = synth_df['ACCOUNT_NO'].nunique()

    real_records = len(real_df)
    synth_records = len(synth_df)

    print(f"\nOverall counts:")
    print(f"  - Total accounts: {total_accounts}")
    print(f"    → Real: {real_accounts}")
    print(f"    → Synthetic: {synth_accounts}")
    print(f"  - Total records: {total_records}")
    print(f"    → Real: {real_records}")
    print(f"    → Synthetic: {synth_records}")

    # Solar penetration by source
    real_solar_accounts = real_df.groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()
    synth_solar_accounts = synth_df.groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()

    print(f"\nSolar penetration:")
    print(f"  - Real data: {real_solar_accounts}/{real_accounts} ({real_solar_accounts / real_accounts * 100:.1f}%)")
    print(f"  - Synthetic data: {synth_solar_accounts}/{synth_accounts} ({synth_solar_accounts / synth_accounts * 100:.1f}%)")

    # Data quality breakdown
    print(f"\nData quality:")
    for quality in ['REAL', 'FILLED', 'SYNTHETIC']:
        count = (master_df['DATA_QUALITY'] == quality).sum()
        percentage = count / total_records * 100
        print(f"  - {quality}: {count} records ({percentage:.1f}%)")

    # Geographic statistics
    print(f"\nGeographic statistics:")
    avg_distance = master_df['DISTANCE_FROM_TF_M'].mean()
    max_distance = master_df['DISTANCE_FROM_TF_M'].max()
    print(f"  - Average distance from transformer: {avg_distance:.1f}m")
    print(f"  - Maximum distance from transformer: {max_distance:.1f}m")

    # Distance distribution
    distance_zones = pd.cut(master_df['DISTANCE_FROM_TF_M'],
                            bins=[0, 50, 100, 200, 300, 400, 500, float('inf')],
                            labels=['0-50m', '50-100m', '100-200m', '200-300m', '300-400m', '400-500m', '500m+'])
    zone_counts = distance_zones.value_counts().sort_index()
    print(f"\nDistance distribution from transformers:")
    for zone, count in zone_counts.items():
        percentage = count / len(master_df) * 100
        print(f"  - {zone}: {count} records ({percentage:.1f}%)")

    # Per transformer summary
    print(f"\nPer transformer summary (with headroom):")
    for tf_code in sorted(master_df['TRANSFORMER_CODE'].unique()):
        tf_data = master_df[master_df['TRANSFORMER_CODE'] == tf_code]
        accounts = tf_data['ACCOUNT_NO'].nunique()
        solar_accounts = tf_data.groupby('ACCOUNT_NO')['HAS_SOLAR'].max().sum()
        source = tf_data['SOURCE'].iloc[0]
        headroom = transformer_summary_df[transformer_summary_df['TRANSFORMER_CODE'] == tf_code]['HEADROOM'].iloc[0]

        print(f"  - {tf_code} ({source}):")
        print(f"      Accounts: {accounts}, Solar: {solar_accounts} ({solar_accounts / accounts * 100:.1f}%)")
        print(f"      Headroom: {headroom} connections available")

    # Print final totals
    print(f"\n{'=' * 60}")
    print("GRAND TOTALS")
    print("=" * 60)
    print(f"Total transformers: {real_tf_count + synth_tf_count}")
    print(f"Total accounts: {total_accounts}")
    print(f"Total records (12 months × accounts): {total_records}")
    print(f"Total solar accounts: {real_solar_accounts + synth_solar_accounts}")
    print(f"Total headroom across all transformers: {transformer_summary_df['HEADROOM'].sum()} connections")

    return master_df, real_expanded_data, synthetic_transformers, transformer_summary_df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    master_df, real_data, synthetic_data, transformer_summary = main()

    # Print final confirmation
    print("PROCESSING COMPLETE!\n")
    print(f"✓ Processed {len(real_data)} real transformers")
    print(f"✓ Generated {len(synthetic_data)} synthetic transformers")
    print(f"✓ Created master dataset with {master_df['ACCOUNT_NO'].nunique()} accounts")
    print(f"✓ Added geographic coordinates for all transformers and customers")
    print(f"✓ Calculated capacity headroom for all transformers")
    print(f"✓ All files saved to: {PROCESSED_DIR}")