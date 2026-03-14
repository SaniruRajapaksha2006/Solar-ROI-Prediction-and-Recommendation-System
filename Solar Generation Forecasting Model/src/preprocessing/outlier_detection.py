class OutlierDetector:

    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.outlier_bounds = {}
        self.removed_accounts = []

    def filter_residential_capacity(self, df, max_kw=20):
        """
        Remove commercial / industrial accounts before model training.

        Residential solar in Sri Lanka is almost always <= 20 kW.
        Large inverters (50 kW, 100 kW) belong to factories or commercial
        buildings — they distort the model and must be excluded.

        Args:
            df      : DataFrame containing 'INV_CAPACITY' column
            max_kw  : Maximum allowed inverter capacity (default: 20 kW)

        Returns:
            Filtered DataFrame with only residential-scale systems
        """
        print(f"\nFiltering commercial accounts (INV_CAPACITY > {max_kw} kW)...")
        print("-" * 60)

        initial_len = len(df)
        df_clean = df[df['INV_CAPACITY'] <= max_kw].copy()
        removed = initial_len - len(df_clean)

        if removed == 0:
            print(f"  No commercial accounts found (all <= {max_kw} kW)")
        else:
            removed_accounts = df[df['INV_CAPACITY'] > max_kw]['ACCOUNT_NO'].unique()
            print(f"  Removed {removed:,} records from {len(removed_accounts)} commercial account(s)")
            for acc in removed_accounts:
                cap = df[df['ACCOUNT_NO'] == acc]['INV_CAPACITY'].iloc[0]
                print(f"    Account {acc}: {cap} kW")

        print(f"\n  Remaining: {len(df_clean):,} residential records")
        print("-" * 60)
        return df_clean


    def remove_high_export_accounts(self, df, max_export=700):
        print(f"\nRemoving accounts with avg export > {max_export} kWh...")
        print("-" * 60)

        # Calculate average export per account
        account_avg = df.groupby('ACCOUNT_NO')['EXPORT_kWh'].mean()

        # Find accounts to remove
        high_export_accounts = account_avg[account_avg > max_export].index.tolist()

        if not high_export_accounts:
            print(f" No accounts with avg export > {max_export} kWh")
            return df

        self.removed_accounts = high_export_accounts

        # Show which accounts
        print(f"  Found {len(high_export_accounts)} high-export accounts:")
        for acc in high_export_accounts:
            avg = account_avg[acc]
            print(f"    Account {acc}: {avg:.0f} kWh avg")

        # Remove
        initial = len(df)
        df_clean = df[~df['ACCOUNT_NO'].isin(high_export_accounts)].copy()
        removed = initial - len(df_clean)

        print(f"\n  Removed {removed:,} records from {len(high_export_accounts)} accounts")
        print("-" * 60)
        return df_clean

    def detect_monthly_outliers(self, df, column='EXPORT_kWh', threshold=1.5):
        print(f"\nDetecting outliers in {column} by MONTH...")
        print("-" * 60)

        df_result = df.copy()
        df_result['is_outlier'] = False

        total_outliers = 0

        # Detect outliers per month
        for month, group_df in df.groupby('Month'):
            n = len(group_df)

            # Calculate IQR bounds
            Q1 = group_df[column].quantile(0.25)
            Q3 = group_df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            # Mark outliers
            outlier_mask = (group_df[column] < lower) | (group_df[column] > upper)
            df_result.loc[group_df.index[outlier_mask], 'is_outlier'] = True

            outlier_count = outlier_mask.sum()
            total_outliers += outlier_count

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            if outlier_count > 0:
                print(f"  {month_names[month - 1]}: {outlier_count}/{n} outliers "
                      f"[{lower:.0f}, {upper:.0f}] kWh")
            else:
                print(f"  {month_names[month - 1]}: 0/{n} outliers")

        print(f"\n  Total: {total_outliers} outliers ({total_outliers / len(df) * 100:.1f}%)")
        print("-" * 60)

        return self.remove_outliers(df_result)

    def remove_outliers(self, df):
        if 'is_outlier' not in df.columns:
            raise ValueError("Run detect_monthly_outliers first!")

        print("\nRemoving outliers...")
        print("-" * 60)

        initial = len(df)
        cleaned = df[~df['is_outlier']].copy()
        removed = initial - len(cleaned)

        print(f"  Initial: {initial:,}")
        print(f"  Removed: {removed:,} ({removed / initial * 100:.1f}%)")
        print(f"  Final: {len(cleaned):,}")

        # Drop flag column
        cleaned = cleaned.drop(columns=['is_outlier'])

        print("-" * 60)

        return cleaned