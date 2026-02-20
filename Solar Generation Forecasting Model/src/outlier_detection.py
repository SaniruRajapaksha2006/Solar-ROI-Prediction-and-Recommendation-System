class OutlierDetector:

    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.outlier_bounds = {}
        self.removed_accounts = []


    def remove_high_export_accounts(self, df, max_export=700):
        print(f"\nRemoving accounts with avg export > {max_export} kWh...")
        print("-" * 60)

        # Calculate average export per account
        account_avg = df.groupby('ACCOUNT_NO')['EXPORT_kWh'].mean()

        # Find accounts to remove
        high_export_accounts = account_avg[account_avg > max_export].index.tolist()

        if not high_export_accounts:
            print(f"  ✓ No accounts with avg export > {max_export} kWh")
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