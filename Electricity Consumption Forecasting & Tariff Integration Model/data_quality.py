import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    #report structure
    timestamp: str
    total_records: int
    unique_accounts: int
    missing_values: Dict[str, int]
    outlier_counts: Dict[str, int]
    zero_consumption_months: int
    negative_consumption_months: int
    seasonal_consistency_score: float
    data_completeness_score: float
    quality_issues: List[str]
    overall_score: float


class DataQualityMonitor:
    #Monitors and reports on data quality issues

    def __init__(self, config: Dict):
        """
        Initialize quality monitor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.issues_log = []
        self.quality_history = []

    def check_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        # Comprehensive data quality check

        issues = []
        score = 1.0

        # 1. Basic statistics
        total_records = len(df)
        unique_accounts = df['ACCOUNT_NO'].nunique() if 'ACCOUNT_NO' in df.columns else 0

        # 2. Missing values
        missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing[col] = int(missing_count)
                issues.append(f"Missing values in {col}: {missing_count}")
                score -= 0.05 * (missing_count / total_records)

        # 3. Outlier detection
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outliers[col] = int(outlier_count)
                if outlier_count > total_records * 0.01:  # >1% outliers
                    issues.append(f"High outliers in {col}: {outlier_count}")
                    score -= 0.1

        # 4. Zero consumption (non-solar households)
        zero_count = 0
        if 'NET_CONSUMPTION_kWh' in df.columns:
            if 'HAS_SOLAR' in df.columns:
                zero_mask = (df['NET_CONSUMPTION_kWh'] == 0) & (df['HAS_SOLAR'] == 0)
            else:
                zero_mask = df['NET_CONSUMPTION_kWh'] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                issues.append(f"Zero consumption records: {zero_count}")
                score -= 0.05 * (zero_count / total_records)

        # 5. Negative consumption
        negative_count = 0
        if 'NET_CONSUMPTION_kWh' in df.columns:
            negative_mask = df['NET_CONSUMPTION_kWh'] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                issues.append(f"Negative consumption records: {negative_count}")
                score -= 0.1 * (negative_count / total_records)

        # 6. Seasonal consistency
        seasonal_score = self._check_seasonal_consistency(df)
        if seasonal_score < 0.7:
            issues.append(f"Low seasonal consistency: {seasonal_score:.2f}")
            score -= 0.1

        # 7. Data completeness by account
        completeness_score = self._check_completeness(df)
        if completeness_score < 0.8:
            issues.append(f"Low data completeness: {completeness_score:.2f}")
            score -= 0.1

        # 8. Location validity
        if 'CUSTOMER_LAT' in df.columns and 'CUSTOMER_LON' in df.columns:
            bounds = self.config['similarity']['sri_lanka_bounds']
            invalid_lat = ((df['CUSTOMER_LAT'] < bounds['lat_min']) |
                          (df['CUSTOMER_LAT'] > bounds['lat_max'])).sum()
            invalid_lon = ((df['CUSTOMER_LON'] < bounds['lon_min']) |
                          (df['CUSTOMER_LON'] > bounds['lon_max'])).sum()
            if invalid_lat > 0 or invalid_lon > 0:
                issues.append(f"Invalid locations: {invalid_lat + invalid_lon}")
                score -= 0.05

        # 9. Tariff consistency
        if 'CAL_TARIFF' in df.columns:
            valid_tariffs = ['D1', 'GP1', 'GP2']
            invalid_tariff = (~df['CAL_TARIFF'].isin(valid_tariffs)).sum()
            if invalid_tariff > 0:
                issues.append(f"Invalid tariffs: {invalid_tariff}")
                score -= 0.05

        # Normalize score to 0-1
        score = max(0.1, min(1.0, score))

        # Create report
        report = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            total_records=total_records,
            unique_accounts=unique_accounts,
            missing_values=missing,
            outlier_counts=outliers,
            zero_consumption_months=int(zero_count),
            negative_consumption_months=int(negative_count),
            seasonal_consistency_score=float(seasonal_score),
            data_completeness_score=float(completeness_score),
            quality_issues=issues,
            overall_score=float(score)
        )

        # Log issues
        self._log_issues(report)
        self.quality_history.append(asdict(report))

        logger.info(f"Data quality score: {score:.2%} ({len(issues)} issues)")

        return report

    def _check_seasonal_consistency(self, df: pd.DataFrame) -> float:
        #Check if data follows expected seasonal patterns

        if 'MONTH' not in df.columns or 'NET_CONSUMPTION_kWh' not in df.columns:
            return 0.5

        # Calculate monthly averages
        monthly_avg = df.groupby('MONTH')['NET_CONSUMPTION_kWh'].mean()

        # Expected peak months in Sri Lanka (April, October)
        april_rank = monthly_avg.rank(ascending=False)[4] if 4 in monthly_avg.index else 6
        oct_rank = monthly_avg.rank(ascending=False)[10] if 10 in monthly_avg.index else 6

        # Expected low months (July, August)
        july_rank = monthly_avg.rank(ascending=True)[7] if 7 in monthly_avg.index else 6
        aug_rank = monthly_avg.rank(ascending=True)[8] if 8 in monthly_avg.index else 6

        # Calculate score (lower rank = better)
        peak_score = ((13 - april_rank) + (13 - oct_rank)) / 24
        low_score = ((13 - july_rank) + (13 - aug_rank)) / 24

        return (peak_score + low_score) / 2

    def _check_completeness(self, df: pd.DataFrame) -> float:
        #Check data completeness by household

        if 'ACCOUNT_NO' not in df.columns or 'MONTH' not in df.columns:
            return 0.5

        # Check if households have all 12 months
        completeness_by_account = df.groupby('ACCOUNT_NO')['MONTH'].nunique()
        households_with_all_months = (completeness_by_account == 12).sum()

        if len(completeness_by_account) > 0:
            return households_with_all_months / len(completeness_by_account)
        return 0.0

    def _log_issues(self, report: DataQualityReport):
        #Log quality issues for tracking
        issue_entry = {
            'timestamp': report.timestamp,
            'score': report.overall_score,
            'issues': report.quality_issues
        }
        self.issues_log.append(issue_entry)

    def get_quality_trend(self) -> Dict:
        #Get quality trend over time

        if len(self.quality_history) < 2:
            return {'trend': 'insufficient_data'}

        scores = [q['overall_score'] for q in self.quality_history]
        timestamps = [q['timestamp'] for q in self.quality_history]

        # Calculate trend
        from scipy import stats
        x = list(range(len(scores)))
        slope, _, r_value, p_value, _ = stats.linregress(x, scores)

        return {
            'trend': 'improving' if slope > 0.01 else 'degrading' if slope < -0.01 else 'stable',
            'slope': float(slope),
            'current_score': scores[-1],
            'min_score': min(scores),
            'max_score': max(scores),
            'volatility': float(np.std(scores))
        }

    def get_quality_summary(self) -> str:
        if not self.issues_log:
            return "✅ No data quality issues detected"

        latest = self.issues_log[-1]
        total_issues = len(self.issues_log)

        summary = []
        summary.append("📊 DATA QUALITY SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Current Score: {latest['score']:.1%}")
        summary.append(f"Total Checks: {total_issues}")
        summary.append(f"Current Issues: {len(latest['issues'])}")

        if latest['issues']:
            summary.append("\nTop Issues:")
            for issue in latest['issues'][:5]:
                summary.append(f"  • {issue}")

        # Get trend
        trend = self.get_quality_trend()
        summary.append(f"\nTrend: {trend.get('trend', 'unknown').upper()}")

        return "\n".join(summary)

    def get_recommendations(self) -> List[str]:
        if not self.issues_log:
            return ["No quality issues detected"]

        latest = self.issues_log[-1]
        recommendations = []

        if latest['score'] < 0.5:
            recommendations.append("URGENT: Major data quality issues - investigate immediately")

        if 'Missing values' in str(latest['issues']):
            recommendations.append("Address missing values through imputation or filtering")

        if 'outliers' in str(latest['issues']):
            recommendations.append("Review outlier handling - consider robust scaling")

        if 'Zero consumption' in str(latest['issues']):
            recommendations.append("Investigate zero consumption records - possible meter issues")

        if 'seasonal consistency' in str(latest['issues']):
            recommendations.append("Seasonal patterns deviate from expectations - review data source")

        if not recommendations:
            recommendations.append("Data quality acceptable - continue monitoring")

        return recommendations