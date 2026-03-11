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