import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    # Detects concept drift in consumption patterns and model performance
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config['monitoring']['drift_detection']['window_size']
        self.threshold = config['monitoring']['drift_detection']['threshold']

        # Storage for historical data
        self.performance_history = []
        self.feature_distributions = []
        self.prediction_history = []
        self.actual_history = []

        # Drift alerts
        self.alerts = []

    def check_prediction_drift(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               timestamp: datetime) -> Dict:
        # Calculate errors
        errors = np.abs(y_true - y_pred)
        mae = np.mean(errors)

        # Add to history
        self.performance_history.append({
            'timestamp': timestamp,
            'mae': mae,
            'n_samples': len(y_true)
        })

        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size * 2:]

        # Check for drift
        if len(self.performance_history) >= self.window_size * 2:
            # Compare recent performance to baseline
            recent = [p['mae'] for p in self.performance_history[-self.window_size:]]
            baseline = [p['mae'] for p in self.performance_history[:self.window_size]]

            recent_mean = np.mean(recent)
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)

            if baseline_std > 0:
                z_score = (recent_mean - baseline_mean) / baseline_std

                drift_detected = abs(z_score) > self.threshold

                if drift_detected:
                    alert = {
                        'timestamp': timestamp,
                        'type': 'prediction_error_drift',
                        'z_score': float(z_score),
                        'recent_mae': float(recent_mean),
                        'baseline_mae': float(baseline_mean),
                        'severity': 'high' if abs(z_score) > self.threshold * 1.5 else 'medium'
                    }
                    self.alerts.append(alert)
                    logger.warning(f"Drift detected: z_score={z_score:.2f}")

                    return alert

        return {'drift_detected': False}

    def check_feature_drift(self, current_features: Dict,
                           timestamp: datetime) -> Dict:
        # Add to history
        self.feature_distributions.append({
            'timestamp': timestamp,
            'features': current_features
        })

        # Keep only recent history
        if len(self.feature_distributions) > self.window_size * 2:
            self.feature_distributions = self.feature_distributions[-self.window_size * 2:]

        # Need enough history
        if len(self.feature_distributions) < self.window_size * 2:
            return {'drift_detected': False}

        # Extract recent and baseline features
        recent_features = self.feature_distributions[-self.window_size:]
        baseline_features = self.feature_distributions[:self.window_size]

        # Check each feature
        drifted_features = []

        for feature_name in current_features.keys():
            recent_values = [f['features'].get(feature_name, 0) for f in recent_features]
            baseline_values = [f['features'].get(feature_name, 0) for f in baseline_features]

            # Statistical test
            try:
                if len(recent_values) >= 5 and len(baseline_values) >= 5:
                    stat, p_value = stats.ks_2samp(recent_values, baseline_values)

                    if p_value < 0.05:  # Significant difference
                        drifted_features.append({
                            'feature': feature_name,
                            'ks_statistic': float(stat),
                            'p_value': float(p_value)
                        })
            except Exception as e:
                logger.debug(f"Error testing feature {feature_name}: {e}")

        if drifted_features:
            alert = {
                'timestamp': timestamp,
                'type': 'feature_drift',
                'drifted_features': drifted_features,
                'n_drifted': len(drifted_features),
                'severity': 'high' if len(drifted_features) > 3 else 'medium'
            }
            self.alerts.append(alert)
            logger.warning(f"Feature drift detected: {len(drifted_features)} features changed")

            return alert

        return {'drift_detected': False}

    def check_seasonal_drift(self, monthly_patterns: Dict[int, float],
                            timestamp: datetime) -> Dict:
        # This would compare with historical seasonal patterns
        # Placeholder implementation
        return {'drift_detected': False}

    def check_cluster_stability(self, cluster_assignments: Dict,
                               timestamp: datetime) -> Dict:
        if not hasattr(self, 'previous_clusters'):
            self.previous_clusters = cluster_assignments
            return {'stability': 1.0, 'changed': 0}

        # Calculate Jaccard similarity between clusterings
        # Simplified - count households that changed cluster
        changed = 0
        total = 0

        for account, cluster in cluster_assignments.items():
            if account in self.previous_clusters:
                total += 1
                if cluster != self.previous_clusters[account]:
                    changed += 1

        if total > 0:
            stability = 1.0 - (changed / total)

            if stability < 0.8:  # Threshold
                alert = {
                    'timestamp': timestamp,
                    'type': 'cluster_instability',
                    'stability': float(stability),
                    'changed_households': changed,
                    'severity': 'high' if stability < 0.6 else 'medium'
                }
                self.alerts.append(alert)
                logger.warning(f"Cluster instability detected: stability={stability:.2f}")

                return alert

        self.previous_clusters = cluster_assignments
        return {'drift_detected': False, 'stability': stability if total > 0 else 1.0}

    def detect_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                  features: Dict, cluster_assignments: Dict,
                  timestamp: datetime) -> Dict:
        results = {
            'timestamp': timestamp.isoformat(),
            'drift_detected': False,
            'alerts': []
        }

        # Check prediction drift
        pred_result = self.check_prediction_drift(y_true, y_pred, timestamp)
        if pred_result.get('drift_detected', False):
            results['drift_detected'] = True
            results['alerts'].append(pred_result)

        # Check feature drift
        feature_result = self.check_feature_drift(features, timestamp)
        if feature_result.get('drift_detected', False):
            results['drift_detected'] = True
            results['alerts'].append(feature_result)

        # Check cluster stability
        cluster_result = self.check_cluster_stability(cluster_assignments, timestamp)
        if cluster_result.get('drift_detected', False):
            results['drift_detected'] = True
            results['alerts'].append(cluster_result)

        results['alert_count'] = len(results['alerts'])

        return results

    def get_drift_summary(self) -> Dict:
        if not self.alerts:
            return {'status': 'healthy', 'message': 'No drift detected'}

        # Count by type
        alert_types = {}
        severities = {'high': 0, 'medium': 0, 'low': 0}

        for alert in self.alerts:
            alert_type = alert.get('type', 'unknown')
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

            severity = alert.get('severity', 'medium')
            severities[severity] = severities.get(severity, 0) + 1

        # Most recent alerts
        recent = self.alerts[-5:] if self.alerts else []

        return {
            'status': 'drift_detected' if self.alerts else 'healthy',
            'total_alerts': len(self.alerts),
            'by_type': alert_types,
            'by_severity': severities,
            'recent_alerts': recent,
            'recommendation': self._get_recommendation(alert_types, severities)
        }

    def _get_recommendation(self, alert_types: Dict, severities: Dict) -> str:
        if severities.get('high', 0) > 0:
            return "URGENT: Model retraining recommended immediately"
        elif severities.get('medium', 0) > 3:
            return "WARNING: Consider retraining model within 1 month"
        elif alert_types.get('feature_drift', 0) > 0:
            return "INFO: Feature distributions changing - monitor closely"
        else:
            return "OK: No action needed"

    def save_alerts(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                'alerts': self.alerts,
                'summary': self.get_drift_summary()
            }, f, indent=2, default=str)

        logger.info(f"Drift alerts saved to {filepath}")