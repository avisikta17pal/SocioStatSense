"""Change point detection for identifying structural breaks in time series."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import ruptures as rpt
from scipy import stats
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ChangePointDetector:
    """Detects structural breaks and regime changes in time series data."""
    
    def __init__(self, 
                 method: str = "pelt",
                 model: str = "rbf",
                 min_size: int = 24,
                 jump: int = 1,
                 pen: float = 10.0):
        
        self.method = method
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.pen = pen
        
        # History of detected change points
        self.change_point_history = []
        
    def detect_change_points(self, data: pd.DataFrame, 
                           columns: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Detect change points in specified columns."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        change_points = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 2 * self.min_size:
                logger.warning(f"Insufficient data for change point detection in {col}: {len(series)} points")
                continue
            
            try:
                # Detect change points using ruptures
                cps = self._detect_univariate_change_points(series.values)
                
                # Convert to timestamps and add metadata
                cp_info = []
                for cp_idx in cps:
                    if cp_idx < len(series):
                        timestamp = series.index[cp_idx]
                        confidence = self._calculate_confidence(series.values, cp_idx)
                        
                        cp_info.append({
                            'timestamp': timestamp,
                            'index': cp_idx,
                            'confidence': confidence,
                            'change_magnitude': self._calculate_change_magnitude(series.values, cp_idx),
                            'change_type': self._classify_change_type(series.values, cp_idx)
                        })
                
                change_points[col] = cp_info
                logger.info(f"Detected {len(cp_info)} change points in {col}")
                
            except Exception as e:
                logger.error(f"Error detecting change points in {col}: {str(e)}")
                change_points[col] = []
        
        # Update history
        self._update_change_point_history(change_points)
        
        return change_points
    
    def _detect_univariate_change_points(self, signal: np.ndarray) -> List[int]:
        """Detect change points in a univariate signal."""
        if self.method == "pelt":
            # PELT (Pruned Exact Linear Time) algorithm
            if self.model == "rbf":
                algo = rpt.Pelt(model="rbf").fit(signal)
            elif self.model == "l2":
                algo = rpt.Pelt(model="l2").fit(signal)
            elif self.model == "normal":
                algo = rpt.Pelt(model="normal").fit(signal)
            else:
                algo = rpt.Pelt(model="rbf").fit(signal)
                
            change_points = algo.predict(pen=self.pen)
            
        elif self.method == "binseg":
            # Binary Segmentation
            algo = rpt.Binseg(model=self.model, min_size=self.min_size, jump=self.jump).fit(signal)
            change_points = algo.predict(pen=self.pen)
            
        elif self.method == "window":
            # Window-based detection
            algo = rpt.Window(width=self.min_size, model=self.model).fit(signal)
            change_points = algo.predict(pen=self.pen)
            
        else:
            raise ValueError(f"Unknown change point detection method: {self.method}")
        
        # Remove the last point (end of series) and filter by minimum distance
        change_points = [cp for cp in change_points[:-1] if cp >= self.min_size]
        
        return change_points
    
    def _calculate_confidence(self, signal: np.ndarray, cp_idx: int) -> float:
        """Calculate confidence score for a change point."""
        if cp_idx <= 0 or cp_idx >= len(signal) - 1:
            return 0.0
        
        # Calculate variance before and after change point
        before_segment = signal[max(0, cp_idx - self.min_size):cp_idx]
        after_segment = signal[cp_idx:min(len(signal), cp_idx + self.min_size)]
        
        if len(before_segment) < 2 or len(after_segment) < 2:
            return 0.0
        
        # Use F-test to compare variances
        var_before = np.var(before_segment)
        var_after = np.var(after_segment)
        
        if var_before == 0 or var_after == 0:
            return 0.0
        
        f_stat = var_after / var_before if var_after > var_before else var_before / var_after
        
        # Convert F-statistic to confidence (0-1)
        confidence = min(1.0, f_stat / 10.0)  # Normalize to 0-1 range
        
        return confidence
    
    def _calculate_change_magnitude(self, signal: np.ndarray, cp_idx: int) -> float:
        """Calculate the magnitude of change at a change point."""
        if cp_idx <= 0 or cp_idx >= len(signal) - 1:
            return 0.0
        
        # Calculate means before and after
        before_segment = signal[max(0, cp_idx - self.min_size):cp_idx]
        after_segment = signal[cp_idx:min(len(signal), cp_idx + self.min_size)]
        
        if len(before_segment) == 0 or len(after_segment) == 0:
            return 0.0
        
        mean_before = np.mean(before_segment)
        mean_after = np.mean(after_segment)
        
        # Normalize by standard deviation
        std_before = np.std(before_segment)
        if std_before == 0:
            std_before = 1.0
        
        magnitude = abs(mean_after - mean_before) / std_before
        return magnitude
    
    def _classify_change_type(self, signal: np.ndarray, cp_idx: int) -> str:
        """Classify the type of change (level, trend, variance)."""
        if cp_idx <= self.min_size or cp_idx >= len(signal) - self.min_size:
            return "unknown"
        
        # Segments for analysis
        before_segment = signal[cp_idx - self.min_size:cp_idx]
        after_segment = signal[cp_idx:cp_idx + self.min_size]
        
        # Calculate statistics
        mean_before = np.mean(before_segment)
        mean_after = np.mean(after_segment)
        var_before = np.var(before_segment)
        var_after = np.var(after_segment)
        
        # Calculate trends
        x_before = np.arange(len(before_segment))
        x_after = np.arange(len(after_segment))
        
        try:
            slope_before, _, _, _, _ = stats.linregress(x_before, before_segment)
            slope_after, _, _, _, _ = stats.linregress(x_after, after_segment)
        except:
            slope_before = slope_after = 0
        
        # Classification logic
        mean_change = abs(mean_after - mean_before)
        var_change = abs(var_after - var_before)
        slope_change = abs(slope_after - slope_before)
        
        # Normalize changes
        signal_std = np.std(signal)
        if signal_std > 0:
            mean_change_norm = mean_change / signal_std
            var_change_norm = var_change / (signal_std ** 2)
            slope_change_norm = slope_change / signal_std
        else:
            return "unknown"
        
        # Classify based on dominant change
        if var_change_norm > mean_change_norm and var_change_norm > slope_change_norm:
            return "variance_change"
        elif slope_change_norm > mean_change_norm:
            return "trend_change"
        else:
            return "level_change"
    
    def detect_multivariate_change_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect change points considering multiple variables simultaneously."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        # Prepare multivariate signal
        signal_matrix = data[numeric_cols].fillna(method='ffill').fillna(0).values
        
        if signal_matrix.shape[0] < 2 * self.min_size:
            logger.warning(f"Insufficient data for multivariate change point detection: {signal_matrix.shape[0]} points")
            return []
        
        try:
            # Use multivariate change point detection
            algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(signal_matrix)
            change_points_idx = algo.predict(pen=self.pen * signal_matrix.shape[1])  # Scale penalty by dimensions
            
            # Remove last point and convert to metadata
            multivariate_cps = []
            for cp_idx in change_points_idx[:-1]:
                if cp_idx < len(data):
                    timestamp = data.index[cp_idx]
                    
                    # Calculate which variables contributed most to the change point
                    contributing_vars = self._identify_contributing_variables(
                        signal_matrix, cp_idx, numeric_cols
                    )
                    
                    multivariate_cps.append({
                        'timestamp': timestamp,
                        'index': cp_idx,
                        'contributing_variables': contributing_vars,
                        'confidence': self._calculate_multivariate_confidence(signal_matrix, cp_idx),
                        'type': 'multivariate'
                    })
            
            logger.info(f"Detected {len(multivariate_cps)} multivariate change points")
            return multivariate_cps
            
        except Exception as e:
            logger.error(f"Error in multivariate change point detection: {str(e)}")
            return []
    
    def _identify_contributing_variables(self, signal_matrix: np.ndarray, 
                                       cp_idx: int, column_names: List[str]) -> Dict[str, float]:
        """Identify which variables contributed most to a multivariate change point."""
        contributions = {}
        
        for i, col_name in enumerate(column_names):
            if cp_idx <= self.min_size or cp_idx >= len(signal_matrix) - self.min_size:
                contributions[col_name] = 0.0
                continue
            
            # Calculate change magnitude for this variable
            before_segment = signal_matrix[cp_idx - self.min_size:cp_idx, i]
            after_segment = signal_matrix[cp_idx:cp_idx + self.min_size, i]
            
            mean_before = np.mean(before_segment)
            mean_after = np.mean(after_segment)
            std_before = np.std(before_segment)
            
            if std_before > 0:
                change_magnitude = abs(mean_after - mean_before) / std_before
            else:
                change_magnitude = 0.0
            
            contributions[col_name] = change_magnitude
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v / total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def _calculate_multivariate_confidence(self, signal_matrix: np.ndarray, cp_idx: int) -> float:
        """Calculate confidence for multivariate change point."""
        if cp_idx <= self.min_size or cp_idx >= len(signal_matrix) - self.min_size:
            return 0.0
        
        # Use Hotelling's T² test for multivariate mean difference
        before_segment = signal_matrix[cp_idx - self.min_size:cp_idx, :]
        after_segment = signal_matrix[cp_idx:cp_idx + self.min_size, :]
        
        try:
            # Calculate pooled covariance
            cov_before = np.cov(before_segment.T)
            cov_after = np.cov(after_segment.T)
            pooled_cov = (cov_before + cov_after) / 2
            
            # Calculate mean difference
            mean_diff = np.mean(after_segment, axis=0) - np.mean(before_segment, axis=0)
            
            # Hotelling's T² statistic
            if np.linalg.det(pooled_cov) != 0:
                t_squared = mean_diff.T @ np.linalg.inv(pooled_cov) @ mean_diff
                confidence = min(1.0, t_squared / 100.0)  # Normalize to 0-1
            else:
                confidence = 0.0
                
        except Exception:
            confidence = 0.0
        
        return confidence
    
    def _update_change_point_history(self, change_points: Dict[str, List[Dict[str, Any]]]) -> None:
        """Update the history of detected change points."""
        timestamp = datetime.now()
        
        for variable, cps in change_points.items():
            for cp in cps:
                history_entry = {
                    'detection_timestamp': timestamp,
                    'variable': variable,
                    'change_point_timestamp': cp['timestamp'],
                    'confidence': cp['confidence'],
                    'change_type': cp.get('change_type', 'unknown'),
                    'magnitude': cp.get('change_magnitude', 0.0)
                }
                self.change_point_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.change_point_history) > 1000:
            self.change_point_history = self.change_point_history[-1000:]
    
    def get_recent_change_points(self, hours_back: int = 168) -> List[Dict[str, Any]]:
        """Get change points detected in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_cps = [
            cp for cp in self.change_point_history
            if cp['change_point_timestamp'] >= cutoff_time
        ]
        
        logger.info(f"Found {len(recent_cps)} change points in last {hours_back} hours")
        return recent_cps
    
    def detect_regime_changes(self, data: pd.DataFrame, 
                            window_size: int = 168) -> Dict[str, Any]:
        """Detect regime changes using sliding window analysis."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0 or len(data) < 2 * window_size:
            return {}
        
        regime_changes = {}
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < 2 * window_size:
                continue
            
            # Sliding window regime detection
            regimes = []
            for i in range(window_size, len(series) - window_size, window_size // 2):
                current_window = series.iloc[i:i + window_size]
                previous_window = series.iloc[i - window_size:i]
                
                # Statistical tests for regime change
                regime_info = self._test_regime_change(
                    previous_window.values, 
                    current_window.values,
                    series.index[i]
                )
                
                if regime_info['significant']:
                    regimes.append(regime_info)
            
            regime_changes[col] = regimes
            logger.info(f"Detected {len(regimes)} potential regime changes in {col}")
        
        return regime_changes
    
    def _test_regime_change(self, window1: np.ndarray, window2: np.ndarray, 
                           timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Test for regime change between two windows."""
        # Mean change test
        _, mean_p_value = stats.ttest_ind(window1, window2)
        
        # Variance change test
        _, var_p_value = stats.levene(window1, window2)
        
        # Distribution change test (Kolmogorov-Smirnov)
        _, ks_p_value = stats.ks_2samp(window1, window2)
        
        # Combine p-values using Fisher's method
        combined_statistic = -2 * (np.log(mean_p_value) + np.log(var_p_value) + np.log(ks_p_value))
        combined_p_value = stats.chi2.sf(combined_statistic, df=6)
        
        regime_info = {
            'timestamp': timestamp,
            'mean_change_p': mean_p_value,
            'variance_change_p': var_p_value,
            'distribution_change_p': ks_p_value,
            'combined_p_value': combined_p_value,
            'significant': combined_p_value < 0.05,
            'mean_before': np.mean(window1),
            'mean_after': np.mean(window2),
            'var_before': np.var(window1),
            'var_after': np.var(window2)
        }
        
        return regime_info
    
    def monitor_change_point_frequency(self) -> Dict[str, Any]:
        """Monitor the frequency of change point detection."""
        if not self.change_point_history:
            return {}
        
        df_history = pd.DataFrame(self.change_point_history)
        
        # Group by variable and time periods
        summary = {}
        for variable in df_history['variable'].unique():
            var_data = df_history[df_history['variable'] == variable]
            
            # Calculate frequency metrics
            summary[variable] = {
                'total_change_points': len(var_data),
                'avg_confidence': var_data['confidence'].mean(),
                'change_types': var_data['change_type'].value_counts().to_dict(),
                'recent_24h': len(var_data[
                    var_data['detection_timestamp'] >= datetime.now() - timedelta(hours=24)
                ]),
                'recent_7d': len(var_data[
                    var_data['detection_timestamp'] >= datetime.now() - timedelta(days=7)
                ])
            }
        
        logger.info(f"Change point frequency analysis completed for {len(summary)} variables")
        return summary
    
    def get_change_point_alerts(self, confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get high-confidence change points as alerts."""
        recent_cps = self.get_recent_change_points(hours_back=24)
        
        alerts = [
            cp for cp in recent_cps
            if cp['confidence'] >= confidence_threshold
        ]
        
        # Sort by confidence
        alerts.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Generated {len(alerts)} change point alerts")
        return alerts
    
    def visualize_change_points(self, data: pd.DataFrame, 
                              variable: str) -> Dict[str, Any]:
        """Prepare data for change point visualization."""
        if variable not in data.columns:
            return {}
        
        series = data[variable].dropna()
        change_points = self.detect_change_points(data, [variable])
        
        viz_data = {
            'timestamps': series.index.tolist(),
            'values': series.values.tolist(),
            'change_points': []
        }
        
        if variable in change_points:
            for cp in change_points[variable]:
                viz_data['change_points'].append({
                    'timestamp': cp['timestamp'],
                    'confidence': cp['confidence'],
                    'type': cp.get('change_type', 'unknown'),
                    'magnitude': cp.get('change_magnitude', 0.0)
                })
        
        return viz_data