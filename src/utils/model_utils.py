"""Model utilities for the adaptive modeling framework."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .logging_config import get_logger

logger = get_logger(__name__)


class ModelMetrics:
    """Calculate and track model performance metrics."""
    
    def __init__(self):
        self.history = []
        
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'bias': np.mean(y_pred - y_true),
            'std_residuals': np.std(y_pred - y_true)
        }
        
        # Add directional accuracy for time series
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
            metrics['directional_accuracy'] = directional_accuracy
        
        return metrics
    
    def calculate_forecast_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 prediction_intervals: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Calculate forecast-specific metrics including coverage probability."""
        metrics = self.calculate_regression_metrics(y_true, y_pred)
        
        if prediction_intervals:
            for level, (lower, upper) in prediction_intervals.items():
                coverage = np.mean((y_true >= lower) & (y_true <= upper))
                metrics[f'coverage_{level}'] = coverage
                
                # Average interval width
                avg_width = np.mean(upper - lower)
                metrics[f'avg_interval_width_{level}'] = avg_width
        
        return metrics
    
    def track_metrics(self, timestamp: pd.Timestamp, metrics: Dict[str, float]) -> None:
        """Track metrics over time for monitoring model drift."""
        record = {'timestamp': timestamp, **metrics}
        self.history.append(record)
        
        # Keep only last 1000 records
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
            
        logger.info(f"Tracked metrics at {timestamp}: RMSE={metrics.get('rmse', 0):.4f}")
    
    def detect_model_drift(self, window_size: int = 50, 
                          drift_threshold: float = 0.1) -> Dict[str, bool]:
        """Detect model performance drift using sliding windows."""
        if len(self.history) < 2 * window_size:
            return {}
        
        drift_detected = {}
        df_history = pd.DataFrame(self.history)
        
        for metric in ['rmse', 'mae', 'r2']:
            if metric not in df_history.columns:
                continue
                
            recent_values = df_history[metric].tail(window_size).values
            historical_values = df_history[metric].iloc[-2*window_size:-window_size].values
            
            # Statistical test for difference in means
            _, p_value = stats.ttest_ind(recent_values, historical_values)
            drift_detected[metric] = p_value < drift_threshold
            
        logger.info(f"Model drift detection completed: {drift_detected}")
        return drift_detected


class UncertaintyQuantifier:
    """Quantify and manage prediction uncertainty."""
    
    def __init__(self, quantiles: List[float] = [0.05, 0.25, 0.75, 0.95]):
        self.quantiles = quantiles
        
    def bootstrap_uncertainty(self, model, X: np.ndarray, 
                            n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """Calculate uncertainty using bootstrap sampling."""
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            
            # Generate prediction
            pred = model.predict(X_bootstrap)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate quantiles
        uncertainty_bands = {}
        for q in self.quantiles:
            uncertainty_bands[f'q_{int(q*100)}'] = np.percentile(predictions, q*100, axis=0)
        
        uncertainty_bands['mean'] = np.mean(predictions, axis=0)
        uncertainty_bands['std'] = np.std(predictions, axis=0)
        
        logger.info(f"Bootstrap uncertainty calculated with {n_bootstrap} samples")
        return uncertainty_bands
    
    def bayesian_uncertainty(self, posterior_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate uncertainty from Bayesian posterior samples."""
        uncertainty_bands = {}
        
        # Calculate quantiles from posterior
        for q in self.quantiles:
            uncertainty_bands[f'q_{int(q*100)}'] = np.percentile(posterior_samples, q*100, axis=0)
        
        uncertainty_bands['mean'] = np.mean(posterior_samples, axis=0)
        uncertainty_bands['std'] = np.std(posterior_samples, axis=0)
        
        # Calculate highest density intervals
        hdi_95 = self._calculate_hdi(posterior_samples, 0.95)
        uncertainty_bands['hdi_lower_95'] = hdi_95[:, 0]
        uncertainty_bands['hdi_upper_95'] = hdi_95[:, 1]
        
        logger.info("Bayesian uncertainty quantification completed")
        return uncertainty_bands
    
    def _calculate_hdi(self, samples: np.ndarray, credible_mass: float = 0.95) -> np.ndarray:
        """Calculate Highest Density Interval (HDI)."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
            
        hdis = np.zeros((samples.shape[1], 2))
        
        for i in range(samples.shape[1]):
            sorted_samples = np.sort(samples[:, i])
            n_samples = len(sorted_samples)
            interval_size = int(np.ceil(credible_mass * n_samples))
            
            # Find the shortest interval
            interval_widths = sorted_samples[interval_size:] - sorted_samples[:-interval_size]
            min_idx = np.argmin(interval_widths)
            
            hdis[i, 0] = sorted_samples[min_idx]
            hdis[i, 1] = sorted_samples[min_idx + interval_size]
            
        return hdis
    
    def prediction_intervals_coverage(self, y_true: np.ndarray,
                                    prediction_intervals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate coverage probability for prediction intervals."""
        coverage_metrics = {}
        
        for level_name, bounds in prediction_intervals.items():
            if level_name.startswith('q_') and len(bounds) == len(y_true):
                # Extract confidence level from quantile name
                if 'q_5' in level_name or 'q_95' in level_name:
                    level = 0.90
                elif 'q_25' in level_name or 'q_75' in level_name:
                    level = 0.50
                else:
                    continue
                    
                # Find corresponding bounds
                lower_key = level_name if '5' in level_name else None
                upper_key = level_name if '95' in level_name or '75' in level_name else None
                
                if lower_key and upper_key:
                    lower_bounds = prediction_intervals[lower_key]
                    upper_bounds = prediction_intervals[upper_key]
                    coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
                    coverage_metrics[f'coverage_{int(level*100)}'] = coverage
        
        logger.info(f"Coverage probability calculated: {coverage_metrics}")
        return coverage_metrics


def calculate_information_criteria(log_likelihood: float, n_params: int, 
                                 n_samples: int) -> Dict[str, float]:
    """Calculate AIC, BIC, and other information criteria."""
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_samples) * n_params - 2 * log_likelihood
    aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
    
    return {
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
        'log_likelihood': log_likelihood
    }


def calculate_feature_importance(model, feature_names: List[str], 
                               method: str = "permutation") -> Dict[str, float]:
    """Calculate feature importance using various methods."""
    importance_dict = {}
    
    if hasattr(model, 'coef_') and method == "coefficients":
        # Linear model coefficients
        for name, coef in zip(feature_names, model.coef_):
            importance_dict[name] = abs(coef)
            
    elif hasattr(model, 'feature_importances_') and method == "tree_based":
        # Tree-based model importances
        for name, importance in zip(feature_names, model.feature_importances_):
            importance_dict[name] = importance
    
    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    logger.info(f"Feature importance calculated using {method} method")
    return sorted_importance