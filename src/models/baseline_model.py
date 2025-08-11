"""Baseline regression model with online learning capabilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

from ..utils.model_utils import ModelMetrics, UncertaintyQuantifier
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BaselineRegressionModel:
    """Baseline multivariate regression model with online learning."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 alpha: float = 0.01,
                 l1_ratio: float = 0.15,
                 max_features: int = 50,
                 random_state: int = 42):
        
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize models
        self.models = {}
        self.feature_selectors = {}
        self.selected_features = {}
        self.is_fitted = False
        
        # Metrics tracking
        self.metrics_tracker = ModelMetrics()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Model history for incremental learning
        self.training_history = []
        
    def _initialize_model(self, target_name: str) -> SGDRegressor:
        """Initialize a single SGD regressor for a target variable."""
        return SGDRegressor(
            learning_rate='adaptive',
            eta0=self.learning_rate,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            warm_start=True,
            max_iter=1000
        )
    
    def _initialize_feature_selector(self, k: int) -> SelectKBest:
        """Initialize feature selector."""
        return SelectKBest(score_func=f_regression, k=min(k, self.max_features))
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaselineRegressionModel':
        """Fit the model to training data."""
        logger.info(f"Fitting baseline model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Remove rows with all NaN values
        valid_indices = ~(X.isnull().all(axis=1) | y.isnull().all(axis=1))
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices].fillna(0)
        
        if len(X_clean) == 0:
            logger.warning("No valid training data available")
            return self
        
        # Initialize models for each target
        for target_col in y_clean.columns:
            logger.info(f"Fitting model for target: {target_col}")
            
            # Feature selection
            y_target = y_clean[target_col].dropna()
            X_target = X_clean.loc[y_target.index]
            
            if len(y_target) < 10:  # Minimum samples required
                logger.warning(f"Insufficient data for {target_col}: {len(y_target)} samples")
                continue
            
            # Initialize feature selector
            self.feature_selectors[target_col] = self._initialize_feature_selector(
                min(self.max_features, X_target.shape[1])
            )
            
            # Select features
            X_selected = self.feature_selectors[target_col].fit_transform(X_target, y_target)
            self.selected_features[target_col] = self.feature_selectors[target_col].get_feature_names_out(X_target.columns)
            
            # Initialize and fit model
            self.models[target_col] = self._initialize_model(target_col)
            self.models[target_col].fit(X_selected, y_target)
            
            # Calculate initial metrics
            y_pred = self.models[target_col].predict(X_selected)
            metrics = self.metrics_tracker.calculate_regression_metrics(y_target.values, y_pred)
            
            logger.info(f"Model fitted for {target_col}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        
        self.is_fitted = True
        return self
    
    def partial_fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaselineRegressionModel':
        """Incrementally update the model with new data."""
        if not self.is_fitted:
            return self.fit(X, y)
        
        logger.info(f"Partial fitting with {X.shape[0]} new samples")
        
        # Remove rows with all NaN values
        valid_indices = ~(X.isnull().all(axis=1) | y.isnull().all(axis=1))
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices].fillna(0)
        
        if len(X_clean) == 0:
            logger.warning("No valid data for partial fit")
            return self
        
        # Update each model
        for target_col in y_clean.columns:
            if target_col not in self.models:
                logger.warning(f"Target {target_col} not in fitted models")
                continue
            
            y_target = y_clean[target_col].dropna()
            X_target = X_clean.loc[y_target.index]
            
            if len(y_target) == 0:
                continue
            
            # Use pre-selected features
            if target_col in self.selected_features:
                available_features = [f for f in self.selected_features[target_col] if f in X_target.columns]
                X_selected = X_target[available_features]
                
                # Partial fit
                self.models[target_col].partial_fit(X_selected, y_target)
                
                # Track metrics
                y_pred = self.models[target_col].predict(X_selected)
                metrics = self.metrics_tracker.calculate_regression_metrics(y_target.values, y_pred)
                self.metrics_tracker.track_metrics(datetime.now(), metrics)
        
        logger.info("Partial fitting completed")
        return self
    
    def predict(self, X: pd.DataFrame, 
                return_uncertainty: bool = True) -> Dict[str, Any]:
        """Make predictions with uncertainty quantification."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_clean = X.fillna(0)
        predictions = {}
        
        for target_col, model in self.models.items():
            if target_col not in self.selected_features:
                continue
                
            # Use selected features
            available_features = [f for f in self.selected_features[target_col] if f in X_clean.columns]
            
            if not available_features:
                logger.warning(f"No features available for {target_col}")
                continue
                
            X_selected = X_clean[available_features]
            
            # Make prediction
            y_pred = model.predict(X_selected)
            
            predictions[target_col] = {
                'mean': y_pred,
                'timestamp': X.index.tolist() if hasattr(X.index, 'tolist') else list(range(len(y_pred)))
            }
            
            # Add uncertainty if requested
            if return_uncertainty:
                # Use bootstrap for uncertainty estimation
                uncertainty = self.uncertainty_quantifier.bootstrap_uncertainty(
                    model, X_selected.values, n_bootstrap=50
                )
                predictions[target_col].update(uncertainty)
        
        logger.info(f"Generated predictions for {len(predictions)} targets")
        return predictions
    
    def forecast(self, X: pd.DataFrame, steps: int = 24) -> Dict[str, Any]:
        """Generate multi-step forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        
        for target_col in self.models.keys():
            if target_col not in self.selected_features:
                continue
            
            # Multi-step forecast using recursive prediction
            forecast_values = []
            forecast_uncertainty = []
            
            X_current = X.copy()
            
            for step in range(steps):
                # Predict next step
                pred_result = self.predict(X_current.tail(1), return_uncertainty=True)
                
                if target_col in pred_result:
                    pred_mean = pred_result[target_col]['mean'][0]
                    pred_std = pred_result[target_col].get('std', [0])[0]
                    
                    forecast_values.append(pred_mean)
                    forecast_uncertainty.append(pred_std)
                    
                    # Update X_current with prediction for next step
                    # This is a simplified approach - in practice, you'd need more sophisticated lag updating
                    next_timestamp = X_current.index[-1] + pd.Timedelta(hours=1)
                    
                    # Create next row (simplified - copy last row and update target)
                    next_row = X_current.iloc[-1].copy()
                    X_current.loc[next_timestamp] = next_row
                    
                    # Update lagged features if they exist
                    lag_cols = [col for col in X_current.columns if f"{target_col}_lag_1" in col]
                    if lag_cols:
                        X_current.loc[next_timestamp, lag_cols[0]] = pred_mean
                else:
                    break
            
            if forecast_values:
                forecasts[target_col] = {
                    'mean': np.array(forecast_values),
                    'std': np.array(forecast_uncertainty),
                    'steps': steps,
                    'horizon_hours': steps
                }
                
                # Add confidence intervals
                mean_forecast = np.array(forecast_values)
                std_forecast = np.array(forecast_uncertainty)
                
                forecasts[target_col]['lower_95'] = mean_forecast - 1.96 * std_forecast
                forecasts[target_col]['upper_95'] = mean_forecast + 1.96 * std_forecast
                forecasts[target_col]['lower_50'] = mean_forecast - 0.67 * std_forecast
                forecasts[target_col]['upper_50'] = mean_forecast + 0.67 * std_forecast
        
        logger.info(f"Generated {steps}-step forecasts for {len(forecasts)} targets")
        return forecasts
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each target variable."""
        importance_dict = {}
        
        for target_col, model in self.models.items():
            if target_col in self.selected_features and hasattr(model, 'coef_'):
                feature_names = self.selected_features[target_col]
                importances = np.abs(model.coef_)
                
                # Normalize importances
                if importances.sum() > 0:
                    importances = importances / importances.sum()
                
                importance_dict[target_col] = dict(zip(feature_names, importances))
                
                # Sort by importance
                importance_dict[target_col] = dict(
                    sorted(importance_dict[target_col].items(), 
                          key=lambda x: x[1], reverse=True)
                )
        
        logger.info(f"Calculated feature importance for {len(importance_dict)} targets")
        return importance_dict
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        model_data = {
            'models': self.models,
            'feature_selectors': self.feature_selectors,
            'selected_features': self.selected_features,
            'is_fitted': self.is_fitted,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'max_features': self.max_features
            },
            'metrics_history': self.metrics_tracker.history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaselineRegressionModel':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_selectors = model_data['feature_selectors']
        self.selected_features = model_data['selected_features']
        self.is_fitted = model_data['is_fitted']
        self.metrics_tracker.history = model_data.get('metrics_history', [])
        
        # Restore hyperparameters
        hyperparams = model_data.get('hyperparameters', {})
        self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
        self.alpha = hyperparams.get('alpha', self.alpha)
        self.l1_ratio = hyperparams.get('l1_ratio', self.l1_ratio)
        self.max_features = hyperparams.get('max_features', self.max_features)
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'is_fitted': self.is_fitted,
            'n_targets': len(self.models),
            'target_variables': list(self.models.keys()),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'max_features': self.max_features
            },
            'feature_counts': {
                target: len(features) for target, features in self.selected_features.items()
            },
            'training_samples': len(self.training_history),
            'last_update': self.metrics_tracker.history[-1]['timestamp'] if self.metrics_tracker.history else None
        }
        
        # Add latest performance metrics
        if self.metrics_tracker.history:
            latest_metrics = self.metrics_tracker.history[-1]
            summary['latest_performance'] = {
                k: v for k, v in latest_metrics.items() if k != 'timestamp'
            }
        
        return summary
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        evaluation_results = {}
        
        # Remove rows with all NaN values
        valid_indices = ~(X.isnull().all(axis=1) | y.isnull().all(axis=1))
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices].fillna(0)
        
        for target_col in y_clean.columns:
            if target_col not in self.models:
                continue
                
            y_target = y_clean[target_col].dropna()
            X_target = X_clean.loc[y_target.index]
            
            if len(y_target) == 0:
                continue
            
            # Use selected features
            if target_col in self.selected_features:
                available_features = [f for f in self.selected_features[target_col] if f in X_target.columns]
                X_selected = X_target[available_features]
                
                # Make predictions
                y_pred = self.models[target_col].predict(X_selected)
                
                # Calculate metrics
                metrics = self.metrics_tracker.calculate_regression_metrics(y_target.values, y_pred)
                evaluation_results[target_col] = metrics
        
        logger.info(f"Model evaluation completed for {len(evaluation_results)} targets")
        return evaluation_results
    
    def detect_concept_drift(self) -> Dict[str, bool]:
        """Detect concept drift in model performance."""
        return self.metrics_tracker.detect_model_drift()
    
    def retrain_if_drift(self, X: pd.DataFrame, y: pd.DataFrame, 
                        drift_threshold: float = 0.05) -> bool:
        """Retrain model if significant drift is detected."""
        drift_detected = self.detect_concept_drift()
        
        # Check if any target shows significant drift
        significant_drift = any(drift_detected.values())
        
        if significant_drift:
            logger.warning("Concept drift detected - retraining model")
            self.fit(X, y)
            return True
        
        return False
    
    def get_prediction_explanation(self, X: pd.DataFrame, 
                                 target_col: str) -> Dict[str, Any]:
        """Get explanation for predictions including feature contributions."""
        if target_col not in self.models or target_col not in self.selected_features:
            return {}
        
        X_clean = X.fillna(0)
        available_features = [f for f in self.selected_features[target_col] if f in X_clean.columns]
        X_selected = X_clean[available_features]
        
        # Get model coefficients
        model = self.models[target_col]
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            feature_contributions = X_selected.iloc[-1].values * coefficients
            
            explanation = {
                'prediction': model.predict(X_selected.tail(1))[0],
                'intercept': model.intercept_ if hasattr(model, 'intercept_') else 0,
                'feature_contributions': dict(zip(available_features, feature_contributions)),
                'total_contribution': feature_contributions.sum(),
                'top_contributors': dict(
                    sorted(
                        zip(available_features, feature_contributions),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:10]
                )
            }
            
            return explanation
        
        return {}
    
    def update_learning_rate(self, new_lr: float) -> None:
        """Update learning rate for all models."""
        self.learning_rate = new_lr
        for model in self.models.values():
            if hasattr(model, 'eta0'):
                model.eta0 = new_lr
        
        logger.info(f"Updated learning rate to {new_lr}")
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information for all models."""
        convergence_info = {}
        
        for target_col, model in self.models.items():
            info = {
                'n_iter': getattr(model, 'n_iter_', 0),
                'converged': getattr(model, 'n_iter_', 0) < 1000
            }
            convergence_info[target_col] = info
        
        return convergence_info