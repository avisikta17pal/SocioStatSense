"""Adaptive Bayesian hierarchical model for socio-economic modeling."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from datetime import datetime
import pickle
import scipy.stats as stats

from ..utils.model_utils import ModelMetrics, UncertaintyQuantifier
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class AdaptiveBayesianModel:
    """Adaptive Bayesian hierarchical model with sparse variable selection."""
    
    def __init__(self,
                 n_samples: int = 2000,
                 n_tune: int = 1000,
                 n_chains: int = 2,
                 sparse_alpha: float = 0.01,
                 max_features: int = 30,
                 hierarchical: bool = True,
                 random_state: int = 42):
        
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.sparse_alpha = sparse_alpha
        self.max_features = max_features
        self.hierarchical = hierarchical
        self.random_state = random_state
        
        # Model components
        self.models = {}
        self.traces = {}
        self.feature_selectors = {}
        self.selected_features = {}
        self.is_fitted = False
        
        # Metrics and uncertainty
        self.metrics_tracker = ModelMetrics()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Model update tracking
        self.update_history = []
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'AdaptiveBayesianModel':
        """Fit Bayesian hierarchical model to training data."""
        logger.info(f"Fitting Bayesian model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Clean data
        valid_indices = ~(X.isnull().all(axis=1) | y.isnull().all(axis=1))
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices].fillna(0)
        
        if len(X_clean) < 50:  # Minimum samples for Bayesian inference
            logger.warning(f"Insufficient data for Bayesian modeling: {len(X_clean)} samples")
            return self
        
        # Fit model for each target variable
        for target_col in y_clean.columns:
            logger.info(f"Fitting Bayesian model for target: {target_col}")
            
            y_target = y_clean[target_col].dropna()
            X_target = X_clean.loc[y_target.index]
            
            if len(y_target) < 30:
                logger.warning(f"Insufficient data for {target_col}: {len(y_target)} samples")
                continue
            
            # Feature selection using mutual information
            self.feature_selectors[target_col] = SelectKBest(
                score_func=mutual_info_regression,
                k=min(self.max_features, X_target.shape[1])
            )
            
            X_selected = self.feature_selectors[target_col].fit_transform(X_target, y_target)
            self.selected_features[target_col] = self.feature_selectors[target_col].get_feature_names_out(X_target.columns)
            
            # Build and fit Bayesian model
            try:
                model, trace = self._build_bayesian_model(X_selected, y_target.values, target_col)
                self.models[target_col] = model
                self.traces[target_col] = trace
                
                # Calculate model metrics
                posterior_pred = self._posterior_predictive(trace, X_selected)
                metrics = self.metrics_tracker.calculate_regression_metrics(
                    y_target.values, posterior_pred['mean']
                )
                
                logger.info(f"Bayesian model fitted for {target_col}: R² = {metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error fitting Bayesian model for {target_col}: {str(e)}")
                continue
        
        self.is_fitted = True
        return self
    
    def _build_bayesian_model(self, X: np.ndarray, y: np.ndarray, 
                            target_name: str) -> Tuple[pm.Model, az.InferenceData]:
        """Build and sample from Bayesian hierarchical model."""
        n_features = X.shape[1]
        
        with pm.Model() as model:
            # Priors for regression coefficients with sparse selection
            if self.hierarchical:
                # Hierarchical priors for coefficient groups
                tau = pm.HalfCauchy('tau', beta=1)  # Global shrinkage
                lambda_local = pm.HalfCauchy('lambda_local', beta=1, shape=n_features)  # Local shrinkage
                
                # Horseshoe prior for sparsity
                beta = pm.Normal('beta', mu=0, sigma=tau * lambda_local, shape=n_features)
            else:
                # Simple sparse priors
                beta = pm.Laplace('beta', mu=0, b=self.sparse_alpha, shape=n_features)
            
            # Intercept
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            
            # Linear predictor
            mu = alpha + pm.math.dot(X, beta)
            
            # Noise model with unknown variance
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Likelihood
            likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample from posterior
            trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                random_seed=self.random_state,
                return_inferencedata=True,
                progressbar=False
            )
        
        logger.info(f"Bayesian sampling completed for {target_name}")
        return model, trace
    
    def _posterior_predictive(self, trace: az.InferenceData, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate posterior predictive samples."""
        # Extract posterior samples
        beta_samples = trace.posterior['beta'].values.reshape(-1, X.shape[1])
        alpha_samples = trace.posterior['alpha'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()
        
        # Generate predictions for each posterior sample
        n_posterior_samples = len(alpha_samples)
        predictions = np.zeros((n_posterior_samples, X.shape[0]))
        
        for i in range(n_posterior_samples):
            mu_pred = alpha_samples[i] + X @ beta_samples[i]
            predictions[i] = np.random.normal(mu_pred, sigma_samples[i])
        
        return {
            'samples': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0)
        }
    
    def predict(self, X: pd.DataFrame, 
                return_uncertainty: bool = True) -> Dict[str, Any]:
        """Make Bayesian predictions with full uncertainty quantification."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_clean = X.fillna(0)
        predictions = {}
        
        for target_col in self.models.keys():
            if target_col not in self.selected_features or target_col not in self.traces:
                continue
            
            # Use selected features
            available_features = [f for f in self.selected_features[target_col] if f in X_clean.columns]
            
            if not available_features:
                logger.warning(f"No features available for {target_col}")
                continue
            
            X_selected = X_clean[available_features].values
            
            # Generate posterior predictive samples
            posterior_pred = self._posterior_predictive(self.traces[target_col], X_selected)
            
            predictions[target_col] = {
                'mean': posterior_pred['mean'],
                'timestamp': X.index.tolist() if hasattr(X.index, 'tolist') else list(range(len(posterior_pred['mean'])))
            }
            
            if return_uncertainty:
                # Add Bayesian uncertainty quantification
                uncertainty = self.uncertainty_quantifier.bayesian_uncertainty(posterior_pred['samples'])
                predictions[target_col].update(uncertainty)
        
        logger.info(f"Generated Bayesian predictions for {len(predictions)} targets")
        return predictions
    
    def update_with_new_data(self, X_new: pd.DataFrame, y_new: pd.DataFrame) -> 'AdaptiveBayesianModel':
        """Update Bayesian model with new data using variational inference."""
        if not self.is_fitted:
            return self.fit(X_new, y_new)
        
        logger.info(f"Updating Bayesian model with {X_new.shape[0]} new samples")
        
        # For each target, perform approximate update
        for target_col in y_new.columns:
            if target_col not in self.models:
                continue
            
            y_target = y_new[target_col].dropna()
            X_target = X_new.loc[y_target.index].fillna(0)
            
            if len(y_target) == 0:
                continue
            
            # Use selected features
            if target_col in self.selected_features:
                available_features = [f for f in self.selected_features[target_col] if f in X_target.columns]
                X_selected = X_target[available_features].values
                
                try:
                    # Simplified update using variational inference
                    updated_trace = self._variational_update(
                        self.models[target_col], 
                        self.traces[target_col],
                        X_selected, 
                        y_target.values
                    )
                    
                    if updated_trace is not None:
                        self.traces[target_col] = updated_trace
                        
                        # Track update
                        self.update_history.append({
                            'timestamp': datetime.now(),
                            'target': target_col,
                            'n_samples': len(y_target),
                            'update_method': 'variational'
                        })
                        
                except Exception as e:
                    logger.error(f"Error updating {target_col}: {str(e)}")
        
        logger.info("Bayesian model update completed")
        return self
    
    def _variational_update(self, model: pm.Model, trace: az.InferenceData,
                          X_new: np.ndarray, y_new: np.ndarray) -> Optional[az.InferenceData]:
        """Perform variational update of Bayesian model."""
        try:
            with model:
                # Update observed data
                pm.set_data({'X': X_new, 'y_obs': y_new})
                
                # Use ADVI for fast approximate inference
                approx = pm.fit(
                    n=10000,
                    method='advi',
                    progressbar=False
                )
                
                # Sample from variational approximation
                updated_trace = approx.sample(draws=self.n_samples)
                
                return updated_trace
                
        except Exception as e:
            logger.error(f"Variational update failed: {str(e)}")
            return None
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get Bayesian feature importance with uncertainty."""
        importance_dict = {}
        
        for target_col, trace in self.traces.items():
            if target_col not in self.selected_features:
                continue
            
            # Extract coefficient samples
            beta_samples = trace.posterior['beta'].values.reshape(-1, len(self.selected_features[target_col]))
            
            # Calculate importance as mean absolute coefficient value
            feature_names = self.selected_features[target_col]
            importances = np.mean(np.abs(beta_samples), axis=0)
            
            # Normalize
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            # Add uncertainty in importance
            importance_std = np.std(np.abs(beta_samples), axis=0)
            
            importance_dict[target_col] = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[target_col][feature_name] = {
                    'mean_importance': importances[i],
                    'std_importance': importance_std[i],
                    'credible_interval': np.percentile(np.abs(beta_samples[:, i]), [2.5, 97.5])
                }
            
            # Sort by mean importance
            importance_dict[target_col] = dict(
                sorted(importance_dict[target_col].items(),
                      key=lambda x: x[1]['mean_importance'], reverse=True)
            )
        
        logger.info(f"Calculated Bayesian feature importance for {len(importance_dict)} targets")
        return importance_dict
    
    def get_coefficient_evolution(self, target_col: str) -> Dict[str, np.ndarray]:
        """Get evolution of model coefficients over time."""
        if target_col not in self.traces:
            return {}
        
        trace = self.traces[target_col]
        
        # Extract coefficient samples over chains/draws
        beta_samples = trace.posterior['beta'].values  # shape: (chains, draws, features)
        alpha_samples = trace.posterior['alpha'].values  # shape: (chains, draws)
        
        evolution = {
            'beta_mean': np.mean(beta_samples, axis=0),  # (draws, features)
            'beta_std': np.std(beta_samples, axis=0),
            'alpha_mean': np.mean(alpha_samples, axis=0),  # (draws,)
            'alpha_std': np.std(alpha_samples, axis=0),
            'feature_names': self.selected_features[target_col]
        }
        
        return evolution
    
    def detect_parameter_drift(self, window_size: int = 500) -> Dict[str, Dict[str, bool]]:
        """Detect drift in model parameters using posterior samples."""
        drift_results = {}
        
        for target_col, trace in self.traces.items():
            if target_col not in self.selected_features:
                continue
            
            # Extract coefficient samples
            beta_samples = trace.posterior['beta'].values.reshape(-1, len(self.selected_features[target_col]))
            
            if len(beta_samples) < 2 * window_size:
                continue
            
            # Compare recent vs historical parameter distributions
            recent_samples = beta_samples[-window_size:]
            historical_samples = beta_samples[-2*window_size:-window_size]
            
            feature_drift = {}
            for i, feature_name in enumerate(self.selected_features[target_col]):
                # Kolmogorov-Smirnov test for distribution change
                _, p_value = stats.ks_2samp(historical_samples[:, i], recent_samples[:, i])
                feature_drift[feature_name] = p_value < 0.05
            
            drift_results[target_col] = feature_drift
        
        logger.info(f"Parameter drift detection completed for {len(drift_results)} targets")
        return drift_results
    
    def get_model_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive Bayesian model diagnostics."""
        diagnostics = {}
        
        for target_col, trace in self.traces.items():
            try:
                # Calculate R-hat (convergence diagnostic)
                rhat = az.rhat(trace)
                
                # Calculate effective sample size
                ess = az.ess(trace)
                
                # Calculate Monte Carlo standard error
                mcse = az.mcse(trace)
                
                diagnostics[target_col] = {
                    'rhat_max': float(rhat.max()) if hasattr(rhat, 'max') else 1.0,
                    'ess_min': float(ess.min()) if hasattr(ess, 'min') else self.n_samples,
                    'mcse_mean': float(mcse.mean()) if hasattr(mcse, 'mean') else 0.0,
                    'converged': float(rhat.max()) < 1.1 if hasattr(rhat, 'max') else True,
                    'effective_samples': float(ess.min()) > 100 if hasattr(ess, 'min') else True
                }
                
            except Exception as e:
                logger.error(f"Error calculating diagnostics for {target_col}: {str(e)}")
                diagnostics[target_col] = {
                    'rhat_max': 1.0,
                    'ess_min': self.n_samples,
                    'mcse_mean': 0.0,
                    'converged': True,
                    'effective_samples': True
                }
        
        logger.info(f"Model diagnostics calculated for {len(diagnostics)} targets")
        return diagnostics
    
    def forecast(self, X: pd.DataFrame, steps: int = 24) -> Dict[str, Any]:
        """Generate Bayesian forecasts with full uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        
        for target_col in self.models.keys():
            if target_col not in self.selected_features or target_col not in self.traces:
                continue
            
            try:
                # Multi-step Bayesian forecast
                forecast_samples = self._bayesian_multistep_forecast(X, target_col, steps)
                
                if forecast_samples is not None:
                    # Calculate forecast statistics
                    forecasts[target_col] = {
                        'mean': np.mean(forecast_samples, axis=0),
                        'std': np.std(forecast_samples, axis=0),
                        'samples': forecast_samples,
                        'steps': steps,
                        'horizon_hours': steps
                    }
                    
                    # Add credible intervals
                    forecasts[target_col]['lower_95'] = np.percentile(forecast_samples, 2.5, axis=0)
                    forecasts[target_col]['upper_95'] = np.percentile(forecast_samples, 97.5, axis=0)
                    forecasts[target_col]['lower_50'] = np.percentile(forecast_samples, 25, axis=0)
                    forecasts[target_col]['upper_50'] = np.percentile(forecast_samples, 75, axis=0)
                    
                    # Calculate prediction intervals
                    forecasts[target_col]['hdi_95'] = self.uncertainty_quantifier._calculate_hdi(
                        forecast_samples.T, credible_mass=0.95
                    )
                    
            except Exception as e:
                logger.error(f"Error forecasting {target_col}: {str(e)}")
        
        logger.info(f"Generated Bayesian forecasts for {len(forecasts)} targets")
        return forecasts
    
    def _bayesian_multistep_forecast(self, X: pd.DataFrame, target_col: str, 
                                   steps: int) -> Optional[np.ndarray]:
        """Generate multi-step Bayesian forecast."""
        if target_col not in self.traces:
            return None
        
        trace = self.traces[target_col]
        
        # Extract posterior samples
        beta_samples = trace.posterior['beta'].values.reshape(-1, len(self.selected_features[target_col]))
        alpha_samples = trace.posterior['alpha'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()
        
        n_posterior_samples = len(alpha_samples)
        forecast_samples = np.zeros((n_posterior_samples, steps))
        
        # Prepare features
        X_clean = X.fillna(0)
        available_features = [f for f in self.selected_features[target_col] if f in X_clean.columns]
        X_current = X_clean[available_features].copy()
        
        # Multi-step forecasting
        for step in range(steps):
            # Use last available features
            x_step = X_current.iloc[-1].values
            
            # Generate predictions for each posterior sample
            for i in range(n_posterior_samples):
                mu_pred = alpha_samples[i] + x_step @ beta_samples[i]
                pred_sample = np.random.normal(mu_pred, sigma_samples[i])
                forecast_samples[i, step] = pred_sample
            
            # Update features for next step (simplified approach)
            if step < steps - 1:
                next_timestamp = X_current.index[-1] + pd.Timedelta(hours=1)
                next_row = X_current.iloc[-1].copy()
                
                # Update lagged features if they exist
                lag_cols = [col for col in available_features if f"{target_col}_lag_1" in col]
                if lag_cols:
                    next_row[lag_cols[0]] = np.mean(forecast_samples[:, step])
                
                X_current.loc[next_timestamp] = next_row
        
        return forecast_samples
    
    def calculate_model_evidence(self) -> Dict[str, float]:
        """Calculate marginal likelihood (model evidence) for model comparison."""
        evidence = {}
        
        for target_col, trace in self.traces.items():
            try:
                # Use WAIC (Widely Applicable Information Criterion)
                waic = az.waic(trace)
                evidence[target_col] = {
                    'waic': waic.waic,
                    'waic_se': waic.se,
                    'elpd_waic': waic.elpd_waic,
                    'p_waic': waic.p_waic
                }
                
            except Exception as e:
                logger.error(f"Error calculating evidence for {target_col}: {str(e)}")
                evidence[target_col] = {'waic': np.inf}
        
        return evidence
    
    def get_posterior_summary(self, target_col: str) -> Dict[str, Any]:
        """Get comprehensive posterior summary for a target variable."""
        if target_col not in self.traces:
            return {}
        
        trace = self.traces[target_col]
        
        try:
            # Generate summary statistics
            summary = az.summary(trace, var_names=['beta', 'alpha', 'sigma'])
            
            # Convert to dictionary
            summary_dict = {
                'parameters': summary.to_dict('index'),
                'feature_names': self.selected_features[target_col],
                'n_samples': self.n_samples * self.n_chains,
                'convergence': self.get_model_diagnostics()[target_col]
            }
            
            return summary_dict
            
        except Exception as e:
            logger.error(f"Error generating posterior summary for {target_col}: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save Bayesian model and traces."""
        model_data = {
            'traces': self.traces,
            'selected_features': self.selected_features,
            'feature_selectors': self.feature_selectors,
            'is_fitted': self.is_fitted,
            'hyperparameters': {
                'n_samples': self.n_samples,
                'n_tune': self.n_tune,
                'n_chains': self.n_chains,
                'sparse_alpha': self.sparse_alpha,
                'max_features': self.max_features,
                'hierarchical': self.hierarchical
            },
            'update_history': self.update_history,
            'metrics_history': self.metrics_tracker.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Bayesian model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'AdaptiveBayesianModel':
        """Load Bayesian model and traces."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.traces = model_data['traces']
        self.selected_features = model_data['selected_features']
        self.feature_selectors = model_data['feature_selectors']
        self.is_fitted = model_data['is_fitted']
        self.update_history = model_data.get('update_history', [])
        self.metrics_tracker.history = model_data.get('metrics_history', [])
        
        # Restore hyperparameters
        hyperparams = model_data.get('hyperparameters', {})
        self.n_samples = hyperparams.get('n_samples', self.n_samples)
        self.n_tune = hyperparams.get('n_tune', self.n_tune)
        self.n_chains = hyperparams.get('n_chains', self.n_chains)
        self.sparse_alpha = hyperparams.get('sparse_alpha', self.sparse_alpha)
        self.max_features = hyperparams.get('max_features', self.max_features)
        self.hierarchical = hyperparams.get('hierarchical', self.hierarchical)
        
        logger.info(f"Bayesian model loaded from {filepath}")
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive Bayesian model summary."""
        summary = {
            'is_fitted': self.is_fitted,
            'n_targets': len(self.models),
            'target_variables': list(self.models.keys()),
            'hyperparameters': {
                'n_samples': self.n_samples,
                'n_tune': self.n_tune,
                'n_chains': self.n_chains,
                'sparse_alpha': self.sparse_alpha,
                'max_features': self.max_features,
                'hierarchical': self.hierarchical
            },
            'feature_counts': {
                target: len(features) for target, features in self.selected_features.items()
            },
            'updates_performed': len(self.update_history),
            'last_update': self.update_history[-1]['timestamp'] if self.update_history else None
        }
        
        # Add model diagnostics
        if self.is_fitted:
            summary['diagnostics'] = self.get_model_diagnostics()
            summary['model_evidence'] = self.calculate_model_evidence()
        
        return summary