"""Intervention analysis for what-if scenarios and causal effect estimation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import networkx as nx
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InterventionAnalyzer:
    """Analyzer for causal interventions and what-if scenarios."""
    
    def __init__(self, 
                 causal_graph: Optional[nx.DiGraph] = None,
                 intervention_method: str = "do_calculus",
                 bootstrap_samples: int = 1000):
        
        self.causal_graph = causal_graph
        self.intervention_method = intervention_method
        self.bootstrap_samples = bootstrap_samples
        
        # Fitted models for intervention
        self.causal_models = {}
        self.baseline_data = None
        
    def set_causal_graph(self, causal_graph: nx.DiGraph) -> None:
        """Set the causal graph for intervention analysis."""
        self.causal_graph = causal_graph
        logger.info(f"Causal graph set with {causal_graph.number_of_nodes()} nodes, {causal_graph.number_of_edges()} edges")
    
    def fit_causal_models(self, data: pd.DataFrame) -> 'InterventionAnalyzer':
        """Fit causal models for each variable based on the causal graph."""
        if self.causal_graph is None:
            logger.warning("No causal graph available - fitting models based on correlations")
            return self._fit_correlation_based_models(data)
        
        self.baseline_data = data.copy()
        logger.info(f"Fitting causal models for {self.causal_graph.number_of_nodes()} variables")
        
        # Fit a model for each variable
        for node in self.causal_graph.nodes():
            if node not in data.columns:
                continue
            
            # Get parents (causal variables) for this node
            parents = list(self.causal_graph.predecessors(node))
            available_parents = [p for p in parents if p in data.columns]
            
            if not available_parents:
                # No causal parents - use simple mean model
                self.causal_models[node] = {
                    'type': 'constant',
                    'value': data[node].mean(),
                    'parents': []
                }
                continue
            
            # Prepare training data
            y = data[node].dropna()
            X = data[available_parents].loc[y.index].fillna(0)
            
            if len(y) < 10:
                logger.warning(f"Insufficient data for {node}: {len(y)} samples")
                continue
            
            try:
                # Fit causal model (using Random Forest for non-linear relationships)
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                
                # Calculate model performance
                y_pred = model.predict(X)
                r2 = 1 - np.var(y - y_pred) / np.var(y)
                
                self.causal_models[node] = {
                    'type': 'random_forest',
                    'model': model,
                    'parents': available_parents,
                    'r2_score': r2,
                    'feature_importance': dict(zip(available_parents, model.feature_importances_))
                }
                
                logger.info(f"Fitted causal model for {node}: R² = {r2:.4f}, parents = {available_parents}")
                
            except Exception as e:
                logger.error(f"Error fitting model for {node}: {str(e)}")
                continue
        
        return self
    
    def _fit_correlation_based_models(self, data: pd.DataFrame) -> 'InterventionAnalyzer':
        """Fit models based on correlation structure when no causal graph is available."""
        self.baseline_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # For each variable, use highly correlated variables as predictors
        for target_var in numeric_cols:
            # Find highly correlated variables (excluding self)
            correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
            predictors = correlations[correlations > 0.3].index.tolist()
            predictors = [p for p in predictors if p != target_var][:5]  # Top 5 predictors
            
            if not predictors:
                self.causal_models[target_var] = {
                    'type': 'constant',
                    'value': data[target_var].mean(),
                    'parents': []
                }
                continue
            
            # Fit model
            y = data[target_var].dropna()
            X = data[predictors].loc[y.index].fillna(0)
            
            if len(y) < 10:
                continue
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                y_pred = model.predict(X)
                r2 = 1 - np.var(y - y_pred) / np.var(y)
                
                self.causal_models[target_var] = {
                    'type': 'linear',
                    'model': model,
                    'parents': predictors,
                    'r2_score': r2
                }
                
            except Exception as e:
                logger.error(f"Error fitting correlation-based model for {target_var}: {str(e)}")
        
        return self
    
    def simulate_intervention(self, 
                            interventions: Dict[str, float],
                            forecast_horizon: int = 24,
                            n_simulations: int = 100) -> Dict[str, Any]:
        """Simulate the effect of interventions on the system."""
        if not self.causal_models or self.baseline_data is None:
            raise ValueError("Causal models must be fitted before intervention simulation")
        
        logger.info(f"Simulating intervention: {interventions} over {forecast_horizon} steps")
        
        # Get baseline scenario (no intervention)
        baseline_forecast = self._simulate_scenario({}, forecast_horizon, n_simulations)
        
        # Get intervention scenario
        intervention_forecast = self._simulate_scenario(interventions, forecast_horizon, n_simulations)
        
        # Calculate causal effects
        causal_effects = self._calculate_causal_effects(baseline_forecast, intervention_forecast)
        
        return {
            'interventions': interventions,
            'forecast_horizon': forecast_horizon,
            'baseline_forecast': baseline_forecast,
            'intervention_forecast': intervention_forecast,
            'causal_effects': causal_effects,
            'simulation_metadata': {
                'n_simulations': n_simulations,
                'timestamp': datetime.now(),
                'variables_affected': list(causal_effects.keys())
            }
        }
    
    def _simulate_scenario(self, interventions: Dict[str, float], 
                          horizon: int, n_simulations: int) -> Dict[str, Any]:
        """Simulate a scenario with given interventions."""
        # Get initial state from last observation
        initial_state = self.baseline_data.iloc[-1].copy()
        
        # Variables to track
        variables = list(self.causal_models.keys())
        
        # Storage for simulation results
        simulation_results = {var: np.zeros((n_simulations, horizon)) for var in variables}
        
        for sim in range(n_simulations):
            # Initialize current state
            current_state = initial_state.copy()
            
            # Apply interventions
            for var, value in interventions.items():
                if var in current_state.index:
                    current_state[var] = value
            
            # Simulate forward
            for step in range(horizon):
                next_state = current_state.copy()
                
                # Update each variable based on its causal model
                for var in variables:
                    if var in self.causal_models:
                        model_info = self.causal_models[var]
                        
                        if model_info['type'] == 'constant':
                            next_value = model_info['value']
                            
                        elif model_info['type'] in ['random_forest', 'linear']:
                            parents = model_info['parents']
                            if parents:
                                # Get parent values
                                parent_values = [current_state.get(p, 0) for p in parents]
                                
                                # Predict next value
                                X_pred = np.array(parent_values).reshape(1, -1)
                                next_value = model_info['model'].predict(X_pred)[0]
                                
                                # Add noise for realistic simulation
                                noise_std = 0.05 * abs(next_value)  # 5% noise
                                next_value += np.random.normal(0, noise_std)
                            else:
                                next_value = current_state[var]
                        else:
                            next_value = current_state[var]
                        
                        # Apply interventions (override predicted values)
                        if var in interventions:
                            next_value = interventions[var]
                        
                        next_state[var] = next_value
                        simulation_results[var][sim, step] = next_value
                
                current_state = next_state
        
        # Calculate summary statistics
        scenario_summary = {}
        for var in variables:
            scenario_summary[var] = {
                'mean': np.mean(simulation_results[var], axis=0),
                'std': np.std(simulation_results[var], axis=0),
                'percentile_5': np.percentile(simulation_results[var], 5, axis=0),
                'percentile_25': np.percentile(simulation_results[var], 25, axis=0),
                'percentile_75': np.percentile(simulation_results[var], 75, axis=0),
                'percentile_95': np.percentile(simulation_results[var], 95, axis=0),
                'all_simulations': simulation_results[var]
            }
        
        return scenario_summary
    
    def _calculate_causal_effects(self, baseline: Dict[str, Any], 
                                intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate causal effects by comparing baseline and intervention scenarios."""
        causal_effects = {}
        
        for var in baseline.keys():
            if var in intervention:
                baseline_mean = baseline[var]['mean']
                intervention_mean = intervention[var]['mean']
                
                # Calculate absolute and relative effects
                absolute_effect = intervention_mean - baseline_mean
                relative_effect = (intervention_mean - baseline_mean) / (baseline_mean + 1e-8) * 100
                
                # Calculate uncertainty in effects
                baseline_std = baseline[var]['std']
                intervention_std = intervention[var]['std']
                effect_uncertainty = np.sqrt(baseline_std**2 + intervention_std**2)
                
                causal_effects[var] = {
                    'absolute_effect': absolute_effect,
                    'relative_effect_percent': relative_effect,
                    'effect_uncertainty': effect_uncertainty,
                    'significant_effect': np.abs(absolute_effect) > 2 * effect_uncertainty,
                    'effect_direction': np.where(absolute_effect > 0, 'positive', 'negative'),
                    'max_absolute_effect': np.max(np.abs(absolute_effect)),
                    'time_to_peak_effect': np.argmax(np.abs(absolute_effect))
                }
        
        return causal_effects
    
    def analyze_policy_intervention(self, 
                                  policy_variable: str,
                                  policy_values: List[float],
                                  target_variables: List[str],
                                  horizon: int = 72) -> Dict[str, Any]:
        """Analyze the effect of different policy intervention levels."""
        if policy_variable not in self.causal_models:
            raise ValueError(f"Policy variable {policy_variable} not in causal models")
        
        logger.info(f"Analyzing policy intervention on {policy_variable} with {len(policy_values)} scenarios")
        
        policy_analysis = {
            'policy_variable': policy_variable,
            'policy_values': policy_values,
            'target_variables': target_variables,
            'scenarios': {}
        }
        
        # Simulate each policy scenario
        for policy_value in policy_values:
            intervention = {policy_variable: policy_value}
            
            # Run intervention simulation
            result = self.simulate_intervention(
                interventions=intervention,
                forecast_horizon=horizon,
                n_simulations=50  # Reduced for multiple scenarios
            )
            
            # Extract effects on target variables
            scenario_effects = {}
            for target_var in target_variables:
                if target_var in result['causal_effects']:
                    scenario_effects[target_var] = result['causal_effects'][target_var]
            
            policy_analysis['scenarios'][str(policy_value)] = {
                'intervention_value': policy_value,
                'causal_effects': scenario_effects,
                'forecast': {var: result['intervention_forecast'][var] for var in target_variables if var in result['intervention_forecast']}
            }
        
        # Find optimal policy value for each target
        policy_analysis['optimal_policies'] = self._find_optimal_policies(policy_analysis, target_variables)
        
        logger.info(f"Policy analysis completed for {len(policy_values)} scenarios")
        return policy_analysis
    
    def _find_optimal_policies(self, policy_analysis: Dict[str, Any], 
                             target_variables: List[str]) -> Dict[str, Any]:
        """Find optimal policy values for different objectives."""
        optimal_policies = {}
        
        for target_var in target_variables:
            # Extract effects for this target across all scenarios
            effects = []
            policy_values = []
            
            for scenario_key, scenario_data in policy_analysis['scenarios'].items():
                if target_var in scenario_data['causal_effects']:
                    effect = scenario_data['causal_effects'][target_var]['max_absolute_effect']
                    effects.append(effect)
                    policy_values.append(float(scenario_key))
            
            if effects:
                # Find policy that maximizes positive effect or minimizes negative effect
                max_effect_idx = np.argmax(effects)
                min_effect_idx = np.argmin(effects)
                
                optimal_policies[target_var] = {
                    'maximize_effect': {
                        'policy_value': policy_values[max_effect_idx],
                        'expected_effect': effects[max_effect_idx]
                    },
                    'minimize_effect': {
                        'policy_value': policy_values[min_effect_idx],
                        'expected_effect': effects[min_effect_idx]
                    }
                }
        
        return optimal_policies
    
    def estimate_treatment_effect(self, 
                                treatment_var: str,
                                outcome_var: str,
                                treatment_value: float,
                                control_value: float = None) -> Dict[str, Any]:
        """Estimate average treatment effect using causal models."""
        if treatment_var not in self.causal_models or outcome_var not in self.causal_models:
            raise ValueError("Treatment and outcome variables must be in causal models")
        
        if control_value is None:
            control_value = self.baseline_data[treatment_var].mean()
        
        logger.info(f"Estimating treatment effect: {treatment_var} = {treatment_value} vs {control_value}")
        
        # Simulate treatment scenario
        treatment_result = self.simulate_intervention(
            interventions={treatment_var: treatment_value},
            forecast_horizon=24,
            n_simulations=self.bootstrap_samples
        )
        
        # Simulate control scenario
        control_result = self.simulate_intervention(
            interventions={treatment_var: control_value},
            forecast_horizon=24,
            n_simulations=self.bootstrap_samples
        )
        
        # Calculate Average Treatment Effect (ATE)
        if outcome_var in treatment_result['intervention_forecast'] and outcome_var in control_result['intervention_forecast']:
            treatment_outcomes = treatment_result['intervention_forecast'][outcome_var]['all_simulations']
            control_outcomes = control_result['intervention_forecast'][outcome_var]['all_simulations']
            
            # Calculate ATE for each time step
            ate_timeseries = np.mean(treatment_outcomes - control_outcomes, axis=0)
            ate_uncertainty = np.std(treatment_outcomes - control_outcomes, axis=0)
            
            # Overall ATE (average over time)
            overall_ate = np.mean(ate_timeseries)
            overall_ate_std = np.std(np.mean(treatment_outcomes - control_outcomes, axis=1))
            
            # Calculate confidence intervals
            ate_ci_lower = overall_ate - 1.96 * overall_ate_std
            ate_ci_upper = overall_ate + 1.96 * overall_ate_std
            
            treatment_effect = {
                'treatment_variable': treatment_var,
                'outcome_variable': outcome_var,
                'treatment_value': treatment_value,
                'control_value': control_value,
                'average_treatment_effect': overall_ate,
                'ate_standard_error': overall_ate_std,
                'ate_confidence_interval': [ate_ci_lower, ate_ci_upper],
                'ate_timeseries': ate_timeseries,
                'ate_uncertainty_timeseries': ate_uncertainty,
                'significant': abs(overall_ate) > 1.96 * overall_ate_std,
                'effect_size': abs(overall_ate) / (np.std(control_outcomes) + 1e-8)
            }
            
            logger.info(f"Treatment effect estimated: ATE = {overall_ate:.4f} ± {overall_ate_std:.4f}")
            return treatment_effect
        
        return {}
    
    def counterfactual_analysis(self, 
                              observed_data: pd.DataFrame,
                              counterfactual_interventions: Dict[str, float],
                              time_point: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """Perform counterfactual analysis: what would have happened if...?"""
        if time_point is None:
            time_point = observed_data.index[-1]
        
        if time_point not in observed_data.index:
            raise ValueError(f"Time point {time_point} not in observed data")
        
        logger.info(f"Performing counterfactual analysis at {time_point}")
        
        # Get observed state at the time point
        observed_state = observed_data.loc[time_point]
        
        # Simulate counterfactual scenario
        counterfactual_result = self.simulate_intervention(
            interventions=counterfactual_interventions,
            forecast_horizon=1,  # Single time point
            n_simulations=self.bootstrap_samples
        )
        
        # Calculate counterfactual effects
        counterfactual_effects = {}
        
        for var in self.causal_models.keys():
            if var in observed_data.columns and var in counterfactual_result['intervention_forecast']:
                observed_value = observed_state[var]
                counterfactual_dist = counterfactual_result['intervention_forecast'][var]['all_simulations'][:, 0]
                
                counterfactual_mean = np.mean(counterfactual_dist)
                counterfactual_std = np.std(counterfactual_dist)
                
                counterfactual_effects[var] = {
                    'observed_value': observed_value,
                    'counterfactual_mean': counterfactual_mean,
                    'counterfactual_std': counterfactual_std,
                    'counterfactual_effect': counterfactual_mean - observed_value,
                    'effect_probability': np.mean(counterfactual_dist > observed_value),
                    'percentile_rank': stats.percentileofscore(counterfactual_dist, observed_value)
                }
        
        return {
            'time_point': time_point,
            'interventions': counterfactual_interventions,
            'counterfactual_effects': counterfactual_effects,
            'observed_state': observed_state.to_dict()
        }
    
    def sensitivity_analysis(self, 
                           intervention_var: str,
                           outcome_var: str,
                           intervention_range: Tuple[float, float],
                           n_points: int = 20) -> Dict[str, Any]:
        """Perform sensitivity analysis for intervention effects."""
        min_val, max_val = intervention_range
        intervention_values = np.linspace(min_val, max_val, n_points)
        
        logger.info(f"Performing sensitivity analysis: {intervention_var} -> {outcome_var}")
        
        sensitivity_results = {
            'intervention_variable': intervention_var,
            'outcome_variable': outcome_var,
            'intervention_values': intervention_values.tolist(),
            'effects': [],
            'uncertainties': []
        }
        
        # Test each intervention value
        for intervention_value in intervention_values:
            try:
                # Estimate treatment effect
                effect_result = self.estimate_treatment_effect(
                    treatment_var=intervention_var,
                    outcome_var=outcome_var,
                    treatment_value=intervention_value
                )
                
                if effect_result:
                    sensitivity_results['effects'].append(effect_result['average_treatment_effect'])
                    sensitivity_results['uncertainties'].append(effect_result['ate_standard_error'])
                else:
                    sensitivity_results['effects'].append(0.0)
                    sensitivity_results['uncertainties'].append(0.0)
                    
            except Exception as e:
                logger.error(f"Error in sensitivity analysis at {intervention_value}: {str(e)}")
                sensitivity_results['effects'].append(0.0)
                sensitivity_results['uncertainties'].append(0.0)
        
        # Calculate dose-response relationship
        sensitivity_results['dose_response'] = {
            'linear_slope': np.polyfit(intervention_values, sensitivity_results['effects'], 1)[0],
            'correlation': np.corrcoef(intervention_values, sensitivity_results['effects'])[0, 1],
            'optimal_dose': intervention_values[np.argmax(np.abs(sensitivity_results['effects']))]
        }
        
        logger.info("Sensitivity analysis completed")
        return sensitivity_results
    
    def multi_intervention_analysis(self, 
                                  intervention_combinations: List[Dict[str, float]],
                                  target_variables: List[str],
                                  horizon: int = 24) -> Dict[str, Any]:
        """Analyze multiple intervention combinations."""
        logger.info(f"Analyzing {len(intervention_combinations)} intervention combinations")
        
        multi_analysis = {
            'intervention_combinations': intervention_combinations,
            'target_variables': target_variables,
            'results': []
        }
        
        # Analyze each combination
        for i, interventions in enumerate(intervention_combinations):
            try:
                result = self.simulate_intervention(
                    interventions=interventions,
                    forecast_horizon=horizon,
                    n_simulations=50  # Reduced for multiple combinations
                )
                
                # Extract target variable effects
                combination_result = {
                    'combination_id': i,
                    'interventions': interventions,
                    'effects': {var: result['causal_effects'][var] for var in target_variables if var in result['causal_effects']},
                    'total_effect_magnitude': sum(
                        abs(result['causal_effects'][var]['max_absolute_effect'])
                        for var in target_variables if var in result['causal_effects']
                    )
                }
                
                multi_analysis['results'].append(combination_result)
                
            except Exception as e:
                logger.error(f"Error analyzing combination {i}: {str(e)}")
        
        # Rank combinations by effectiveness
        multi_analysis['results'].sort(key=lambda x: x['total_effect_magnitude'], reverse=True)
        
        # Find best combination for each target
        multi_analysis['best_combinations'] = self._find_best_combinations(multi_analysis, target_variables)
        
        logger.info("Multi-intervention analysis completed")
        return multi_analysis
    
    def _find_best_combinations(self, multi_analysis: Dict[str, Any], 
                              target_variables: List[str]) -> Dict[str, Any]:
        """Find best intervention combinations for each target variable."""
        best_combinations = {}
        
        for target_var in target_variables:
            best_positive = None
            best_negative = None
            max_positive_effect = 0
            max_negative_effect = 0
            
            for result in multi_analysis['results']:
                if target_var in result['effects']:
                    effect = result['effects'][target_var]['max_absolute_effect']
                    
                    if isinstance(effect, np.ndarray):
                        effect = np.max(effect)
                    
                    if effect > max_positive_effect:
                        max_positive_effect = effect
                        best_positive = result
                    elif effect < max_negative_effect:
                        max_negative_effect = effect
                        best_negative = result
            
            best_combinations[target_var] = {
                'best_positive_effect': best_positive,
                'best_negative_effect': best_negative
            }
        
        return best_combinations
    
    def generate_intervention_report(self, 
                                   intervention_results: Dict[str, Any]) -> str:
        """Generate a human-readable intervention analysis report."""
        report_lines = []
        
        report_lines.append("# Causal Intervention Analysis Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Intervention summary
        interventions = intervention_results['interventions']
        report_lines.append("## Interventions Applied")
        for var, value in interventions.items():
            report_lines.append(f"- {var}: {value:.4f}")
        report_lines.append("")
        
        # Causal effects summary
        report_lines.append("## Causal Effects Summary")
        causal_effects = intervention_results['causal_effects']
        
        for var, effects in causal_effects.items():
            effect_magnitude = effects['max_absolute_effect']
            if isinstance(effect_magnitude, np.ndarray):
                effect_magnitude = np.max(effect_magnitude)
            
            significance = "significant" if effects['significant_effect'] else "not significant"
            direction = effects['effect_direction']
            
            report_lines.append(f"### {var}")
            report_lines.append(f"- Maximum effect: {effect_magnitude:.4f} ({direction})")
            report_lines.append(f"- Statistical significance: {significance}")
            report_lines.append(f"- Time to peak effect: {effects['time_to_peak_effect']} hours")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        significant_effects = [
            (var, effects) for var, effects in causal_effects.items()
            if effects['significant_effect']
        ]
        
        if significant_effects:
            report_lines.append("### Variables with Significant Effects:")
            for var, effects in significant_effects:
                effect_mag = effects['max_absolute_effect']
                if isinstance(effect_mag, np.ndarray):
                    effect_mag = np.max(effect_mag)
                report_lines.append(f"- {var}: {effect_mag:.4f} ({effects['effect_direction']})")
        else:
            report_lines.append("No statistically significant effects detected.")
        
        return "\n".join(report_lines)
    
    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of intervention analysis capabilities."""
        summary = {
            'causal_graph_available': self.causal_graph is not None,
            'n_causal_models': len(self.causal_models),
            'modeled_variables': list(self.causal_models.keys()),
            'intervention_method': self.intervention_method,
            'baseline_data_available': self.baseline_data is not None
        }
        
        if self.causal_graph:
            summary['causal_graph_metrics'] = {
                'n_nodes': self.causal_graph.number_of_nodes(),
                'n_edges': self.causal_graph.number_of_edges(),
                'density': nx.density(self.causal_graph)
            }
        
        if self.causal_models:
            model_performance = {}
            for var, model_info in self.causal_models.items():
                if 'r2_score' in model_info:
                    model_performance[var] = model_info['r2_score']
            
            summary['model_performance'] = model_performance
            summary['average_model_r2'] = np.mean(list(model_performance.values())) if model_performance else 0
        
        return summary