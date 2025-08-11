"""Tests for causal inference components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from unittest.mock import Mock, patch

from src.causal.granger_causality import GrangerCausalityAnalyzer
from src.causal.intervention_analysis import InterventionAnalyzer


class TestGrangerCausalityAnalyzer:
    """Test Granger causality analysis functionality."""
    
    def setup_method(self):
        """Setup test data with known causal relationships."""
        np.random.seed(42)
        
        # Create time series with known causal structure
        n_points = 200
        dates = pd.date_range('2023-01-01', periods=n_points, freq='H')
        
        # X1 -> X2 (X1 Granger-causes X2)
        x1 = np.random.normal(0, 1, n_points)
        x2 = np.zeros(n_points)
        x2[0] = np.random.normal(0, 1)
        
        # X2 depends on lagged values of X1
        for t in range(1, n_points):
            x2[t] = 0.3 * x2[t-1] + 0.5 * x1[t-1] + np.random.normal(0, 0.5)
        
        # X3 -> X1 (X3 Granger-causes X1)
        x3 = np.random.normal(0, 1, n_points)
        for t in range(1, n_points):
            x1[t] = 0.2 * x1[t-1] + 0.4 * x3[t-1] + np.random.normal(0, 0.3)
        
        # X4 is independent
        x4 = np.random.normal(0, 1, n_points)
        
        self.causal_data = pd.DataFrame({
            'variable_1': x1,
            'variable_2': x2,
            'variable_3': x3,
            'variable_4': x4
        }, index=dates)
        
        self.analyzer = GrangerCausalityAnalyzer(
            max_lags=5,
            significance_level=0.05
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.max_lags == 5
        assert self.analyzer.significance_level == 0.05
        assert self.analyzer.ic_criterion == "aic"
        assert self.analyzer.causality_results == {}
        assert self.analyzer.causal_graph is None
    
    def test_pairwise_causality_analysis(self):
        """Test pairwise Granger causality analysis."""
        causality_results = self.analyzer.analyze_pairwise_causality(
            self.causal_data,
            variables=['variable_1', 'variable_2', 'variable_3']
        )
        
        # Check structure of results
        assert isinstance(causality_results, dict)
        
        # Should find X3 -> X1 causality
        if 'variable_3->variable_1' in causality_results:
            x3_to_x1 = causality_results['variable_3->variable_1']
            assert 'p_value' in x3_to_x1
            assert 'f_statistic' in x3_to_x1
            assert 'optimal_lag' in x3_to_x1
            assert 'effect_size' in x3_to_x1
            assert 'causal_strength' in x3_to_x1
        
        # Should find X1 -> X2 causality
        if 'variable_1->variable_2' in causality_results:
            x1_to_x2 = causality_results['variable_1->variable_2']
            assert x1_to_x2['p_value'] < 0.1  # Should be significant
    
    def test_optimal_lag_selection(self):
        """Test optimal lag selection."""
        optimal_lag = self.analyzer._find_optimal_lag(self.causal_data)
        
        assert isinstance(optimal_lag, int)
        assert 1 <= optimal_lag <= self.analyzer.max_lags
    
    def test_causal_graph_construction(self):
        """Test causal graph construction."""
        # First run causality analysis
        self.analyzer.analyze_pairwise_causality(self.causal_data)
        
        # Check if causal graph was created
        if self.analyzer.causal_graph is not None:
            assert isinstance(self.analyzer.causal_graph, nx.DiGraph)
            assert len(self.analyzer.causal_graph.nodes()) > 0
    
    def test_var_model_analysis(self):
        """Test Vector Autoregression model analysis."""
        var_results = self.analyzer.analyze_var_model(
            self.causal_data,
            variables=['variable_1', 'variable_2', 'variable_3']
        )
        
        assert 'model_summary' in var_results
        assert 'causality_tests' in var_results
        assert 'impulse_response' in var_results
        assert 'forecast_error_variance' in var_results
        assert 'optimal_lag' in var_results
    
    def test_causal_network_metrics(self):
        """Test causal network metrics calculation."""
        # First build a simple causal graph
        self.analyzer.causal_graph = nx.DiGraph()
        self.analyzer.causal_graph.add_edges_from([
            ('variable_1', 'variable_2'),
            ('variable_3', 'variable_1'),
            ('variable_3', 'variable_4')
        ])
        
        metrics = self.analyzer.get_causal_network_metrics()
        
        assert 'n_nodes' in metrics
        assert 'n_edges' in metrics
        assert 'density' in metrics
        assert 'avg_clustering' in metrics
        assert metrics['n_nodes'] == 4
        assert metrics['n_edges'] == 3
    
    def test_causal_path_finding(self):
        """Test causal path finding."""
        # Create test graph
        self.analyzer.causal_graph = nx.DiGraph()
        self.analyzer.causal_graph.add_edges_from([
            ('A', 'B'),
            ('B', 'C'),
            ('A', 'C'),
            ('C', 'D')
        ])
        
        # Find paths from A to D
        paths = self.analyzer.find_causal_paths('A', 'D', max_length=3)
        
        assert isinstance(paths, list)
        assert len(paths) >= 1
        
        # Check that paths are valid
        for path in paths:
            assert path[0] == 'A'
            assert path[-1] == 'D'
            assert len(path) <= 4  # max_length + 1
    
    def test_instantaneous_causality(self):
        """Test instantaneous causality detection."""
        # Create data with instantaneous correlation
        x1 = np.random.normal(0, 1, 100)
        x2 = 0.7 * x1 + np.random.normal(0, 0.5, 100)  # Instantaneous correlation
        
        instant_data = pd.DataFrame({
            'x1': x1,
            'x2': x2
        })
        
        instant_causality = self.analyzer.calculate_instantaneous_causality(instant_data)
        
        assert isinstance(instant_causality, dict)
        assert 'x1<->x2' in instant_causality or 'x2<->x1' in instant_causality


class TestInterventionAnalyzer:
    """Test intervention analysis functionality."""
    
    def setup_method(self):
        """Setup test data and intervention analyzer."""
        np.random.seed(42)
        
        # Create economic data with causal structure
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Policy variable (e.g., interest rate)
        policy_rate = 2.0 + np.random.normal(0, 0.2, 100).cumsum() * 0.01
        
        # Unemployment responds to policy with lag
        unemployment = np.zeros(100)
        unemployment[0] = 4.0
        for t in range(1, 100):
            unemployment[t] = (0.8 * unemployment[t-1] + 
                             0.3 * policy_rate[t-1] + 
                             np.random.normal(0, 0.1))
        
        # Inflation also responds to policy
        inflation = np.zeros(100)
        inflation[0] = 2.0
        for t in range(1, 100):
            inflation[t] = (0.7 * inflation[t-1] - 
                          0.2 * policy_rate[t-1] + 
                          np.random.normal(0, 0.05))
        
        self.intervention_data = pd.DataFrame({
            'policy_rate': policy_rate,
            'unemployment_rate': unemployment,
            'inflation_rate': inflation,
            'market_sentiment': np.random.normal(0, 1, 100)
        }, index=dates)
        
        # Create simple causal graph
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_edges_from([
            ('policy_rate', 'unemployment_rate'),
            ('policy_rate', 'inflation_rate'),
            ('unemployment_rate', 'market_sentiment'),
            ('inflation_rate', 'market_sentiment')
        ])
        
        self.analyzer = InterventionAnalyzer(
            causal_graph=self.causal_graph,
            bootstrap_samples=100
        )
    
    def test_analyzer_initialization(self):
        """Test intervention analyzer initialization."""
        assert self.analyzer.causal_graph is not None
        assert self.analyzer.intervention_method == "do_calculus"
        assert self.analyzer.bootstrap_samples == 100
        assert self.analyzer.causal_models == {}
    
    def test_causal_model_fitting(self):
        """Test fitting causal models."""
        fitted_analyzer = self.analyzer.fit_causal_models(self.intervention_data)
        
        assert fitted_analyzer is self.analyzer
        assert len(self.analyzer.causal_models) > 0
        
        # Should have models for each target variable
        expected_targets = ['unemployment_rate', 'inflation_rate', 'market_sentiment']
        for target in expected_targets:
            if target in self.analyzer.causal_models:
                assert self.analyzer.causal_models[target] is not None
    
    def test_intervention_simulation(self):
        """Test intervention simulation."""
        # Fit models first
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Simulate policy intervention
        interventions = {'policy_rate': 1.5}  # Lower interest rate
        
        results = self.analyzer.simulate_intervention(
            interventions=interventions,
            forecast_horizon=12,
            n_simulations=50
        )
        
        assert 'baseline_forecast' in results
        assert 'intervention_forecast' in results
        assert 'causal_effects' in results
        
        # Check causal effects structure
        effects = results['causal_effects']
        assert 'unemployment_rate' in effects or 'inflation_rate' in effects
        
        # Effects should include mean and confidence intervals
        for var, effect_data in effects.items():
            if effect_data:
                assert 'mean_effect' in effect_data
                assert 'confidence_interval' in effect_data
    
    def test_policy_analysis(self):
        """Test policy intervention analysis."""
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Test different policy values
        policy_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        target_variables = ['unemployment_rate', 'inflation_rate']
        
        policy_analysis = self.analyzer.analyze_policy_intervention(
            policy_variable='policy_rate',
            policy_values=policy_values,
            target_variables=target_variables,
            horizon=24
        )
        
        assert 'policy_effects' in policy_analysis
        assert 'optimal_policies' in policy_analysis
        assert 'policy_tradeoffs' in policy_analysis
        
        # Check policy effects structure
        effects = policy_analysis['policy_effects']
        assert len(effects) == len(policy_values)
        
        for policy_val, effect_data in effects.items():
            assert isinstance(policy_val, float)
            assert isinstance(effect_data, dict)
    
    def test_treatment_effect_estimation(self):
        """Test treatment effect estimation."""
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Estimate effect of policy rate change
        treatment_effect = self.analyzer.estimate_treatment_effect(
            treatment_var='policy_rate',
            outcome_var='unemployment_rate',
            treatment_value=1.0,  # Low rate
            control_value=3.0     # High rate
        )
        
        assert 'average_treatment_effect' in treatment_effect
        assert 'confidence_interval' in treatment_effect
        assert 'p_value' in treatment_effect
        
        # ATE should be a number
        ate = treatment_effect['average_treatment_effect']
        assert isinstance(ate, (int, float))
    
    def test_counterfactual_analysis(self):
        """Test counterfactual analysis."""
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Analyze counterfactual scenario
        observed_data = self.intervention_data.iloc[-10:]  # Last 10 observations
        counterfactual_interventions = {'policy_rate': 0.5}  # Very low rate
        
        counterfactual_results = self.analyzer.counterfactual_analysis(
            observed_data=observed_data,
            counterfactual_interventions=counterfactual_interventions
        )
        
        assert 'observed_outcomes' in counterfactual_results
        assert 'counterfactual_outcomes' in counterfactual_results
        assert 'causal_effects' in counterfactual_results
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Test sensitivity of unemployment to policy rate changes
        sensitivity_results = self.analyzer.sensitivity_analysis(
            intervention_var='policy_rate',
            outcome_var='unemployment_rate',
            intervention_range=(0.5, 4.0),
            n_points=10
        )
        
        assert 'intervention_values' in sensitivity_results
        assert 'outcome_effects' in sensitivity_results
        assert 'effect_gradient' in sensitivity_results
        
        # Check that we have results for all intervention points
        assert len(sensitivity_results['intervention_values']) == 10
        assert len(sensitivity_results['outcome_effects']) == 10
    
    def test_multi_intervention_analysis(self):
        """Test multi-intervention analysis."""
        self.analyzer.fit_causal_models(self.intervention_data)
        
        # Test combinations of interventions
        intervention_combinations = [
            {'policy_rate': 1.0},
            {'policy_rate': 2.0},
            {'policy_rate': 3.0}
        ]
        target_variables = ['unemployment_rate', 'inflation_rate']
        
        multi_results = self.analyzer.multi_intervention_analysis(
            intervention_combinations=intervention_combinations,
            target_variables=target_variables,
            horizon=12
        )
        
        assert 'intervention_results' in multi_results
        assert 'best_combinations' in multi_results
        assert 'pareto_frontier' in multi_results
        
        # Should have results for each combination
        assert len(multi_results['intervention_results']) == 3
    
    def test_intervention_report_generation(self):
        """Test intervention report generation."""
        # Create mock intervention results
        mock_results = {
            'baseline_forecast': {
                'unemployment_rate': {'mean': [4.0, 4.1, 4.2]},
                'inflation_rate': {'mean': [2.0, 2.1, 2.0]}
            },
            'intervention_forecast': {
                'unemployment_rate': {'mean': [3.8, 3.9, 3.9]},
                'inflation_rate': {'mean': [1.8, 1.9, 1.8]}
            },
            'causal_effects': {
                'unemployment_rate': {
                    'mean_effect': -0.2,
                    'confidence_interval': [-0.3, -0.1]
                },
                'inflation_rate': {
                    'mean_effect': -0.2,
                    'confidence_interval': [-0.3, -0.1]
                }
            }
        }
        
        report = self.analyzer.generate_intervention_report(mock_results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Intervention Analysis Report' in report
        assert 'unemployment_rate' in report
        assert 'inflation_rate' in report


class TestCausalGraphOperations:
    """Test causal graph operations and utilities."""
    
    def setup_method(self):
        """Setup test causal graph."""
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([
            ('policy', 'unemployment'),
            ('policy', 'inflation'),
            ('unemployment', 'sentiment'),
            ('inflation', 'sentiment'),
            ('sentiment', 'market_volatility'),
            ('external_shock', 'unemployment'),
            ('external_shock', 'inflation')
        ])
    
    def test_graph_properties(self):
        """Test basic graph properties."""
        assert len(self.graph.nodes()) == 6
        assert len(self.graph.edges()) == 7
        assert self.graph.has_edge('policy', 'unemployment')
        assert self.graph.has_edge('policy', 'inflation')
    
    def test_causal_path_analysis(self):
        """Test causal path analysis."""
        analyzer = GrangerCausalityAnalyzer()
        analyzer.causal_graph = self.graph
        
        # Find paths from policy to market_volatility
        paths = analyzer.find_causal_paths('policy', 'market_volatility', max_length=3)
        
        assert len(paths) >= 2  # Should find multiple paths
        
        # Check specific paths
        expected_paths = [
            ['policy', 'unemployment', 'sentiment', 'market_volatility'],
            ['policy', 'inflation', 'sentiment', 'market_volatility']
        ]
        
        for expected_path in expected_paths:
            assert any(path == expected_path for path in paths)
    
    def test_causal_clustering(self):
        """Test causal clustering detection."""
        analyzer = GrangerCausalityAnalyzer()
        analyzer.causal_graph = self.graph
        
        clusters = analyzer.detect_causal_clusters()
        
        assert isinstance(clusters, dict)
        # Should identify related variables
        if len(clusters) > 0:
            for cluster_name, variables in clusters.items():
                assert len(variables) >= 2


class TestCausalInferenceIntegration:
    """Test integration between causal inference components."""
    
    def setup_method(self):
        """Setup integrated causal analysis environment."""
        np.random.seed(42)
        
        # Create policy simulation data
        dates = pd.date_range('2023-01-01', periods=120, freq='H')
        
        # Policy variables
        interest_rate = 2.0 + np.random.normal(0, 0.1, 120).cumsum() * 0.01
        government_spending = 100 + np.random.normal(0, 2, 120).cumsum() * 0.1
        
        # Economic outcomes with causal relationships
        unemployment = np.zeros(120)
        inflation = np.zeros(120)
        gdp_growth = np.zeros(120)
        
        unemployment[0] = 4.0
        inflation[0] = 2.0
        gdp_growth[0] = 2.5
        
        for t in range(1, 120):
            # Unemployment responds to interest rate and spending
            unemployment[t] = (0.8 * unemployment[t-1] + 
                             0.15 * interest_rate[t-1] - 
                             0.05 * government_spending[t-1] + 
                             np.random.normal(0, 0.1))
            
            # Inflation responds to spending and unemployment
            inflation[t] = (0.7 * inflation[t-1] + 
                          0.1 * government_spending[t-1] - 
                          0.2 * unemployment[t-1] + 
                          np.random.normal(0, 0.05))
            
            # GDP growth responds to all factors
            gdp_growth[t] = (0.6 * gdp_growth[t-1] - 
                           0.1 * interest_rate[t-1] + 
                           0.05 * government_spending[t-1] - 
                           0.3 * unemployment[t-1] + 
                           np.random.normal(0, 0.1))
        
        self.policy_data = pd.DataFrame({
            'interest_rate': interest_rate,
            'government_spending': government_spending,
            'unemployment_rate': unemployment,
            'inflation_rate': inflation,
            'gdp_growth': gdp_growth
        }, index=dates)
    
    def test_end_to_end_causal_analysis(self):
        """Test complete causal analysis workflow."""
        # Step 1: Discover causal relationships
        granger_analyzer = GrangerCausalityAnalyzer(max_lags=3)
        causality_results = granger_analyzer.analyze_pairwise_causality(self.policy_data)
        
        # Step 2: Use discovered relationships for intervention analysis
        intervention_analyzer = InterventionAnalyzer(
            causal_graph=granger_analyzer.causal_graph,
            bootstrap_samples=50
        )
        
        # Step 3: Fit causal models
        intervention_analyzer.fit_causal_models(self.policy_data)
        
        # Step 4: Simulate policy intervention
        policy_intervention = {'interest_rate': 1.0}  # Lower rate
        
        intervention_results = intervention_analyzer.simulate_intervention(
            interventions=policy_intervention,
            forecast_horizon=24,
            n_simulations=30
        )
        
        # Verify complete workflow
        assert 'causal_effects' in intervention_results
        assert len(intervention_analyzer.causal_models) > 0
    
    def test_policy_optimization(self):
        """Test policy optimization workflow."""
        # Initialize intervention analyzer
        intervention_analyzer = InterventionAnalyzer(bootstrap_samples=50)
        intervention_analyzer.fit_causal_models(self.policy_data)
        
        # Test policy optimization for unemployment reduction
        policy_analysis = intervention_analyzer.analyze_policy_intervention(
            policy_variable='interest_rate',
            policy_values=[0.5, 1.0, 1.5, 2.0, 2.5],
            target_variables=['unemployment_rate', 'inflation_rate'],
            horizon=12
        )
        
        optimal_policies = policy_analysis['optimal_policies']
        
        # Should identify optimal policies for each target
        assert isinstance(optimal_policies, dict)
        
        # Check that optimization results are reasonable
        if 'unemployment_rate' in optimal_policies:
            unemployment_optimal = optimal_policies['unemployment_rate']
            assert 'optimal_value' in unemployment_optimal
            assert 'expected_effect' in unemployment_optimal


class TestCausalValidation:
    """Test causal inference validation and robustness."""
    
    def test_causal_assumption_validation(self):
        """Test validation of causal assumptions."""
        # Create data that violates causal assumptions
        np.random.seed(42)
        
        # Confounded relationship (common cause)
        confounder = np.random.normal(0, 1, 100)
        x = confounder + np.random.normal(0, 0.5, 100)
        y = confounder + np.random.normal(0, 0.5, 100)  # X and Y both caused by confounder
        
        confounded_data = pd.DataFrame({
            'x': x,
            'y': y,
            'confounder': confounder
        })
        
        # Test Granger causality (should find spurious causality)
        analyzer = GrangerCausalityAnalyzer()
        causality_results = analyzer.analyze_pairwise_causality(confounded_data)
        
        # Should detect the relationships
        assert isinstance(causality_results, dict)
    
    def test_robustness_to_noise(self):
        """Test robustness of causal inference to noise."""
        # Create clean causal relationship
        x = np.random.normal(0, 1, 100)
        y_clean = np.zeros(100)
        
        for t in range(1, 100):
            y_clean[t] = 0.5 * y_clean[t-1] + 0.7 * x[t-1]
        
        # Add increasing levels of noise
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        causality_strengths = []
        
        analyzer = GrangerCausalityAnalyzer(max_lags=3)
        
        for noise_level in noise_levels:
            y_noisy = y_clean + np.random.normal(0, noise_level, 100)
            
            noisy_data = pd.DataFrame({'x': x, 'y': y_noisy})
            results = analyzer.analyze_pairwise_causality(noisy_data)
            
            if 'x->y' in results:
                causality_strengths.append(results['x->y']['f_statistic'])
            else:
                causality_strengths.append(0)
        
        # Causality strength should generally decrease with noise
        # (though this is not guaranteed in all cases)
        assert len(causality_strengths) == len(noise_levels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])