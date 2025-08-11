"""Granger causality analysis for detecting causal relationships in time series."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import networkx as nx

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class GrangerCausalityAnalyzer:
    """Analyzer for Granger causality relationships in socio-economic data."""
    
    def __init__(self, 
                 max_lags: int = 12,
                 significance_level: float = 0.05,
                 ic_criterion: str = "aic"):
        
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.ic_criterion = ic_criterion
        
        # Results storage
        self.causality_matrix = None
        self.causal_graph = None
        self.var_model = None
        
    def analyze_pairwise_causality(self, data: pd.DataFrame, 
                                 variables: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Analyze pairwise Granger causality between variables."""
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove variables with insufficient data
        valid_variables = []
        for var in variables:
            if var in data.columns:
                series = data[var].dropna()
                if len(series) >= 3 * self.max_lags:  # Minimum data requirement
                    valid_variables.append(var)
        
        if len(valid_variables) < 2:
            logger.warning("Insufficient variables for Granger causality analysis")
            return {}
        
        logger.info(f"Analyzing Granger causality for {len(valid_variables)} variables")
        
        causality_results = {}
        
        # Test all pairs
        for i, cause_var in enumerate(valid_variables):
            causality_results[cause_var] = {}
            
            for effect_var in valid_variables:
                if cause_var == effect_var:
                    continue
                
                try:
                    # Prepare data for Granger test
                    test_data = data[[effect_var, cause_var]].dropna()
                    
                    if len(test_data) < 3 * self.max_lags:
                        continue
                    
                    # Perform Granger causality test
                    gc_result = self._granger_test(test_data, cause_var, effect_var)
                    causality_results[cause_var][effect_var] = gc_result
                    
                except Exception as e:
                    logger.error(f"Error testing {cause_var} -> {effect_var}: {str(e)}")
                    continue
        
        # Build causality matrix
        self.causality_matrix = self._build_causality_matrix(causality_results, valid_variables)
        
        # Build causal graph
        self.causal_graph = self._build_causal_graph(causality_results)
        
        logger.info(f"Granger causality analysis completed for {len(valid_variables)} variables")
        return causality_results
    
    def _granger_test(self, data: pd.DataFrame, cause_var: str, effect_var: str) -> Dict[str, Any]:
        """Perform Granger causality test between two variables."""
        # Prepare data (effect variable first, then cause variable)
        test_data = data[[effect_var, cause_var]].values
        
        # Determine optimal lag using information criteria
        optimal_lag = self._find_optimal_lag(data[[effect_var, cause_var]])
        
        # Perform Granger causality test
        gc_test = grangercausalitytests(test_data, maxlag=optimal_lag, verbose=False)
        
        # Extract results for optimal lag
        test_result = gc_test[optimal_lag][0]
        
        # Get F-statistic and p-value
        f_stat = test_result['ssr_ftest'][0]
        p_value = test_result['ssr_ftest'][1]
        
        # Calculate effect size (partial correlation)
        effect_size = self._calculate_effect_size(data, cause_var, effect_var, optimal_lag)
        
        result = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'optimal_lag': optimal_lag,
            'effect_size': effect_size,
            'direction': f"{cause_var} -> {effect_var}",
            'strength': self._classify_causal_strength(f_stat, p_value)
        }
        
        return result
    
    def _find_optimal_lag(self, data: pd.DataFrame) -> int:
        """Find optimal lag length using information criteria."""
        try:
            # Fit VAR model to determine optimal lag
            var_model = VAR(data.dropna())
            lag_order = var_model.select_order(maxlags=self.max_lags)
            
            if self.ic_criterion == "aic":
                optimal_lag = lag_order.aic
            elif self.ic_criterion == "bic":
                optimal_lag = lag_order.bic
            elif self.ic_criterion == "hqic":
                optimal_lag = lag_order.hqic
            else:
                optimal_lag = lag_order.aic
            
            return max(1, min(optimal_lag, self.max_lags))
            
        except Exception:
            # Default to lag 1 if optimization fails
            return 1
    
    def _calculate_effect_size(self, data: pd.DataFrame, cause_var: str, 
                             effect_var: str, lag: int) -> float:
        """Calculate effect size for Granger causality."""
        try:
            # Create lagged variables
            effect_series = data[effect_var].dropna()
            cause_lagged = data[cause_var].shift(lag).dropna()
            
            # Align series
            common_index = effect_series.index.intersection(cause_lagged.index)
            if len(common_index) < 10:
                return 0.0
            
            effect_aligned = effect_series.loc[common_index]
            cause_aligned = cause_lagged.loc[common_index]
            
            # Calculate partial correlation (controlling for past values of effect)
            if len(effect_aligned) > lag:
                effect_lagged = effect_aligned.shift(1).dropna()
                common_index2 = effect_lagged.index.intersection(cause_aligned.index).intersection(effect_aligned.index)
                
                if len(common_index2) > 10:
                    # Partial correlation coefficient
                    corr_matrix = pd.DataFrame({
                        'effect': effect_aligned.loc[common_index2],
                        'cause': cause_aligned.loc[common_index2],
                        'effect_lag': effect_lagged.loc[common_index2]
                    }).corr()
                    
                    # Calculate partial correlation
                    r_xy = corr_matrix.loc['effect', 'cause']
                    r_xz = corr_matrix.loc['effect', 'effect_lag']
                    r_yz = corr_matrix.loc['cause', 'effect_lag']
                    
                    partial_corr = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
                    return abs(partial_corr)
            
            # Fallback to simple correlation
            correlation = np.corrcoef(effect_aligned, cause_aligned)[0, 1]
            return abs(correlation)
            
        except Exception:
            return 0.0
    
    def _classify_causal_strength(self, f_stat: float, p_value: float) -> str:
        """Classify the strength of causal relationship."""
        if p_value >= self.significance_level:
            return "none"
        elif f_stat >= 10:
            return "strong"
        elif f_stat >= 5:
            return "moderate"
        else:
            return "weak"
    
    def _build_causality_matrix(self, causality_results: Dict[str, Dict[str, Any]], 
                              variables: List[str]) -> pd.DataFrame:
        """Build causality matrix from pairwise results."""
        n_vars = len(variables)
        matrix = np.zeros((n_vars, n_vars))
        
        for i, cause_var in enumerate(variables):
            for j, effect_var in enumerate(variables):
                if cause_var in causality_results and effect_var in causality_results[cause_var]:
                    result = causality_results[cause_var][effect_var]
                    # Use F-statistic as strength measure
                    matrix[i, j] = result['f_statistic'] if result['significant'] else 0
        
        causality_df = pd.DataFrame(matrix, index=variables, columns=variables)
        return causality_df
    
    def _build_causal_graph(self, causality_results: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
        """Build directed causal graph from Granger causality results."""
        graph = nx.DiGraph()
        
        # Add nodes
        all_variables = set()
        for cause_var, effects in causality_results.items():
            all_variables.add(cause_var)
            all_variables.update(effects.keys())
        
        graph.add_nodes_from(all_variables)
        
        # Add edges for significant causal relationships
        for cause_var, effects in causality_results.items():
            for effect_var, result in effects.items():
                if result['significant']:
                    graph.add_edge(
                        cause_var, 
                        effect_var,
                        weight=result['f_statistic'],
                        p_value=result['p_value'],
                        lag=result['optimal_lag'],
                        strength=result['strength']
                    )
        
        return graph
    
    def analyze_var_model(self, data: pd.DataFrame, 
                         variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze Vector Autoregression model for causal relationships."""
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select variables with sufficient data
        valid_data = data[variables].dropna()
        
        if len(valid_data) < 3 * self.max_lags:
            logger.warning("Insufficient data for VAR analysis")
            return {}
        
        try:
            # Fit VAR model
            var_model = VAR(valid_data)
            
            # Select optimal lag order
            lag_order = var_model.select_order(maxlags=self.max_lags)
            optimal_lag = getattr(lag_order, self.ic_criterion)
            
            # Fit VAR with optimal lag
            var_fitted = var_model.fit(optimal_lag)
            self.var_model = var_fitted
            
            # Granger causality tests
            causality_tests = var_fitted.test_causality(causing=variables, caused=variables)
            
            # Impulse response analysis
            irf = var_fitted.irf(periods=24)  # 24-step impulse responses
            
            # Variance decomposition
            fevd = var_fitted.fevd(periods=24)
            
            var_results = {
                'optimal_lag': optimal_lag,
                'aic': var_fitted.aic,
                'bic': var_fitted.bic,
                'causality_tests': self._parse_causality_tests(causality_tests),
                'impulse_responses': self._parse_irf(irf, variables),
                'variance_decomposition': self._parse_fevd(fevd, variables),
                'model_summary': str(var_fitted.summary())
            }
            
            logger.info(f"VAR analysis completed with lag order {optimal_lag}")
            return var_results
            
        except Exception as e:
            logger.error(f"Error in VAR analysis: {str(e)}")
            return {}
    
    def _parse_causality_tests(self, causality_tests) -> Dict[str, Any]:
        """Parse VAR causality test results."""
        try:
            return {
                'test_statistic': float(causality_tests.test_statistic),
                'p_value': float(causality_tests.pvalue),
                'critical_value': float(causality_tests.critical_value),
                'significant': float(causality_tests.pvalue) < self.significance_level
            }
        except Exception:
            return {}
    
    def _parse_irf(self, irf, variables: List[str]) -> Dict[str, Any]:
        """Parse impulse response function results."""
        try:
            irf_data = {}
            for i, response_var in enumerate(variables):
                irf_data[response_var] = {}
                for j, shock_var in enumerate(variables):
                    irf_data[response_var][shock_var] = irf.irfs[:, i, j].tolist()
            
            return irf_data
        except Exception:
            return {}
    
    def _parse_fevd(self, fevd, variables: List[str]) -> Dict[str, Any]:
        """Parse forecast error variance decomposition results."""
        try:
            fevd_data = {}
            for i, var in enumerate(variables):
                fevd_data[var] = {}
                for j, shock_var in enumerate(variables):
                    fevd_data[var][shock_var] = fevd.decomp[:, i, j].tolist()
            
            return fevd_data
        except Exception:
            return {}
    
    def get_causal_network_metrics(self) -> Dict[str, Any]:
        """Calculate network metrics for the causal graph."""
        if self.causal_graph is None:
            return {}
        
        try:
            metrics = {
                'n_nodes': self.causal_graph.number_of_nodes(),
                'n_edges': self.causal_graph.number_of_edges(),
                'density': nx.density(self.causal_graph),
                'is_dag': nx.is_directed_acyclic_graph(self.causal_graph)
            }
            
            # Node-level metrics
            in_degrees = dict(self.causal_graph.in_degree())
            out_degrees = dict(self.causal_graph.out_degree())
            
            metrics['node_metrics'] = {
                'in_degrees': in_degrees,
                'out_degrees': out_degrees,
                'most_influential': max(out_degrees.items(), key=lambda x: x[1])[0] if out_degrees else None,
                'most_influenced': max(in_degrees.items(), key=lambda x: x[1])[0] if in_degrees else None
            }
            
            # Calculate centrality measures
            if self.causal_graph.number_of_edges() > 0:
                betweenness = nx.betweenness_centrality(self.causal_graph)
                closeness = nx.closeness_centrality(self.causal_graph)
                
                metrics['centrality'] = {
                    'betweenness': betweenness,
                    'closeness': closeness
                }
            
            logger.info("Causal network metrics calculated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {str(e)}")
    
    def find_causal_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find causal paths between two variables."""
        if self.causal_graph is None:
            return []
        
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.causal_graph, 
                source=source, 
                target=target, 
                cutoff=max_length
            ))
            
            # Sort by path length and strength
            path_strengths = []
            for path in paths:
                strength = 1.0
                for i in range(len(path) - 1):
                    edge_data = self.causal_graph.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        strength *= edge_data.get('weight', 1.0)
                path_strengths.append((path, strength))
            
            # Sort by strength
            path_strengths.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(paths)} causal paths from {source} to {target}")
            return [path for path, _ in path_strengths]
            
        except Exception as e:
            logger.error(f"Error finding causal paths: {str(e)}")
    
    def detect_causal_clusters(self) -> Dict[str, List[str]]:
        """Detect clusters of causally related variables."""
        if self.causal_graph is None:
            return {}
        
        try:
            # Convert to undirected graph for clustering
            undirected_graph = self.causal_graph.to_undirected()
            
            # Find connected components
            clusters = list(nx.connected_components(undirected_graph))
            
            cluster_dict = {}
            for i, cluster in enumerate(clusters):
                cluster_dict[f"cluster_{i}"] = list(cluster)
            
            logger.info(f"Detected {len(cluster_dict)} causal clusters")
            return cluster_dict
            
        except Exception as e:
            logger.error(f"Error detecting causal clusters: {str(e)}")
    
    def calculate_instantaneous_causality(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate instantaneous causality using VAR model."""
        if self.var_model is None:
            logger.warning("VAR model not fitted - run analyze_var_model first")
            return {}
        
        try:
            # Test for instantaneous causality
            inst_causality = {}
            
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for var in variables:
                if var in self.var_model.names:
                    # Test instantaneous causality for this variable
                    test_result = self.var_model.test_inst_causality(causing=var)
                    inst_causality[var] = {
                        'test_statistic': float(test_result.test_statistic),
                        'p_value': float(test_result.pvalue),
                        'significant': float(test_result.pvalue) < self.significance_level
                    }
            
            logger.info(f"Instantaneous causality calculated for {len(inst_causality)} variables")
            return inst_causality
            
        except Exception as e:
            logger.error(f"Error calculating instantaneous causality: {str(e)}")
    
    def get_causal_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive causal analysis report."""
        if self.causality_matrix is None:
            return {}
        
        report = {
            'summary_statistics': {
                'total_relationships_tested': self.causality_matrix.size,
                'significant_relationships': (self.causality_matrix > 0).sum().sum(),
                'average_causal_strength': self.causality_matrix[self.causality_matrix > 0].mean()
            },
            'strongest_relationships': self._get_strongest_relationships(),
            'causal_hubs': self._identify_causal_hubs(),
            'network_metrics': self.get_causal_network_metrics(),
            'clusters': self.detect_causal_clusters()
        }
        
        return report
    
    def _get_strongest_relationships(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the strongest causal relationships."""
        if self.causal_graph is None:
            return []
        
        # Get all edges with their weights
        edges_with_weights = []
        for source, target, data in self.causal_graph.edges(data=True):
            edges_with_weights.append({
                'source': source,
                'target': target,
                'strength': data.get('weight', 0),
                'p_value': data.get('p_value', 1),
                'lag': data.get('lag', 1),
                'classification': data.get('strength', 'unknown')
            })
        
        # Sort by strength
        edges_with_weights.sort(key=lambda x: x['strength'], reverse=True)
        
        return edges_with_weights[:top_k]
    
    def _identify_causal_hubs(self) -> Dict[str, Any]:
        """Identify variables that are causal hubs (high influence or high susceptibility)."""
        if self.causal_graph is None:
            return {}
        
        # Calculate in-degree and out-degree
        in_degrees = dict(self.causal_graph.in_degree())
        out_degrees = dict(self.causal_graph.out_degree())
        
        # Identify hubs
        hubs = {
            'causal_drivers': sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5],
            'causal_receivers': sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5],
            'bidirectional_hubs': []
        }
        
        # Find variables with both high in-degree and out-degree
        for var in self.causal_graph.nodes():
            if in_degrees.get(var, 0) >= 2 and out_degrees.get(var, 0) >= 2:
                hubs['bidirectional_hubs'].append((var, in_degrees[var] + out_degrees[var]))
        
        hubs['bidirectional_hubs'].sort(key=lambda x: x[1], reverse=True)
        
        return hubs
    
    def test_causal_stability(self, data: pd.DataFrame, 
                            window_size: int = 100, 
                            step_size: int = 50) -> Dict[str, Any]:
        """Test stability of causal relationships over time."""
        if len(data) < 2 * window_size:
            logger.warning("Insufficient data for causal stability analysis")
            return {}
        
        stability_results = {}
        variables = data.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit for performance
        
        # Sliding window analysis
        window_results = []
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            
            # Analyze causality in this window
            window_causality = self.analyze_pairwise_causality(window_data, variables)
            
            # Extract significant relationships
            significant_pairs = []
            for cause_var, effects in window_causality.items():
                for effect_var, result in effects.items():
                    if result['significant']:
                        significant_pairs.append((cause_var, effect_var))
            
            window_results.append({
                'start_time': data.index[start_idx],
                'end_time': data.index[end_idx-1],
                'significant_pairs': significant_pairs,
                'n_relationships': len(significant_pairs)
            })
        
        # Analyze stability
        all_pairs = set()
        for window in window_results:
            all_pairs.update(window['significant_pairs'])
        
        pair_stability = {}
        for pair in all_pairs:
            appearances = sum(1 for window in window_results if pair in window['significant_pairs'])
            stability = appearances / len(window_results)
            pair_stability[f"{pair[0]} -> {pair[1]}"] = stability
        
        stability_results = {
            'window_results': window_results,
            'pair_stability': pair_stability,
            'most_stable_relationships': sorted(pair_stability.items(), key=lambda x: x[1], reverse=True)[:10],
            'average_stability': np.mean(list(pair_stability.values())) if pair_stability else 0
        }
        
        logger.info(f"Causal stability analysis completed over {len(window_results)} windows")
        return stability_results
    
    def export_causal_graph(self, filepath: str, format: str = "gexf") -> None:
        """Export causal graph to file."""
        if self.causal_graph is None:
            logger.warning("No causal graph to export")
            return
        
        try:
            if format == "gexf":
                nx.write_gexf(self.causal_graph, filepath)
            elif format == "graphml":
                nx.write_graphml(self.causal_graph, filepath)
            elif format == "edgelist":
                nx.write_edgelist(self.causal_graph, filepath)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"Causal graph exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting causal graph: {str(e)}")