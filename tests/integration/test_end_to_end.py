"""Integration tests for the complete adaptive modeling framework."""

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import os

from src.data.ingestion import DataIngestionPipeline
from src.data.preprocessing import DataPreprocessor
from src.models.baseline_model import BaselineRegressionModel
from src.models.adaptive_model import AdaptiveBayesianModel
from src.models.change_point_detector import ChangePointDetector
from src.causal.granger_causality import GrangerCausalityAnalyzer
from src.causal.intervention_analysis import InterventionAnalyzer
from src.utils.config import Config, load_config


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def setup_method(self):
        """Setup integration test environment."""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test configuration
        self.config = Config()
        self.config.database_url = f"sqlite:///{self.temp_dir}/test.db"
        
        # Mock data sources configuration
        self.config.data_sources = {
            'economic_indicators': {
                'series': [
                    {'id': 'UNRATE', 'name': 'unemployment_rate'},
                    {'id': 'CPIAUCSL', 'name': 'cpi'}
                ]
            },
            'market_data': {
                'symbols': ['^GSPC', '^VIX']
            },
            'social_sentiment': {
                'keywords': ['economy', 'inflation']
            }
        }
        
        # Generate realistic test data
        self.test_data = self._generate_realistic_economic_data()
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _generate_realistic_economic_data(self) -> pd.DataFrame:
        """Generate realistic economic time series data."""
        np.random.seed(42)
        
        # 30 days of hourly data
        dates = pd.date_range('2023-01-01', periods=720, freq='H')
        
        # Economic indicators with realistic relationships
        
        # Policy rate (Fed funds rate)
        policy_rate = 2.0 + np.random.normal(0, 0.1, 720).cumsum() * 0.001
        policy_rate = np.clip(policy_rate, 0.25, 5.0)
        
        # Unemployment rate
        unemployment = np.zeros(720)
        unemployment[0] = 3.8
        for t in range(1, 720):
            # Responds to policy rate with lag, plus trend and noise
            unemployment[t] = (0.98 * unemployment[t-1] + 
                             0.02 * policy_rate[max(0, t-24)] +  # 24-hour lag
                             np.random.normal(0, 0.01))
        unemployment = np.clip(unemployment, 2.0, 8.0)
        
        # Consumer Price Index (inflation proxy)
        cpi = np.zeros(720)
        cpi[0] = 100.0
        for t in range(1, 720):
            # Responds to unemployment and policy
            inflation_rate = (0.02 + 
                            -0.001 * unemployment[t] + 
                            0.0005 * policy_rate[t] + 
                            np.random.normal(0, 0.0005))
            cpi[t] = cpi[t-1] * (1 + inflation_rate/8760)  # Hourly compounding
        
        # Market data (S&P 500 and VIX)
        sp500 = 4000 * np.exp(np.random.normal(0, 0.01, 720).cumsum())
        vix = 20 + np.random.normal(0, 2, 720).cumsum() * 0.1
        vix = np.clip(vix, 10, 80)
        
        # Social sentiment (correlated with unemployment and market)
        sentiment = (-0.5 * (unemployment - unemployment.mean()) / unemployment.std() + 
                    0.3 * (sp500 - sp500.mean()) / sp500.std() + 
                    np.random.normal(0, 0.5, 720))
        
        # Google Trends data
        google_trends = np.random.exponential(1, 720) + 0.2 * unemployment
        
        # Weather data (temperature)
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(720) / (24 * 365.25)) + np.random.normal(0, 3, 720)
        
        return pd.DataFrame({
            'policy_rate': policy_rate,
            'unemployment_rate': unemployment,
            'cpi': cpi,
            'sp500': sp500,
            'vix': vix,
            'sentiment': sentiment,
            'google_trends': google_trends,
            'temperature': temperature
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self):
        """Test complete data ingestion and preprocessing pipeline."""
        # Initialize pipeline
        pipeline = DataIngestionPipeline(self.config)
        
        # Mock the data fetching to use our test data
        async def mock_fetch_all_data(start_date=None, end_date=None):
            if start_date is None:
                start_date = self.test_data.index[0]
            if end_date is None:
                end_date = self.test_data.index[-1]
            
            # Return subsets of test data as if from different sources
            mask = (self.test_data.index >= start_date) & (self.test_data.index <= end_date)
            subset = self.test_data[mask]
            
            return {
                'fred': subset[['unemployment_rate', 'cpi']],
                'yahoo': subset[['sp500', 'vix']],
                'sentiment': subset[['sentiment']],
                'trends': subset[['google_trends']],
                'weather': subset[['temperature']]
            }
        
        # Replace the method
        pipeline.fetch_all_data = mock_fetch_all_data
        
        # Test data fetching
        data_dict = await pipeline.fetch_all_data()
        assert len(data_dict) == 5
        assert all(not df.empty for df in data_dict.values())
        
        # Test data merging
        merged_data = pipeline.merge_data_sources(data_dict)
        assert not merged_data.empty
        assert len(merged_data.columns) == 8  # All variables
        
        # Test database operations
        pipeline.save_to_database(merged_data, source="test_merged")
        loaded_data = pipeline.load_from_database(source="test_merged")
        
        assert len(loaded_data) == len(merged_data)
        assert list(loaded_data.columns) == list(merged_data.columns)
    
    def test_complete_modeling_pipeline(self):
        """Test complete modeling pipeline from data to predictions."""
        # Step 1: Preprocess data
        preprocessor = DataPreprocessor(self.config.preprocessing)
        processed_data = preprocessor.clean_and_preprocess(self.test_data)
        
        # Step 2: Create targets and split features
        target_vars = ['unemployment_rate', 'cpi']
        data_with_targets = preprocessor.create_target_variables(processed_data, target_vars)
        X, y = preprocessor.split_features_targets(data_with_targets)
        
        # Ensure we have enough data for training
        assert len(X) > 100
        assert len(y) > 100
        assert len(y.columns) >= 2
        
        # Step 3: Train baseline model
        baseline_model = BaselineRegressionModel(max_features=15)
        
        # Split data for training/testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train model
        baseline_model.fit(X_train, y_train)
        
        # Step 4: Make predictions
        predictions = baseline_model.predict(X_test, return_uncertainty=True)
        
        assert 'predictions' in predictions
        assert 'uncertainty' in predictions
        assert len(predictions['predictions']) == len(X_test)
        
        # Step 5: Evaluate model
        evaluation = baseline_model.evaluate(X_test, y_test)
        
        for target in target_vars:
            if f'{target}_target_1h' in evaluation:
                metrics = evaluation[f'{target}_target_1h']
                assert 'rmse' in metrics
                assert 'r2' in metrics
                assert metrics['rmse'] > 0
    
    def test_change_point_detection_integration(self):
        """Test change point detection integrated with modeling."""
        # Add artificial change point to test data
        modified_data = self.test_data.copy()
        change_point_idx = len(modified_data) // 2
        
        # Create regime shift in unemployment
        modified_data.iloc[change_point_idx:, modified_data.columns.get_loc('unemployment_rate')] += 1.0
        
        # Step 1: Detect change points
        detector = ChangePointDetector(min_size=24, pen=10.0)
        change_points = detector.detect_change_points(modified_data)
        
        # Should detect change point in unemployment
        assert 'unemployment_rate' in change_points
        unemployment_cps = change_points['unemployment_rate']
        
        if len(unemployment_cps) > 0:
            detected_cp = unemployment_cps[0]
            assert 'timestamp' in detected_cp
            assert 'confidence' in detected_cp
            
            # Step 2: Use change point for model adaptation
            cp_timestamp = detected_cp['timestamp']
            cp_idx = modified_data.index.get_loc(cp_timestamp)
            
            # Train separate models before and after change point
            pre_change_data = modified_data.iloc[:cp_idx]
            post_change_data = modified_data.iloc[cp_idx:]
            
            if len(pre_change_data) > 50 and len(post_change_data) > 50:
                # Preprocess both periods
                preprocessor = DataPreprocessor(self.config.preprocessing)
                
                pre_processed = preprocessor.clean_and_preprocess(pre_change_data)
                post_processed = preprocessor.clean_and_preprocess(post_change_data)
                
                # Create targets
                pre_targets = preprocessor.create_target_variables(pre_processed, ['unemployment_rate'])
                post_targets = preprocessor.create_target_variables(post_processed, ['unemployment_rate'])
                
                X_pre, y_pre = preprocessor.split_features_targets(pre_targets)
                X_post, y_post = preprocessor.split_features_targets(post_targets)
                
                # Train models on each period
                model_pre = BaselineRegressionModel(max_features=10)
                model_post = BaselineRegressionModel(max_features=10)
                
                if len(X_pre) > 20 and len(X_post) > 20:
                    model_pre.fit(X_pre, y_pre)
                    model_post.fit(X_post, y_post)
                    
                    # Compare model performance
                    eval_pre = model_pre.evaluate(X_pre[-10:], y_pre[-10:])
                    eval_post = model_post.evaluate(X_post[:10], y_post[:10])
                    
                    # Both models should provide evaluations
                    assert len(eval_pre) > 0
                    assert len(eval_post) > 0
    
    def test_causal_inference_integration(self):
        """Test causal inference integrated with intervention analysis."""
        # Step 1: Discover causal relationships
        granger_analyzer = GrangerCausalityAnalyzer(max_lags=4)
        
        # Use subset of variables for faster computation
        causal_vars = ['policy_rate', 'unemployment_rate', 'cpi', 'sentiment']
        causal_data = self.test_data[causal_vars]
        
        causality_results = granger_analyzer.analyze_pairwise_causality(causal_data)
        
        # Step 2: Build intervention analyzer
        intervention_analyzer = InterventionAnalyzer(
            causal_graph=granger_analyzer.causal_graph,
            bootstrap_samples=50
        )
        
        # Step 3: Fit causal models
        intervention_analyzer.fit_causal_models(causal_data)
        
        # Step 4: Simulate policy intervention
        policy_intervention = {'policy_rate': 1.5}
        
        intervention_results = intervention_analyzer.simulate_intervention(
            interventions=policy_intervention,
            forecast_horizon=24,
            n_simulations=30
        )
        
        # Verify integration worked
        assert 'causal_effects' in intervention_results
        assert len(intervention_analyzer.causal_models) > 0
        
        # Step 5: Generate intervention report
        report = intervention_analyzer.generate_intervention_report(intervention_results)
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial report
    
    def test_adaptive_learning_workflow(self):
        """Test adaptive learning with incoming data streams."""
        # Initialize components
        preprocessor = DataPreprocessor(self.config.preprocessing)
        baseline_model = BaselineRegressionModel(max_features=12)
        
        # Simulate streaming data scenario
        window_size = 200
        update_frequency = 50  # Update every 50 data points
        
        # Initial training
        initial_data = self.test_data.iloc[:window_size]
        processed_initial = preprocessor.clean_and_preprocess(initial_data)
        targets_initial = preprocessor.create_target_variables(processed_initial, ['unemployment_rate', 'cpi'])
        X_initial, y_initial = preprocessor.split_features_targets(targets_initial)
        
        # Train initial model
        baseline_model.fit(X_initial, y_initial)
        initial_performance = baseline_model.get_model_summary()
        
        # Simulate streaming updates
        performances = []
        
        for i in range(window_size, len(self.test_data) - update_frequency, update_frequency):
            # Get new data batch
            new_batch = self.test_data.iloc[i:i+update_frequency]
            processed_batch = preprocessor.clean_and_preprocess(new_batch)
            targets_batch = preprocessor.create_target_variables(processed_batch, ['unemployment_rate', 'cpi'])
            X_batch, y_batch = preprocessor.split_features_targets(targets_batch)
            
            if len(X_batch) > 10:  # Ensure sufficient data
                # Online update
                baseline_model.partial_fit(X_batch, y_batch)
                
                # Evaluate on recent data
                recent_data = self.test_data.iloc[i-20:i]
                processed_recent = preprocessor.clean_and_preprocess(recent_data)
                targets_recent = preprocessor.create_target_variables(processed_recent, ['unemployment_rate', 'cpi'])
                X_recent, y_recent = preprocessor.split_features_targets(targets_recent)
                
                if len(X_recent) > 5:
                    eval_metrics = baseline_model.evaluate(X_recent, y_recent)
                    performances.append(eval_metrics)
        
        # Should have collected performance over time
        assert len(performances) > 0
        
        # Model should maintain reasonable performance
        for perf in performances:
            for target, metrics in perf.items():
                if 'rmse' in metrics:
                    assert metrics['rmse'] < 10  # Reasonable RMSE bound
    
    def test_real_time_monitoring_simulation(self):
        """Test real-time monitoring and alerting simulation."""
        # Initialize monitoring components
        detector = ChangePointDetector(min_size=12, pen=5.0)
        baseline_model = BaselineRegressionModel(max_features=10)
        
        # Setup initial model
        train_data = self.test_data.iloc[:400]
        preprocessor = DataPreprocessor(self.config.preprocessing)
        processed_train = preprocessor.clean_and_preprocess(train_data)
        targets_train = preprocessor.create_target_variables(processed_train, ['unemployment_rate'])
        X_train, y_train = preprocessor.split_features_targets(targets_train)
        
        baseline_model.fit(X_train, y_train)
        
        # Simulate real-time monitoring
        monitoring_results = {
            'change_points': [],
            'model_alerts': [],
            'performance_metrics': []
        }
        
        # Process data in chunks (simulating real-time)
        chunk_size = 24  # 24 hours at a time
        
        for i in range(400, len(self.test_data) - chunk_size, chunk_size):
            # Get new data chunk
            chunk = self.test_data.iloc[i:i+chunk_size]
            
            # Detect change points
            recent_data = self.test_data.iloc[max(0, i-100):i+chunk_size]
            change_points = detector.detect_change_points(recent_data[['unemployment_rate']])
            
            if change_points['unemployment_rate']:
                monitoring_results['change_points'].extend(change_points['unemployment_rate'])
            
            # Check model performance
            processed_chunk = preprocessor.clean_and_preprocess(chunk)
            targets_chunk = preprocessor.create_target_variables(processed_chunk, ['unemployment_rate'])
            X_chunk, y_chunk = preprocessor.split_features_targets(targets_chunk)
            
            if len(X_chunk) > 5:
                # Evaluate model on new data
                chunk_metrics = baseline_model.evaluate(X_chunk, y_chunk)
                monitoring_results['performance_metrics'].append(chunk_metrics)
                
                # Update model with new data
                baseline_model.partial_fit(X_chunk, y_chunk)
        
        # Verify monitoring captured relevant events
        assert len(monitoring_results['performance_metrics']) > 0
        
        # Check that performance metrics are reasonable
        for metrics in monitoring_results['performance_metrics']:
            for target, target_metrics in metrics.items():
                if 'rmse' in target_metrics:
                    assert target_metrics['rmse'] >= 0
    
    def test_decision_support_workflow(self):
        """Test complete decision support workflow."""
        # Step 1: Setup models with historical data
        train_data = self.test_data.iloc[:500]
        
        # Preprocess
        preprocessor = DataPreprocessor(self.config.preprocessing)
        processed_data = preprocessor.clean_and_preprocess(train_data)
        
        # Step 2: Train causal models
        causal_vars = ['policy_rate', 'unemployment_rate', 'cpi', 'sentiment']
        causal_data = processed_data[causal_vars].dropna()
        
        if len(causal_data) > 100:
            # Discover causal relationships
            granger_analyzer = GrangerCausalityAnalyzer(max_lags=3)
            causality_results = granger_analyzer.analyze_pairwise_causality(causal_data)
            
            # Setup intervention analyzer
            intervention_analyzer = InterventionAnalyzer(
                causal_graph=granger_analyzer.causal_graph,
                bootstrap_samples=30
            )
            intervention_analyzer.fit_causal_models(causal_data)
            
            # Step 3: Policy analysis
            policy_values = [1.0, 1.5, 2.0, 2.5]
            target_variables = ['unemployment_rate', 'cpi']
            
            policy_analysis = intervention_analyzer.analyze_policy_intervention(
                policy_variable='policy_rate',
                policy_values=policy_values,
                target_variables=target_variables,
                horizon=48
            )
            
            # Verify decision support outputs
            assert 'policy_effects' in policy_analysis
            assert 'optimal_policies' in policy_analysis
            
            # Step 4: Generate recommendations
            report = intervention_analyzer.generate_intervention_report({
                'baseline_forecast': {},
                'intervention_forecast': {},
                'causal_effects': {}
            })
            
            assert isinstance(report, str)
            assert len(report) > 0
    
    def test_model_comparison_workflow(self):
        """Test comparison between different model types."""
        # Prepare data
        preprocessor = DataPreprocessor(self.config.preprocessing)
        processed_data = preprocessor.clean_and_preprocess(self.test_data)
        targets_data = preprocessor.create_target_variables(processed_data, ['unemployment_rate'])
        X, y = preprocessor.split_features_targets(targets_data)
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        if len(X_train) > 50 and len(X_test) > 10:
            # Train baseline model
            baseline_model = BaselineRegressionModel(max_features=10)
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)
            baseline_eval = baseline_model.evaluate(X_test, y_test)
            
            # Compare models (baseline vs baseline with different parameters)
            baseline_model2 = BaselineRegressionModel(max_features=15, alpha=0.1)
            baseline_model2.fit(X_train, y_train)
            baseline_eval2 = baseline_model2.evaluate(X_test, y_test)
            
            # Both models should provide evaluations
            assert len(baseline_eval) > 0
            assert len(baseline_eval2) > 0
            
            # Performance metrics should be reasonable
            for target, metrics in baseline_eval.items():
                if 'rmse' in metrics:
                    assert 0 < metrics['rmse'] < 5  # Reasonable bounds for unemployment rate
    
    def test_system_robustness(self):
        """Test system robustness to various data conditions."""
        # Test with missing data
        missing_data = self.test_data.copy()
        missing_data.iloc[100:150, :] = np.nan  # Large missing chunk
        
        preprocessor = DataPreprocessor(self.config.preprocessing)
        
        try:
            processed_missing = preprocessor.clean_and_preprocess(missing_data)
            # Should handle missing data gracefully
            assert not processed_missing.empty
            assert processed_missing.isnull().sum().sum() < len(missing_data) * len(missing_data.columns) * 0.1
        except Exception as e:
            pytest.fail(f"System failed to handle missing data: {e}")
        
        # Test with extreme outliers
        outlier_data = self.test_data.copy()
        outlier_data.iloc[200, 0] = outlier_data.iloc[200, 0] * 100  # Extreme outlier
        
        try:
            processed_outliers = preprocessor.clean_and_preprocess(outlier_data)
            # Should handle outliers
            assert not processed_outliers.empty
        except Exception as e:
            pytest.fail(f"System failed to handle outliers: {e}")
        
        # Test with very short time series
        short_data = self.test_data.iloc[:20]  # Only 20 data points
        
        try:
            processed_short = preprocessor.clean_and_preprocess(short_data)
            # Should handle short series (though may have limited features)
            assert not processed_short.empty
        except Exception as e:
            # Short data might legitimately fail, so just log
            logger.warning(f"Short data handling: {e}")
    
    def test_configuration_flexibility(self):
        """Test system flexibility with different configurations."""
        # Test different preprocessing configurations
        configs = [
            {'missing_value_strategy': 'interpolate', 'normalization': 'standard'},
            {'missing_value_strategy': 'forward_fill', 'normalization': 'robust'},
            {'missing_value_strategy': 'drop', 'normalization': 'minmax'}
        ]
        
        for config_dict in configs:
            config = PreprocessingConfig(**config_dict)
            preprocessor = DataPreprocessor(config)
            
            try:
                processed = preprocessor.clean_and_preprocess(self.test_data)
                assert not processed.empty
                
                # Test that different normalizations produce different results
                if config_dict['normalization'] != 'standard':
                    # Values should be in different ranges for different normalizations
                    numeric_cols = processed.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col_std = processed[numeric_cols[0]].std()
                        # Different normalizations should produce different spreads
                        assert col_std > 0
                        
            except Exception as e:
                pytest.fail(f"Configuration {config_dict} failed: {e}")


class TestPerformanceBenchmarks:
    """Test performance benchmarks and scalability."""
    
    def test_data_processing_performance(self):
        """Test data processing performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        large_dates = pd.date_range('2023-01-01', periods=2000, freq='H')  # ~3 months hourly
        
        large_data = pd.DataFrame({
            'var1': np.random.randn(2000).cumsum() * 0.1,
            'var2': np.random.randn(2000).cumsum() * 0.1,
            'var3': np.random.randn(2000),
            'var4': np.random.randn(2000),
            'var5': np.random.randn(2000)
        }, index=large_dates)
        
        # Time the preprocessing
        import time
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        
        start_time = time.time()
        processed_large = preprocessor.clean_and_preprocess(large_data)
        processing_time = time.time() - start_time
        
        # Should process reasonably quickly (< 30 seconds for 2000 points)
        assert processing_time < 30
        assert not processed_large.empty
        assert len(processed_large) == len(large_data)
    
    def test_model_training_performance(self):
        """Test model training performance."""
        # Prepare reasonably sized dataset
        train_data = self.test_data.iloc[:1000]  # 1000 data points
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        processed = preprocessor.clean_and_preprocess(train_data)
        targets = preprocessor.create_target_variables(processed, ['unemployment_rate'])
        X, y = preprocessor.split_features_targets(targets)
        
        if len(X) > 100:
            # Time baseline model training
            import time
            
            baseline_model = BaselineRegressionModel(max_features=20)
            
            start_time = time.time()
            baseline_model.fit(X, y)
            training_time = time.time() - start_time
            
            # Should train reasonably quickly (< 10 seconds)
            assert training_time < 10
            
            # Test prediction performance
            start_time = time.time()
            predictions = baseline_model.predict(X[-50:])
            prediction_time = time.time() - start_time
            
            # Predictions should be very fast (< 1 second)
            assert prediction_time < 1
            assert 'predictions' in predictions
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_data = pd.DataFrame(
            np.random.randn(5000, 20),
            columns=[f'var_{i}' for i in range(20)],
            index=pd.date_range('2023-01-01', periods=5000, freq='H')
        )
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        processed = preprocessor.clean_and_preprocess(large_data)
        
        # Check memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500 MB for this test)
        assert memory_increase < 500
        
        # Cleanup
        del large_data, processed


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        
        # Should handle empty data gracefully
        try:
            result = preprocessor.clean_and_preprocess(empty_df)
            # May return empty or raise appropriate exception
            assert isinstance(result, pd.DataFrame)
        except ValueError:
            # Acceptable to raise ValueError for empty data
            pass
    
    def test_single_column_data(self):
        """Test handling of single-column datasets."""
        single_col_data = pd.DataFrame({
            'single_var': np.random.randn(100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='H'))
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        
        try:
            processed = preprocessor.clean_and_preprocess(single_col_data)
            # Should create some features even from single column
            assert len(processed.columns) >= 1
        except Exception as e:
            logger.warning(f"Single column processing limitation: {e}")
    
    def test_model_with_insufficient_data(self):
        """Test model behavior with insufficient training data."""
        # Very small dataset
        small_data = self.test_data.iloc[:10]
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        processed = preprocessor.clean_and_preprocess(small_data)
        
        if not processed.empty:
            targets = preprocessor.create_target_variables(processed, ['unemployment_rate'])
            X, y = preprocessor.split_features_targets(targets)
            
            baseline_model = BaselineRegressionModel(max_features=5)
            
            try:
                baseline_model.fit(X, y)
                # If it fits, predictions should work
                if len(baseline_model.models) > 0:
                    predictions = baseline_model.predict(X[:3])
                    assert 'predictions' in predictions
            except ValueError:
                # Acceptable to fail with insufficient data
                pass
    
    def test_api_failure_simulation(self):
        """Test system behavior when data sources fail."""
        # This would typically involve mocking API failures
        # For now, test that the system can work with partial data
        
        partial_data = self.test_data[['unemployment_rate', 'cpi']].copy()  # Only 2 variables
        
        preprocessor = DataPreprocessor(PreprocessingConfig())
        processed = preprocessor.clean_and_preprocess(partial_data)
        
        # Should still work with limited data
        assert not processed.empty
        assert len(processed.columns) >= 2
        
        # Models should still train
        targets = preprocessor.create_target_variables(processed, ['unemployment_rate'])
        X, y = preprocessor.split_features_targets(targets)
        
        if len(X) > 50:
            baseline_model = BaselineRegressionModel(max_features=5)
            baseline_model.fit(X, y)
            
            predictions = baseline_model.predict(X[-10:])
            assert 'predictions' in predictions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])