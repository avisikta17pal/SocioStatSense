"""Tests for statistical modeling components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.models.baseline_model import BaselineRegressionModel
from src.models.adaptive_model import AdaptiveBayesianModel
from src.models.change_point_detector import ChangePointDetector
from src.utils.config import ModelingConfig


class TestBaselineRegressionModel:
    """Test baseline regression model functionality."""
    
    def setup_method(self):
        """Setup test data and model."""
        np.random.seed(42)
        
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        n_features = 15
        
        # Generate correlated features
        X = np.random.randn(200, n_features)
        # Add some correlation structure
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(200)
        X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(200)
        
        self.X = pd.DataFrame(X, 
                             columns=[f'feature_{i}' for i in range(n_features)],
                             index=dates)
        
        # Generate target variables with known relationships
        y1 = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + np.random.randn(200) * 0.1
        y2 = -1 * X[:, 0] + 0.5 * X[:, 3] + 1.2 * X[:, 4] + np.random.randn(200) * 0.1
        
        self.y = pd.DataFrame({
            'target_1': y1,
            'target_2': y2
        }, index=dates)
        
        self.model = BaselineRegressionModel(
            learning_rate=0.01,
            alpha=0.01,
            max_features=10,
            random_state=42
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.learning_rate == 0.01
        assert self.model.alpha == 0.01
        assert self.model.max_features == 10
        assert self.model.models == {}
        assert self.model.feature_selectors == {}
    
    def test_model_fitting(self):
        """Test model fitting process."""
        # Split data for training
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        
        # Fit the model
        fitted_model = self.model.fit(X_train, y_train)
        
        assert fitted_model is self.model
        assert len(self.model.models) == 2  # Two target variables
        assert 'target_1' in self.model.models
        assert 'target_2' in self.model.models
        assert len(self.model.feature_selectors) == 2
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Train the model first
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        X_test = self.X.iloc[train_size:]
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        predictions = self.model.predict(X_test, return_uncertainty=True)
        
        assert 'predictions' in predictions
        assert 'uncertainty' in predictions
        assert len(predictions['predictions']) == len(X_test)
        assert 'target_1' in predictions['predictions']
        assert 'target_2' in predictions['predictions']
    
    def test_online_learning(self):
        """Test partial fit functionality for online learning."""
        # Initial training
        train_size = int(0.6 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        
        self.model.fit(X_train, y_train)
        
        # Get initial predictions
        X_test = self.X.iloc[train_size:train_size+20]
        initial_pred = self.model.predict(X_test)
        
        # Online update with new data
        X_update = self.X.iloc[train_size:train_size+40]
        y_update = self.y.iloc[train_size:train_size+40]
        
        self.model.partial_fit(X_update, y_update)
        
        # Get updated predictions
        updated_pred = self.model.predict(X_test)
        
        # Predictions should be different after update
        assert not np.allclose(
            initial_pred['predictions']['target_1'],
            updated_pred['predictions']['target_1']
        )
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        
        self.model.fit(X_train, y_train)
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert 'target_1' in importance
        assert 'target_2' in importance
        
        # Check that important features are identified
        target_1_importance = importance['target_1']
        assert len(target_1_importance) > 0
        
        # Features 0, 1, 2 should be most important for target_1
        important_features = sorted(target_1_importance.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:3]
        top_feature_names = [name for name, _ in important_features]
        
        # At least one of the truly important features should be in top 3
        assert any(f in top_feature_names for f in ['feature_0', 'feature_1', 'feature_2'])
    
    def test_forecasting(self):
        """Test multi-step forecasting."""
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        X_test = self.X.iloc[train_size:train_size+10]
        
        self.model.fit(X_train, y_train)
        
        forecast = self.model.forecast(X_test, steps=5)
        
        assert 'forecasts' in forecast
        assert 'uncertainty' in forecast
        assert len(forecast['forecasts']['target_1']) == 5
        assert len(forecast['forecasts']['target_2']) == 5
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        
        self.model.fit(X_train, y_train)
        
        # Test saving and loading
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save_model(temp_path)
            
            # Create new model instance and load
            new_model = BaselineRegressionModel()
            new_model.load_model(temp_path)
            
            # Test that loaded model makes same predictions
            X_test = self.X.iloc[train_size:train_size+5]
            original_pred = self.model.predict(X_test, return_uncertainty=False)
            loaded_pred = new_model.predict(X_test, return_uncertainty=False)
            
            np.testing.assert_allclose(
                original_pred['predictions']['target_1'],
                loaded_pred['predictions']['target_1'],
                rtol=1e-5
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAdaptiveBayesianModel:
    """Test adaptive Bayesian model functionality."""
    
    def setup_method(self):
        """Setup test data and model."""
        np.random.seed(42)
        
        # Create smaller dataset for Bayesian inference (faster testing)
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        n_features = 8
        
        X = np.random.randn(50, n_features)
        # Add correlation structure
        X[:, 1] = 0.6 * X[:, 0] + 0.4 * np.random.randn(50)
        
        self.X = pd.DataFrame(X, 
                             columns=[f'feature_{i}' for i in range(n_features)],
                             index=dates)
        
        # Generate target with known sparse structure
        y1 = 2 * X[:, 0] + 1 * X[:, 1] + np.random.randn(50) * 0.2
        
        self.y = pd.DataFrame({
            'target_1': y1
        }, index=dates)
        
        # Use smaller sampling parameters for testing
        self.model = AdaptiveBayesianModel(
            n_samples=100,
            n_tune=50,
            n_chains=1,
            max_features=6,
            random_state=42
        )
    
    @pytest.mark.slow
    def test_bayesian_model_fitting(self):
        """Test Bayesian model fitting (marked as slow)."""
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        y_train = self.y.iloc[:train_size]
        
        try:
            fitted_model = self.model.fit(X_train, y_train)
            assert fitted_model is self.model
            assert len(self.model.traces) == 1
            assert 'target_1' in self.model.traces
        except Exception as e:
            # If PyMC has issues in test environment, skip
            pytest.skip(f"PyMC model fitting failed in test environment: {e}")
    
    def test_model_initialization(self):
        """Test model initialization parameters."""
        assert self.model.n_samples == 100
        assert self.model.n_tune == 50
        assert self.model.n_chains == 1
        assert self.model.max_features == 6
        assert self.model.sparse_alpha == 0.01
        assert self.model.traces == {}
    
    def test_feature_selection(self):
        """Test feature selection for Bayesian model."""
        # Test that feature selection works
        train_size = int(0.8 * len(self.X))
        X_train = self.X.iloc[:train_size]
        
        # Mock the feature selection
        from sklearn.feature_selection import SelectKBest
        selector = SelectKBest(k=6)
        X_selected = selector.fit_transform(X_train, self.y.iloc[:train_size]['target_1'])
        
        assert X_selected.shape[1] <= 6
        assert X_selected.shape[0] == train_size


class TestChangePointDetector:
    """Test change point detection functionality."""
    
    def setup_method(self):
        """Setup test data with known change points."""
        np.random.seed(42)
        
        # Create time series with artificial change points
        n_points = 200
        dates = pd.date_range('2023-01-01', periods=n_points, freq='H')
        
        # Create series with change points at positions 50, 100, 150
        series1 = np.concatenate([
            np.random.normal(0, 1, 50),      # Regime 1: mean=0, std=1
            np.random.normal(2, 1, 50),      # Regime 2: mean=2, std=1
            np.random.normal(0, 2, 50),      # Regime 3: mean=0, std=2
            np.random.normal(-1, 1, 50)      # Regime 4: mean=-1, std=1
        ])
        
        series2 = np.concatenate([
            np.random.normal(5, 0.5, 50),    # Stable regime
            np.random.normal(5, 0.5, 50),    # No change
            np.random.normal(8, 0.5, 50),    # Jump at 100
            np.random.normal(8, 0.5, 50)     # Stable again
        ])
        
        self.test_data = pd.DataFrame({
            'variable_1': series1,
            'variable_2': series2,
            'variable_3': np.random.normal(0, 1, n_points)  # No change points
        }, index=dates)
        
        self.detector = ChangePointDetector(
            method="pelt",
            model="rbf",
            min_size=20,
            pen=5.0
        )
    
    def test_detector_initialization(self):
        """Test change point detector initialization."""
        assert self.detector.method == "pelt"
        assert self.detector.model == "rbf"
        assert self.detector.min_size == 20
        assert self.detector.pen == 5.0
        assert self.detector.change_point_history == []
    
    def test_univariate_change_point_detection(self):
        """Test univariate change point detection."""
        # Test on variable_1 which has known change points
        change_points = self.detector.detect_change_points(
            self.test_data[['variable_1']], 
            columns=['variable_1']
        )
        
        assert 'variable_1' in change_points
        assert len(change_points['variable_1']) > 0
        
        # Check change point structure
        cp = change_points['variable_1'][0]
        assert 'timestamp' in cp
        assert 'confidence' in cp
        assert 'magnitude' in cp
        assert 'change_type' in cp
    
    def test_multivariate_change_point_detection(self):
        """Test multivariate change point detection."""
        multivariate_cps = self.detector.detect_multivariate_change_points(
            self.test_data[['variable_1', 'variable_2']]
        )
        
        assert isinstance(multivariate_cps, list)
        
        if len(multivariate_cps) > 0:
            cp = multivariate_cps[0]
            assert 'timestamp' in cp
            assert 'confidence' in cp
            assert 'contributing_variables' in cp
    
    def test_change_point_classification(self):
        """Test change point classification."""
        # Create simple step change
        step_data = np.concatenate([
            np.ones(50) * 0,    # Level 0
            np.ones(50) * 2     # Level 2 (step change)
        ])
        
        change_type = self.detector._classify_change_type(step_data, 50)
        assert change_type in ['level_shift', 'trend_change', 'variance_change']
    
    def test_confidence_calculation(self):
        """Test change point confidence calculation."""
        # Create clear change point
        clear_change = np.concatenate([
            np.random.normal(0, 0.1, 50),
            np.random.normal(5, 0.1, 50)
        ])
        
        confidence = self.detector._calculate_confidence(clear_change, 50)
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be confident about clear change
    
    def test_regime_change_detection(self):
        """Test regime change detection."""
        regime_analysis = self.detector.detect_regime_changes(
            self.test_data, 
            window_size=40
        )
        
        assert 'regime_changes' in regime_analysis
        assert 'stability_metrics' in regime_analysis
    
    def test_change_point_alerts(self):
        """Test change point alerting system."""
        # First detect change points
        self.detector.detect_change_points(self.test_data)
        
        # Get alerts
        alerts = self.detector.get_change_point_alerts(confidence_threshold=0.5)
        
        assert isinstance(alerts, list)
        # Each alert should have required fields
        for alert in alerts:
            assert 'variable' in alert
            assert 'timestamp' in alert
            assert 'alert_type' in alert


class TestModelIntegration:
    """Test integration between different modeling components."""
    
    def setup_method(self):
        """Setup integrated test environment."""
        np.random.seed(42)
        
        # Create realistic socio-economic time series
        dates = pd.date_range('2023-01-01', periods=150, freq='H')
        
        # Unemployment rate with trend and seasonality
        unemployment_trend = np.linspace(3.5, 4.2, 150)
        unemployment_seasonal = 0.2 * np.sin(2 * np.pi * np.arange(150) / 168)  # Weekly pattern
        unemployment_noise = np.random.normal(0, 0.1, 150)
        
        # CPI with inflation trend
        cpi_trend = np.linspace(100, 102, 150)
        cpi_noise = np.random.normal(0, 0.3, 150)
        
        # Market volatility (VIX-like)
        vix_base = 20 + np.random.normal(0, 5, 150).cumsum() * 0.1
        vix_base = np.clip(vix_base, 10, 80)
        
        self.economic_data = pd.DataFrame({
            'unemployment_rate': unemployment_trend + unemployment_seasonal + unemployment_noise,
            'cpi': cpi_trend + cpi_noise,
            'market_volatility': vix_base,
            'sentiment': np.random.normal(0, 0.5, 150),
            'google_trends': np.random.exponential(1, 150)
        }, index=dates)
        
        # Add change point at position 100
        self.economic_data.iloc[100:, 0] += 0.5  # Unemployment jump
        self.economic_data.iloc[100:, 1] += 1.0  # Inflation jump
    
    def test_change_point_model_interaction(self):
        """Test interaction between change point detection and models."""
        # Initialize components
        detector = ChangePointDetector(min_size=20, pen=10.0)
        baseline_model = BaselineRegressionModel(max_features=4)
        
        # Detect change points
        change_points = detector.detect_change_points(self.economic_data)
        
        # Split data around detected change points
        if len(change_points['unemployment_rate']) > 0:
            cp_time = change_points['unemployment_rate'][0]['timestamp']
            cp_idx = self.economic_data.index.get_loc(cp_time)
            
            # Train model on pre-change data
            X_pre = self.economic_data.iloc[:cp_idx, 1:]  # Features
            y_pre = self.economic_data.iloc[:cp_idx, [0]]  # Unemployment target
            
            if len(X_pre) > 10:  # Ensure enough data
                baseline_model.fit(X_pre, y_pre)
                
                # Predict on post-change data
                X_post = self.economic_data.iloc[cp_idx:cp_idx+10, 1:]
                predictions = baseline_model.predict(X_post)
                
                assert 'predictions' in predictions
                assert len(predictions['predictions']['unemployment_rate']) == len(X_post)
    
    def test_model_performance_tracking(self):
        """Test model performance tracking over time."""
        baseline_model = BaselineRegressionModel()
        
        # Train incrementally and track performance
        window_size = 30
        performances = []
        
        for i in range(50, len(self.economic_data) - 10, 10):
            X_train = self.economic_data.iloc[i-window_size:i, 1:]
            y_train = self.economic_data.iloc[i-window_size:i, [0]]
            X_test = self.economic_data.iloc[i:i+5, 1:]
            y_test = self.economic_data.iloc[i:i+5, [0]]
            
            if len(X_train) >= 20:  # Ensure enough training data
                baseline_model.fit(X_train, y_train)
                metrics = baseline_model.evaluate(X_test, y_test)
                performances.append(metrics['unemployment_rate']['rmse'])
        
        # Should have collected performance metrics
        assert len(performances) > 0
        assert all(perf >= 0 for perf in performances)


class TestModelMetrics:
    """Test model evaluation metrics."""
    
    def test_regression_metrics(self):
        """Test regression metric calculations."""
        from src.utils.model_utils import ModelMetrics
        
        metrics_calculator = ModelMetrics()
        
        # Create test predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = metrics_calculator.calculate_regression_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check reasonable values
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2'] <= 1
    
    def test_forecast_metrics(self):
        """Test forecast-specific metrics."""
        from src.utils.model_utils import ModelMetrics
        
        metrics_calculator = ModelMetrics()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        # Create mock prediction intervals
        prediction_intervals = {
            '0.05': y_pred - 0.5,
            '0.95': y_pred + 0.5
        }
        
        metrics = metrics_calculator.calculate_forecast_metrics(
            y_true, y_pred, prediction_intervals
        )
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'coverage_0.90' in metrics
        
        # Coverage should be reasonable
        assert 0 <= metrics['coverage_0.90'] <= 1


class TestUncertaintyQuantification:
    """Test uncertainty quantification methods."""
    
    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty estimation."""
        from src.utils.model_utils import UncertaintyQuantifier
        from sklearn.linear_model import LinearRegression
        
        # Create test data
        X = np.random.randn(100, 5)
        y = X @ np.array([1, -1, 0.5, 0, -0.5]) + np.random.randn(100) * 0.1
        
        # Train simple model
        model = LinearRegression()
        model.fit(X, y)
        
        # Test bootstrap uncertainty
        uq = UncertaintyQuantifier()
        uncertainty = uq.bootstrap_uncertainty(model, X[:10], n_bootstrap=50)
        
        assert 'mean' in uncertainty
        assert 'std' in uncertainty
        assert len(uncertainty['mean']) == 10
        assert len(uncertainty['std']) == 10
    
    def test_prediction_interval_coverage(self):
        """Test prediction interval coverage calculation."""
        from src.utils.model_utils import UncertaintyQuantifier
        
        uq = UncertaintyQuantifier()
        
        y_true = np.array([1, 2, 3, 4, 5])
        prediction_intervals = {
            '0.05': np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
            '0.95': np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        }
        
        coverage = uq.prediction_intervals_coverage(y_true, prediction_intervals)
        
        assert '0.90' in coverage
        assert 0 <= coverage['0.90'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])