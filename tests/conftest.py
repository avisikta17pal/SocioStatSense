"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import Config, PreprocessingConfig, ModelingConfig
from src.utils.logging_config import setup_logging


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    config = Config()
    
    # Override with test-specific settings
    config.debug = True
    config.log_level = "WARNING"  # Reduce noise in tests
    
    # Test data sources
    config.data_sources = {
        'economic_indicators': {
            'series': [
                {'id': 'UNRATE', 'name': 'unemployment_rate'},
                {'id': 'CPIAUCSL', 'name': 'cpi'},
                {'id': 'GDP', 'name': 'gdp'}
            ],
            'update_frequency': 'daily'
        },
        'market_data': {
            'symbols': ['^GSPC', '^VIX', 'DXY'],
            'update_frequency': 'hourly'
        },
        'social_sentiment': {
            'keywords': ['economy', 'inflation', 'unemployment'],
            'update_frequency': 'hourly'
        }
    }
    
    return config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_economic_data():
    """Generate sample economic time series data."""
    np.random.seed(42)
    
    # Create 7 days of hourly data
    dates = pd.date_range('2023-01-01', periods=168, freq='H')
    
    # Generate realistic economic indicators
    
    # Policy rate with random walk
    policy_rate = 2.0 + np.random.normal(0, 0.05, 168).cumsum() * 0.01
    policy_rate = np.clip(policy_rate, 0.25, 5.0)
    
    # Unemployment rate with policy response
    unemployment = np.zeros(168)
    unemployment[0] = 3.8
    for t in range(1, 168):
        unemployment[t] = (0.99 * unemployment[t-1] + 
                          0.01 * policy_rate[max(0, t-6)] +  # 6-hour lag
                          np.random.normal(0, 0.005))
    unemployment = np.clip(unemployment, 2.0, 8.0)
    
    # CPI with inflation dynamics
    cpi = 100 * np.exp(np.random.normal(0.02/8760, 0.001, 168).cumsum())
    
    # Market volatility (VIX-like)
    vix = 20 + np.random.normal(0, 1, 168).cumsum() * 0.1
    vix = np.clip(vix, 10, 50)
    
    # Social sentiment
    sentiment = np.random.normal(0, 1, 168)
    
    # Google Trends proxy
    google_trends = np.random.exponential(1, 168)
    
    return pd.DataFrame({
        'policy_rate': policy_rate,
        'unemployment_rate': unemployment,
        'cpi': cpi,
        'vix': vix,
        'sentiment': sentiment,
        'google_trends': google_trends
    }, index=dates)


@pytest.fixture
def sample_causal_data():
    """Generate sample data with known causal relationships."""
    np.random.seed(42)
    
    n_points = 200
    dates = pd.date_range('2023-01-01', periods=n_points, freq='H')
    
    # Create causal chain: X1 -> X2 -> X3
    x1 = np.random.normal(0, 1, n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    
    x2[0] = np.random.normal(0, 1)
    x3[0] = np.random.normal(0, 1)
    
    for t in range(1, n_points):
        # X2 depends on X1 with lag
        x2[t] = 0.3 * x2[t-1] + 0.6 * x1[t-1] + np.random.normal(0, 0.3)
        
        # X3 depends on X2 with lag
        x3[t] = 0.4 * x3[t-1] + 0.5 * x2[t-1] + np.random.normal(0, 0.3)
    
    # Independent variable
    x4 = np.random.normal(0, 1, n_points)
    
    return pd.DataFrame({
        'cause_var': x1,
        'intermediate_var': x2,
        'effect_var': x3,
        'independent_var': x4
    }, index=dates)


@pytest.fixture
def trained_baseline_model(sample_economic_data):
    """Create and train a baseline model with sample data."""
    from src.data.preprocessing import DataPreprocessor
    from src.models.baseline_model import BaselineRegressionModel
    
    # Preprocess data
    preprocessor = DataPreprocessor(PreprocessingConfig())
    processed_data = preprocessor.clean_and_preprocess(sample_economic_data)
    
    # Create targets
    targets_data = preprocessor.create_target_variables(processed_data, ['unemployment_rate'])
    X, y = preprocessor.split_features_targets(targets_data)
    
    # Train model
    model = BaselineRegressionModel(max_features=10, random_state=42)
    
    if len(X) > 20:
        model.fit(X, y)
    
    return model, X, y


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing data sources."""
    return {
        'fred_response': {
            'observations': [
                {'date': '2023-01-01', 'value': '3.5'},
                {'date': '2023-01-02', 'value': '3.6'},
                {'date': '2023-01-03', 'value': '3.4'}
            ]
        },
        'yahoo_response': pd.DataFrame({
            'Open': [100, 101, 99],
            'High': [102, 103, 101],
            'Low': [99, 100, 98],
            'Close': [101, 100, 100],
            'Volume': [1000000, 1100000, 950000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    setup_logging(log_level="WARNING", log_file=None)  # Console only for tests


@pytest.fixture
def change_point_data():
    """Generate data with known change points for testing."""
    np.random.seed(42)
    
    # Create time series with change points at known positions
    n_points = 300
    dates = pd.date_range('2023-01-01', periods=n_points, freq='H')
    
    # Series with level shift at position 100
    level_shift_series = np.concatenate([
        np.random.normal(0, 1, 100),    # Level 0
        np.random.normal(3, 1, 100),    # Level 3 (shift)
        np.random.normal(0, 1, 100)     # Back to level 0
    ])
    
    # Series with variance change at position 150
    variance_change_series = np.concatenate([
        np.random.normal(0, 1, 150),    # Low variance
        np.random.normal(0, 3, 150)     # High variance
    ])
    
    # Series with trend change
    trend_change_series = np.concatenate([
        np.linspace(0, 5, 100) + np.random.normal(0, 0.5, 100),      # Upward trend
        np.linspace(5, 2, 100) + np.random.normal(0, 0.5, 100),      # Downward trend  
        np.linspace(2, 8, 100) + np.random.normal(0, 0.5, 100)       # Strong upward trend
    ])
    
    return pd.DataFrame({
        'level_shift': level_shift_series,
        'variance_change': variance_change_series,
        'trend_change': trend_change_series,
        'no_change': np.random.normal(0, 1, n_points)
    }, index=dates)


class DatabaseTestHelper:
    """Helper class for database testing."""
    
    @staticmethod
    def create_test_database(temp_dir: Path) -> str:
        """Create a test database and return connection string."""
        db_path = temp_dir / "test.db"
        return f"sqlite:///{db_path}"
    
    @staticmethod
    def cleanup_test_database(db_path: str):
        """Clean up test database."""
        if db_path.startswith("sqlite:///"):
            file_path = db_path.replace("sqlite:///", "")
            if os.path.exists(file_path):
                os.unlink(file_path)


@pytest.fixture
def test_database(temp_dir):
    """Create test database for testing."""
    db_path = DatabaseTestHelper.create_test_database(temp_dir)
    yield db_path
    DatabaseTestHelper.cleanup_test_database(db_path)


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, max_time: float = None):
        self.max_time = max_time
        self.elapsed_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed_time = time.time() - self.start_time
        
        if self.max_time and self.elapsed_time > self.max_time:
            pytest.fail(f"Operation took {self.elapsed_time:.2f}s, expected < {self.max_time}s")


@pytest.fixture
def performance_timer():
    """Fixture for performance timing."""
    return PerformanceTimer


# Skip markers for optional dependencies
def pytest_runtest_setup(item):
    """Setup function to handle conditional test skipping."""
    # Skip PyMC tests if PyMC is not available or has issues
    if "bayesian" in item.name.lower():
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            pytest.skip("PyMC not available")
        except Exception:
            pytest.skip("PyMC initialization failed")
    
    # Skip tests requiring specific APIs if not configured
    if "api" in item.name.lower():
        # Could check for API keys here
        pass


# Memory monitoring fixture
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        yield {
            'initial_memory': initial_memory,
            'process': process
        }
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Log significant memory increases
        if memory_increase > 100:  # More than 100MB increase
            print(f"\nWarning: Test increased memory by {memory_increase:.1f}MB")
            
    except ImportError:
        # psutil not available
        yield {'initial_memory': 0, 'process': None}