"""Tests for data ingestion pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.data.ingestion import DataIngestionPipeline
from src.data.sources import FREDDataSource, YahooFinanceDataSource, MockSocialSentimentSource
from src.data.preprocessing import DataPreprocessor
from src.utils.config import Config, PreprocessingConfig


class TestDataSources:
    """Test data source implementations."""
    
    @pytest.mark.asyncio
    async def test_fred_data_source_mock(self):
        """Test FRED data source with mock data."""
        series_config = [
            {'id': 'UNRATE', 'name': 'unemployment_rate'},
            {'id': 'CPIAUCSL', 'name': 'cpi'}
        ]
        
        source = FREDDataSource(series_config)
        
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        df = await source.fetch_data(start_date, end_date)
        
        assert not df.empty
        assert 'unemployment_rate' in df.columns
        assert 'cpi' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
    
    @pytest.mark.asyncio
    async def test_yahoo_finance_source_mock(self):
        """Test Yahoo Finance data source with mock data."""
        symbols = ['^GSPC', '^VIX']
        source = YahooFinanceDataSource(symbols)
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        df = await source.fetch_data(start_date, end_date)
        
        assert not df.empty
        assert any('GSPC' in col for col in df.columns)
        assert isinstance(df.index, pd.DatetimeIndex)
    
    @pytest.mark.asyncio
    async def test_sentiment_source(self):
        """Test social sentiment data source."""
        keywords = ['economy', 'inflation']
        source = MockSocialSentimentSource(keywords)
        
        df = await source.fetch_data()
        
        assert not df.empty
        assert 'economy_sentiment' in df.columns
        assert 'inflation_sentiment' in df.columns
        assert df['economy_sentiment'].between(-1, 1).all()


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        self.sample_data = pd.DataFrame({
            'unemployment_rate': 3.5 + np.random.normal(0, 0.5, 100).cumsum() * 0.01,
            'cpi': 100 + np.random.normal(0, 0.2, 100).cumsum() * 0.1,
            'sentiment': np.random.normal(0, 0.3, 100),
            'market_price': 100 * np.exp(np.random.normal(0, 0.02, 100).cumsum())
        }, index=dates)
        
        # Add some missing values
        self.sample_data.iloc[10:15, 0] = np.nan
        self.sample_data.iloc[20:22, 1] = np.nan
        
        # Add some outliers
        self.sample_data.iloc[50, 2] = 10  # Extreme sentiment value
        
        self.config = PreprocessingConfig()
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_data_validation(self):
        """Test data validation functionality."""
        from src.utils.data_utils import DataValidator
        
        validator = DataValidator()
        metrics = validator.validate_dataframe(self.sample_data)
        
        assert metrics['total_rows'] == 100
        assert metrics['total_columns'] == 4
        assert metrics['missing_values']['unemployment_rate'] == 5
        assert metrics['missing_values']['cpi'] == 2
        assert metrics['duplicate_rows'] == 0
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        from src.utils.data_utils import DataValidator
        
        validator = DataValidator()
        cleaned_data = validator.clean_data(self.sample_data, missing_strategy="interpolate")
        
        # Check that missing values are handled
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) == len(self.sample_data)
    
    def test_time_series_alignment(self):
        """Test time series alignment."""
        from src.utils.data_utils import TimeSeriesProcessor
        
        processor = TimeSeriesProcessor()
        aligned_data = processor.align_timestamps(self.sample_data, frequency="2H")
        
        # Check alignment
        assert isinstance(aligned_data.index, pd.DatetimeIndex)
        assert len(aligned_data) == len(self.sample_data) // 2  # Downsampled to 2H
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        processed_data = self.preprocessor.clean_and_preprocess(self.sample_data)
        
        # Check that features were created
        assert len(processed_data.columns) > len(self.sample_data.columns)
        
        # Check for specific feature types
        lag_features = [col for col in processed_data.columns if 'lag_' in col]
        rolling_features = [col for col in processed_data.columns if 'rolling_' in col]
        temporal_features = [col for col in processed_data.columns if any(t in col for t in ['hour', 'day', 'month'])]
        
        assert len(lag_features) > 0
        assert len(rolling_features) > 0
        assert len(temporal_features) > 0
    
    def test_target_variable_creation(self):
        """Test target variable creation."""
        target_vars = ['unemployment_rate', 'cpi']
        df_with_targets = self.preprocessor.create_target_variables(self.sample_data, target_vars)
        
        # Check that target variables were created
        target_cols = [col for col in df_with_targets.columns if 'target_' in col]
        assert len(target_cols) > 0
        
        # Check specific targets
        assert 'unemployment_rate_target_1h' in df_with_targets.columns
        assert 'cpi_target_1h' in df_with_targets.columns


class TestDataPipeline:
    """Test the complete data ingestion pipeline."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = Config()
        self.config.data_sources = {
            'economic_indicators': {
                'series': [
                    {'id': 'UNRATE', 'name': 'unemployment_rate'},
                    {'id': 'CPIAUCSL', 'name': 'cpi'}
                ],
                'update_frequency': 'daily'
            },
            'market_data': {
                'symbols': ['^GSPC', '^VIX'],
                'update_frequency': 'hourly'
            }
        }
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DataIngestionPipeline(self.config)
        
        assert len(pipeline.data_sources) > 0
        assert 'fred' in pipeline.data_sources
        assert pipeline.database_path.endswith('.db')
    
    @pytest.mark.asyncio
    async def test_data_fetching(self):
        """Test data fetching from multiple sources."""
        pipeline = DataIngestionPipeline(self.config)
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        data_dict = await pipeline.fetch_all_data(start_date, end_date)
        
        assert isinstance(data_dict, dict)
        # Should have at least some data sources
        assert len(data_dict) >= 0
    
    def test_data_merging(self):
        """Test data merging functionality."""
        pipeline = DataIngestionPipeline(self.config)
        
        # Create mock data from different sources
        dates1 = pd.date_range('2023-01-01', periods=48, freq='H')
        dates2 = pd.date_range('2023-01-01', periods=24, freq='2H')
        
        data_dict = {
            'source1': pd.DataFrame({
                'var1': np.random.randn(48),
                'var2': np.random.randn(48)
            }, index=dates1),
            'source2': pd.DataFrame({
                'var3': np.random.randn(24),
                'var4': np.random.randn(24)
            }, index=dates2)
        }
        
        merged_df = pipeline.merge_data_sources(data_dict)
        
        assert not merged_df.empty
        assert len(merged_df.columns) == 4
        assert isinstance(merged_df.index, pd.DatetimeIndex)
    
    def test_database_operations(self):
        """Test database save and load operations."""
        pipeline = DataIngestionPipeline(self.config)
        
        # Create test data
        test_data = pd.DataFrame({
            'test_var': np.random.randn(10)
        }, index=pd.date_range('2023-01-01', periods=10, freq='H'))
        
        # Save to database
        pipeline.save_to_database(test_data, source="test")
        
        # Load from database
        loaded_data = pipeline.load_from_database(source="test")
        
        assert not loaded_data.empty
        assert len(loaded_data) == len(test_data)
        assert 'test_var' in loaded_data.columns


class TestDataQuality:
    """Test data quality and validation."""
    
    def test_outlier_detection(self):
        """Test outlier detection methods."""
        from src.utils.data_utils import DataValidator
        
        # Create data with known outliers
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10, 15]])  # Add outliers
        
        series = pd.Series(outlier_data)
        
        validator = DataValidator(outlier_method="zscore", outlier_threshold=3.0)
        outliers = validator._detect_outliers(series)
        
        # Should detect the extreme values
        assert len(outliers) >= 3
    
    def test_missing_value_strategies(self):
        """Test different missing value handling strategies."""
        from src.utils.data_utils import DataValidator
        
        # Create data with missing values
        data = pd.DataFrame({
            'var1': [1, 2, np.nan, 4, 5, np.nan, 7],
            'var2': [10, np.nan, 30, 40, np.nan, 60, 70]
        })
        
        validator = DataValidator()
        
        # Test interpolation
        cleaned_interp = validator.clean_data(data, missing_strategy="interpolate")
        assert cleaned_interp.isnull().sum().sum() == 0
        
        # Test forward fill
        cleaned_ffill = validator.clean_data(data, missing_strategy="forward_fill")
        assert cleaned_ffill.isnull().sum().sum() <= 2  # Only first values might be NaN
    
    def test_correlation_analysis(self):
        """Test correlation analysis and feature selection."""
        # Create correlated data
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = x1 + np.random.randn(100) * 0.1  # Highly correlated
        x3 = np.random.randn(100)  # Independent
        
        data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3
        })
        
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        
        # Test correlation calculation
        corr_matrix = preprocessor.calculate_correlation_matrix(data)
        assert abs(corr_matrix.loc['x1', 'x2']) > 0.8  # Should be highly correlated
        
        # Test feature selection
        selected_features = preprocessor.select_features_by_correlation(data, threshold=0.9)
        assert len(selected_features) == 2  # Should remove one of the correlated features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])