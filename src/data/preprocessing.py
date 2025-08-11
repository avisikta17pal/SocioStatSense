"""Data preprocessing module for cleaning and feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats

from ..utils.config import PreprocessingConfig
from ..utils.data_utils import DataValidator, TimeSeriesProcessor
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessing for socio-economic time series."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.validator = DataValidator(
            outlier_method=config.outlier_detection,
            outlier_threshold=config.outlier_threshold
        )
        self.ts_processor = TimeSeriesProcessor(
            normalization_method=config.normalization
        )
        self.feature_scalers = {}
        self.is_fitted = False
        
    def clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        if df.empty:
            return df
            
        logger.info(f"Starting preprocessing pipeline for {len(df)} rows, {len(df.columns)} columns")
        
        # Step 1: Validate and get quality metrics
        quality_metrics = self.validator.validate_dataframe(df)
        
        # Step 2: Clean data (handle missing values and outliers)
        df_clean = self.validator.clean_data(
            df,
            missing_strategy=self.config.missing_value_strategy,
            remove_outliers=True
        )
        
        # Step 3: Align timestamps if needed
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean = self.ts_processor.align_timestamps(
                df_clean, 
                frequency=self.config.time_alignment
            )
        
        # Step 4: Create engineered features
        df_engineered = self._engineer_features(df_clean)
        
        # Step 5: Normalize features
        df_normalized = self.ts_processor.normalize_features(df_engineered)
        
        # Step 6: Final validation
        final_metrics = self.validator.validate_dataframe(df_normalized)
        
        logger.info(f"Preprocessing completed: {len(df_normalized)} rows, {len(df_normalized.columns)} columns")
        self.is_fitted = True
        
        return df_normalized
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data."""
        df_features = df.copy()
        
        # Create lagged features
        df_features = self.ts_processor.create_lagged_features(
            df_features, 
            lags=[1, 2, 3, 6, 12, 24]
        )
        
        # Create rolling statistical features
        df_features = self.ts_processor.create_rolling_features(
            df_features,
            windows=[6, 12, 24, 48, 168]  # 6h, 12h, 1d, 2d, 1w
        )
        
        # Create time-based features
        df_features = self._create_temporal_features(df_features)
        
        # Create interaction features for key economic indicators
        df_features = self._create_interaction_features(df_features)
        
        # Create volatility and momentum features
        df_features = self._create_volatility_features(df_features)
        
        logger.info(f"Feature engineering completed: {len(df_features.columns)} total features")
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        df_temporal = df.copy()
        
        # Extract temporal components
        df_temporal['hour'] = df.index.hour
        df_temporal['day_of_week'] = df.index.dayofweek
        df_temporal['day_of_month'] = df.index.day
        df_temporal['month'] = df.index.month
        df_temporal['quarter'] = df.index.quarter
        df_temporal['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df_temporal['is_month_end'] = df.index.is_month_end.astype(int)
        df_temporal['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Create cyclical features
        df_temporal['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df_temporal['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df_temporal['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df_temporal['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df_temporal['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df_temporal['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        logger.info("Created temporal features")
        return df_temporal
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key economic variables."""
        df_interactions = df.copy()
        
        # Define key economic relationships
        interactions = [
            ('unemployment_rate', 'fed_funds_rate'),
            ('cpi', 'fed_funds_rate'),
            ('unemployment_rate', 'cpi'),
        ]
        
        # Add market sentiment interactions
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        economic_cols = ['unemployment_rate', 'cpi', 'gdp', 'fed_funds_rate']
        
        for sent_col in sentiment_cols[:3]:  # Limit to avoid too many features
            for econ_col in economic_cols:
                if sent_col in df.columns and econ_col in df.columns:
                    interactions.append((sent_col, econ_col))
        
        # Create interaction terms
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
        
        logger.info(f"Created {len(interactions)} interaction features")
        return df_interactions
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility and momentum features."""
        df_vol = df.copy()
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and not col.endswith(('_sin', '_cos', '_lag_', '_rolling_')):
                # Calculate returns/changes
                df_vol[f"{col}_returns"] = df[col].pct_change()
                df_vol[f"{col}_diff"] = df[col].diff()
                
                # Calculate volatility (rolling standard deviation of returns)
                if f"{col}_returns" in df_vol.columns:
                    df_vol[f"{col}_volatility_12h"] = df_vol[f"{col}_returns"].rolling(12).std()
                    df_vol[f"{col}_volatility_24h"] = df_vol[f"{col}_returns"].rolling(24).std()
                
                # Calculate momentum indicators
                df_vol[f"{col}_momentum_6h"] = df[col] / df[col].shift(6) - 1
                df_vol[f"{col}_momentum_24h"] = df[col] / df[col].shift(24) - 1
        
        logger.info("Created volatility and momentum features")
        return df_vol
    
    def detect_anomalies(self, df: pd.DataFrame, 
                        contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies in the dataset."""
        if df.empty:
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
        
        # Use Isolation Forest for multivariate anomaly detection
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit on numeric columns only
        X = df[numeric_cols].fillna(0)  # Fill NaN for anomaly detection
        anomaly_labels = iso_forest.fit_predict(X)
        
        # Add anomaly indicator
        df_with_anomalies = df.copy()
        df_with_anomalies['is_anomaly'] = (anomaly_labels == -1).astype(int)
        
        anomaly_count = df_with_anomalies['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        return df_with_anomalies
    
    def create_target_variables(self, df: pd.DataFrame, 
                               target_cols: List[str] = None) -> pd.DataFrame:
        """Create target variables for prediction."""
        if target_cols is None:
            # Default targets: key economic indicators
            target_cols = ['unemployment_rate', 'cpi', 'fed_funds_rate']
        
        df_targets = df.copy()
        
        for target_col in target_cols:
            if target_col in df.columns:
                # Create future values as targets
                for horizon in [1, 3, 6, 12, 24]:  # 1h to 24h ahead
                    df_targets[f"{target_col}_target_{horizon}h"] = df[target_col].shift(-horizon)
        
        logger.info(f"Created target variables for {len(target_cols)} indicators")
        return df_targets
    
    def split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into features and targets."""
        target_cols = [col for col in df.columns if col.endswith('_target_1h')]
        feature_cols = [col for col in df.columns if not col.startswith('target_') and not col.endswith(('_target_1h', '_target_3h', '_target_6h', '_target_12h', '_target_24h'))]
        
        X = df[feature_cols].copy()
        y = df[target_cols].copy()
        
        logger.info(f"Split data: {len(feature_cols)} features, {len(target_cols)} targets")
        return X, y
    
    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by type for interpretability."""
        feature_groups = {
            'economic_indicators': [],
            'market_data': [],
            'sentiment': [],
            'trends': [],
            'weather': [],
            'temporal': [],
            'lagged': [],
            'rolling': [],
            'interactions': [],
            'volatility': []
        }
        
        for col in df.columns:
            if any(indicator in col for indicator in ['unemployment', 'cpi', 'gdp', 'fed_funds']):
                if 'lag_' in col:
                    feature_groups['lagged'].append(col)
                elif 'rolling_' in col:
                    feature_groups['rolling'].append(col)
                elif '_x_' in col:
                    feature_groups['interactions'].append(col)
                elif any(vol_term in col for vol_term in ['volatility', 'returns', 'momentum']):
                    feature_groups['volatility'].append(col)
                else:
                    feature_groups['economic_indicators'].append(col)
                    
            elif any(market_term in col for market_term in ['_close', '_volume', 'GSPC', 'VIX', 'DXY']):
                if 'lag_' in col:
                    feature_groups['lagged'].append(col)
                elif 'rolling_' in col:
                    feature_groups['rolling'].append(col)
                elif any(vol_term in col for vol_term in ['volatility', 'returns', 'momentum']):
                    feature_groups['volatility'].append(col)
                else:
                    feature_groups['market_data'].append(col)
                    
            elif 'sentiment' in col:
                feature_groups['sentiment'].append(col)
            elif 'trends' in col:
                feature_groups['trends'].append(col)
            elif any(weather_term in col for weather_term in ['temperature', 'humidity', 'pressure']):
                feature_groups['weather'].append(col)
            elif any(time_term in col for time_term in ['hour', 'day', 'month', 'quarter', 'weekend', 'sin', 'cos']):
                feature_groups['temporal'].append(col)
            elif 'lag_' in col:
                feature_groups['lagged'].append(col)
            elif 'rolling_' in col:
                feature_groups['rolling'].append(col)
            elif '_x_' in col:
                feature_groups['interactions'].append(col)
            elif any(vol_term in col for vol_term in ['volatility', 'returns', 'momentum']):
                feature_groups['volatility'].append(col)
        
        # Remove empty groups
        feature_groups = {k: v for k, v in feature_groups.items() if v}
        
        logger.info(f"Organized features into {len(feature_groups)} groups")
        return feature_groups
    
    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for feature selection."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        logger.info(f"Calculated correlation matrix for {len(numeric_cols)} features")
        return correlation_matrix
    
    def select_features_by_correlation(self, df: pd.DataFrame, 
                                     threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = self.calculate_correlation_matrix(df)
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for col1, col2 in high_corr_pairs:
            # Keep the feature with fewer missing values
            if df[col1].isnull().sum() <= df[col2].isnull().sum():
                features_to_remove.add(col2)
            else:
                features_to_remove.add(col1)
        
        selected_features = [col for col in df.columns if col not in features_to_remove]
        
        logger.info(f"Feature selection: removed {len(features_to_remove)} highly correlated features")
        return selected_features
    
    def create_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived economic indicators."""
        df_indicators = df.copy()
        
        # Unemployment rate change
        if 'unemployment_rate' in df.columns:
            df_indicators['unemployment_change'] = df['unemployment_rate'].diff()
            df_indicators['unemployment_acceleration'] = df_indicators['unemployment_change'].diff()
        
        # Inflation rate (CPI change)
        if 'cpi' in df.columns:
            df_indicators['inflation_rate'] = df['cpi'].pct_change(periods=12) * 100  # YoY
            df_indicators['inflation_change'] = df_indicators['inflation_rate'].diff()
        
        # Real interest rate (Fed funds - inflation)
        if 'fed_funds_rate' in df.columns and 'inflation_rate' in df_indicators.columns:
            df_indicators['real_interest_rate'] = df['fed_funds_rate'] - df_indicators['inflation_rate']
        
        # Market stress indicator (VIX-based if available)
        vix_cols = [col for col in df.columns if 'VIX' in col and 'close' in col]
        if vix_cols:
            vix_col = vix_cols[0]
            df_indicators['market_stress'] = (df[vix_col] - df[vix_col].rolling(30).mean()) / df[vix_col].rolling(30).std()
        
        # Economic sentiment composite
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        if sentiment_cols:
            df_indicators['economic_sentiment_composite'] = df[sentiment_cols].mean(axis=1)
        
        # Search interest economic indicator
        trends_cols = [col for col in df.columns if 'trends' in col]
        if trends_cols:
            df_indicators['search_interest_composite'] = df[trends_cols].mean(axis=1)
        
        logger.info("Created derived economic indicators")
        return df_indicators
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        df_interactions = df.copy()
        
        # Key economic variable interactions
        key_vars = ['unemployment_rate', 'cpi', 'fed_funds_rate']
        available_key_vars = [var for var in key_vars if var in df.columns]
        
        # Create pairwise interactions
        for i, var1 in enumerate(available_key_vars):
            for var2 in available_key_vars[i+1:]:
                df_interactions[f"{var1}_x_{var2}"] = df[var1] * df[var2]
        
        # Sentiment-economic interactions
        sentiment_cols = [col for col in df.columns if 'sentiment' in col][:2]  # Limit to avoid explosion
        for sent_col in sentiment_cols:
            for econ_var in available_key_vars:
                df_interactions[f"{sent_col}_x_{econ_var}"] = df[sent_col] * df[econ_var]
        
        return df_interactions
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility and momentum features."""
        df_vol = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        base_cols = [col for col in numeric_cols if not any(suffix in col for suffix in ['_lag_', '_rolling_', '_x_', '_sin', '_cos'])]
        
        for col in base_cols[:10]:  # Limit to avoid too many features
            if col in df.columns:
                # Returns
                df_vol[f"{col}_returns"] = df[col].pct_change()
                
                # Volatility (rolling std of returns)
                if f"{col}_returns" in df_vol.columns:
                    df_vol[f"{col}_vol_6h"] = df_vol[f"{col}_returns"].rolling(6).std()
                    df_vol[f"{col}_vol_24h"] = df_vol[f"{col}_returns"].rolling(24).std()
                
                # Momentum
                df_vol[f"{col}_momentum_6h"] = df[col] / df[col].shift(6) - 1
                df_vol[f"{col}_momentum_24h"] = df[col] / df[col].shift(24) - 1
                
                # Relative strength
                df_vol[f"{col}_rsi_14"] = self._calculate_rsi(df[col], window=14)
        
        return df_vol
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations."""
        return {
            'missing_value_strategy': self.config.missing_value_strategy,
            'outlier_detection': self.config.outlier_detection,
            'outlier_threshold': self.config.outlier_threshold,
            'normalization': self.config.normalization,
            'time_alignment': self.config.time_alignment,
            'is_fitted': self.is_fitted,
            'scalers_fitted': len(self.ts_processor.scalers)
        }