"""Data processing utilities for the adaptive modeling framework."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from .logging_config import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates and cleans incoming data streams."""
    
    def __init__(self, outlier_method: str = "iqr", outlier_threshold: float = 3.0):
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a DataFrame and return quality metrics."""
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "data_types": df.dtypes.to_dict(),
            "outliers": {},
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Detect outliers for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col].dropna())
            metrics["outliers"][col] = len(outliers)
            
        logger.info(f"Data validation completed: {metrics['total_rows']} rows, "
                   f"{metrics['total_columns']} columns, "
                   f"{sum(metrics['missing_values'].values())} missing values")
        
        return metrics
    
    def _detect_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using specified method."""
        if self.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series[(series < lower_bound) | (series > upper_bound)].index
            
        elif self.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(series))
            return series[z_scores > self.outlier_threshold].index
            
        elif self.outlier_method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(series.values.reshape(-1, 1))
            return series[outliers == -1].index
            
        else:
            raise ValueError(f"Unknown outlier detection method: {self.outlier_method}")
    
    def clean_data(self, df: pd.DataFrame, 
                   missing_strategy: str = "interpolate",
                   remove_outliers: bool = False) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        df_clean = df.copy()
        
        # Handle missing values
        if missing_strategy == "interpolate":
            df_clean = df_clean.interpolate(method='time')
        elif missing_strategy == "forward_fill":
            df_clean = df_clean.fillna(method='ffill')
        elif missing_strategy == "drop":
            df_clean = df_clean.dropna()
        
        # Remove outliers if requested
        if remove_outliers:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                outlier_indices = self._detect_outliers(df_clean[col].dropna())
                df_clean.loc[outlier_indices, col] = np.nan
            
            # Re-interpolate after outlier removal
            if missing_strategy == "interpolate":
                df_clean = df_clean.interpolate(method='time')
        
        logger.info(f"Data cleaning completed: {len(df_clean)} rows remaining")
        return df_clean


class TimeSeriesProcessor:
    """Processes time series data for modeling."""
    
    def __init__(self, normalization_method: str = "standard"):
        self.normalization_method = normalization_method
        self.scalers = {}
        
    def align_timestamps(self, df: pd.DataFrame, frequency: str = "1H") -> pd.DataFrame:
        """Align time series to uniform frequency."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
            
        # Resample to uniform frequency
        df_aligned = df.resample(frequency).mean()
        
        # Forward fill any remaining NaN values
        df_aligned = df_aligned.fillna(method='ffill')
        
        logger.info(f"Time series aligned to {frequency} frequency: {len(df_aligned)} points")
        return df_aligned
    
    def normalize_features(self, df: pd.DataFrame, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize specified columns using the configured method."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_norm = df.copy()
        
        for col in columns:
            if col not in df_norm.columns:
                continue
                
            if col not in self.scalers:
                if self.normalization_method == "standard":
                    self.scalers[col] = StandardScaler()
                elif self.normalization_method == "robust":
                    self.scalers[col] = RobustScaler()
                elif self.normalization_method == "minmax":
                    self.scalers[col] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # Fit and transform
            values = df_norm[col].dropna().values.reshape(-1, 1)
            if len(values) > 0:
                if not hasattr(self.scalers[col], 'scale_'):
                    self.scalers[col].fit(values)
                df_norm[col] = self.scalers[col].transform(
                    df_norm[col].values.reshape(-1, 1)
                ).flatten()
        
        logger.info(f"Normalized {len(columns)} features using {self.normalization_method}")
        return df_norm
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lagged versions of features for time series modeling."""
        df_lagged = df.copy()
        
        for col in df.columns:
            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        logger.info(f"Created lagged features with lags: {lags}")
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame,
                               windows: List[int] = [6, 12, 24, 48]) -> pd.DataFrame:
        """Create rolling statistical features."""
        df_rolling = df.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            for window in windows:
                df_rolling[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df_rolling[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
                df_rolling[f"{col}_rolling_min_{window}"] = df[col].rolling(window).min()
                df_rolling[f"{col}_rolling_max_{window}"] = df[col].rolling(window).max()
        
        logger.info(f"Created rolling features with windows: {windows}")
        return df_rolling
    
    def detect_seasonality(self, series: pd.Series, 
                          periods: List[int] = [24, 168, 8760]) -> Dict[int, float]:
        """Detect seasonal patterns in time series."""
        seasonality_scores = {}
        
        for period in periods:
            if len(series) >= 2 * period:
                # Calculate autocorrelation at seasonal lag
                autocorr = series.autocorr(lag=period)
                seasonality_scores[period] = abs(autocorr) if not pd.isna(autocorr) else 0.0
        
        logger.info(f"Seasonality detection completed for periods: {periods}")
        return seasonality_scores