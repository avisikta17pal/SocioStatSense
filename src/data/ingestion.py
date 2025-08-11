"""Data ingestion pipeline for orchestrating multiple data sources."""

import asyncio
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .sources import (
    FREDDataSource,
    YahooFinanceDataSource,
    TwitterSentimentSource,
    GoogleTrendsSource,
    WeatherDataSource
)
from .preprocessing import DataPreprocessor
from ..utils.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DataIngestionPipeline:
    """Orchestrates data ingestion from multiple sources."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_sources = {}
        self.preprocessor = DataPreprocessor(config.preprocessing)
        self.database_path = self._setup_database()
        self._initialize_data_sources()
        
    def _setup_database(self) -> str:
        """Setup SQLite database for storing time series data."""
        db_path = "data/socioeconomic.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables if they don't exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_series_data (
                timestamp DATETIME PRIMARY KEY,
                source TEXT,
                data_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                timestamp DATETIME PRIMARY KEY,
                model_name TEXT,
                target_variable TEXT,
                prediction REAL,
                lower_bound REAL,
                upper_bound REAL,
                uncertainty REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_points (
                timestamp DATETIME,
                variable TEXT,
                change_type TEXT,
                confidence REAL,
                PRIMARY KEY (timestamp, variable)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {db_path}")
        return db_path
        
    def _initialize_data_sources(self):
        """Initialize all configured data sources."""
        data_sources_config = self.config.data_sources
        
        # Initialize FRED data source
        if "economic_indicators" in data_sources_config:
            fred_config = data_sources_config["economic_indicators"]
            self.data_sources["fred"] = FREDDataSource(
                series_config=fred_config.get("series", []),
                update_frequency=fred_config.get("update_frequency", "daily")
            )
            
        # Initialize Yahoo Finance data source
        if "market_data" in data_sources_config:
            market_config = data_sources_config["market_data"]
            self.data_sources["yahoo"] = YahooFinanceDataSource(
                symbols=market_config.get("symbols", []),
                update_frequency=market_config.get("update_frequency", "hourly")
            )
            
        # Initialize social sentiment source
        if "social_sentiment" in data_sources_config:
            sentiment_config = data_sources_config["social_sentiment"]
            self.data_sources["sentiment"] = TwitterSentimentSource(
                keywords=sentiment_config.get("keywords", []),
                sample_size=sentiment_config.get("sample_size", 1000),
                update_frequency=sentiment_config.get("update_frequency", "hourly")
            )
            
        # Initialize Google Trends source
        if "google_trends" in data_sources_config:
            trends_config = data_sources_config["google_trends"]
            self.data_sources["trends"] = GoogleTrendsSource(
                keywords=trends_config.get("keywords", []),
                geo=trends_config.get("geo", "US"),
                timeframe=trends_config.get("timeframe", "now 7-d"),
                update_frequency=trends_config.get("update_frequency", "daily")
            )
            
        # Initialize weather data source
        if "weather_data" in data_sources_config:
            weather_config = data_sources_config["weather_data"]
            self.data_sources["weather"] = WeatherDataSource(
                locations=weather_config.get("locations", []),
                metrics=weather_config.get("metrics", ["temperature"]),
                update_frequency=weather_config.get("update_frequency", "daily")
            )
            
        logger.info(f"Initialized {len(self.data_sources)} data sources: {list(self.data_sources.keys())}")
    
    async def fetch_all_data(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data from all sources concurrently."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        logger.info(f"Starting data fetch from {len(self.data_sources)} sources")
        
        # Create concurrent tasks for all data sources
        tasks = []
        source_names = []
        
        for source_name, source in self.data_sources.items():
            if source.should_update():
                task = source.fetch_data(start_date, end_date)
                tasks.append(task)
                source_names.append(source_name)
            else:
                logger.info(f"Skipping {source_name} - not due for update")
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            fetched_data = {}
            for source_name, result in zip(source_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching from {source_name}: {result}")
                    fetched_data[source_name] = pd.DataFrame()
                else:
                    fetched_data[source_name] = result
                    
            logger.info(f"Data fetch completed for {len(fetched_data)} sources")
            return fetched_data
        else:
            logger.info("No data sources needed updating")
            return {}
    
    def merge_data_sources(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources into a unified DataFrame."""
        if not data_dict:
            return pd.DataFrame()
            
        # Find common time range
        all_indices = []
        for df in data_dict.values():
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                all_indices.extend(df.index.tolist())
        
        if not all_indices:
            logger.warning("No valid time series data found")
            return pd.DataFrame()
        
        # Create unified time index
        min_time = min(all_indices)
        max_time = max(all_indices)
        unified_index = pd.date_range(
            start=min_time,
            end=max_time,
            freq=self.config.preprocessing.time_alignment
        )
        
        # Merge all data sources
        merged_data = []
        for source_name, df in data_dict.items():
            if df.empty:
                continue
                
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Skipping {source_name} - no datetime index")
                continue
            
            # Resample to unified frequency
            df_resampled = df.resample(self.config.preprocessing.time_alignment).mean()
            
            # Reindex to unified index
            df_aligned = df_resampled.reindex(unified_index)
            
            merged_data.append(df_aligned)
        
        if merged_data:
            # Concatenate all DataFrames
            final_df = pd.concat(merged_data, axis=1)
            
            # Clean and preprocess
            final_df = self.preprocessor.clean_and_preprocess(final_df)
            
            logger.info(f"Merged data: {len(final_df)} rows, {len(final_df.columns)} columns")
            return final_df
        else:
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame, source: str = "merged") -> None:
        """Save DataFrame to SQLite database."""
        if df.empty:
            return
            
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Save raw data
            for timestamp, row in df.iterrows():
                data_json = row.to_json()
                
                conn.execute(
                    "INSERT OR REPLACE INTO time_series_data (timestamp, source, data_json) VALUES (?, ?, ?)",
                    (timestamp.isoformat(), source, data_json)
                )
            
            conn.commit()
            logger.info(f"Saved {len(df)} records to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
        finally:
            conn.close()
    
    def load_from_database(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          source: str = "merged") -> pd.DataFrame:
        """Load data from SQLite database."""
        conn = sqlite3.connect(self.database_path)
        
        try:
            query = "SELECT timestamp, data_json FROM time_series_data WHERE source = ?"
            params = [source]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
                
            query += " ORDER BY timestamp"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            if rows:
                data_list = []
                timestamps = []
                
                for timestamp_str, data_json in rows:
                    timestamps.append(pd.to_datetime(timestamp_str))
                    data_list.append(pd.Series(pd.read_json(data_json, typ='series')))
                
                df = pd.DataFrame(data_list, index=timestamps)
                logger.info(f"Loaded {len(df)} records from database")
                return df
            else:
                logger.info("No data found in database")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    async def run_continuous_ingestion(self, interval_minutes: int = 60):
        """Run continuous data ingestion loop."""
        logger.info(f"Starting continuous data ingestion with {interval_minutes} minute intervals")
        
        while True:
            try:
                # Fetch new data
                data_dict = await self.fetch_all_data()
                
                if data_dict:
                    # Merge and save data
                    merged_df = self.merge_data_sources(data_dict)
                    if not merged_df.empty:
                        self.save_to_database(merged_df)
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous ingestion: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def get_latest_data(self, hours_back: int = 24) -> pd.DataFrame:
        """Get the most recent data from the database."""
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        return self.load_from_database(start_date, end_date)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of available data."""
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Get data range and counts
            cursor = conn.execute("""
                SELECT 
                    source,
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest_data,
                    MAX(timestamp) as latest_data
                FROM time_series_data 
                GROUP BY source
            """)
            
            summary = {}
            for row in cursor.fetchall():
                source, count, earliest, latest = row
                summary[source] = {
                    'record_count': count,
                    'earliest_data': earliest,
                    'latest_data': latest,
                    'data_span_hours': (
                        pd.to_datetime(latest) - pd.to_datetime(earliest)
                    ).total_seconds() / 3600 if earliest and latest else 0
                }
            
            logger.info(f"Data summary generated for {len(summary)} sources")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {}
        finally:
            conn.close()