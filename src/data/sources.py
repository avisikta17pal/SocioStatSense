"""Data source implementations for various socio-economic data providers."""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time
import json
from pytrends.request import TrendReq
from textblob import TextBlob

from ..utils.config import get_api_key
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, update_frequency: str = "daily"):
        self.name = name
        self.update_frequency = update_frequency
        self.last_update = None
        
    @abstractmethod
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch data from the source."""
        pass
    
    def should_update(self) -> bool:
        """Check if data source should be updated based on frequency."""
        if self.last_update is None:
            return True
            
        frequency_map = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1)
        }
        
        time_since_update = datetime.now() - self.last_update
        return time_since_update >= frequency_map.get(self.update_frequency, timedelta(days=1))


class FREDDataSource(BaseDataSource):
    """Federal Reserve Economic Data (FRED) source."""
    
    def __init__(self, series_config: Dict[str, Any], update_frequency: str = "daily"):
        super().__init__("FRED", update_frequency)
        self.api_key = get_api_key("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred"
        self.series_config = series_config
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch economic indicators from FRED API."""
        if not self.api_key:
            logger.warning("FRED API key not available, using mock data")
            return self._generate_mock_data(start_date, end_date)
        
        all_data = {}
        
        for series in self.series_config:
            series_id = series['id']
            series_name = series['name']
            
            try:
                # Build API URL
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json'
                }
                
                if start_date:
                    params['observation_start'] = start_date.strftime('%Y-%m-%d')
                if end_date:
                    params['observation_end'] = end_date.strftime('%Y-%m-%d')
                
                url = f"{self.base_url}/series/observations"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            if observations:
                                df_series = pd.DataFrame(observations)
                                df_series['date'] = pd.to_datetime(df_series['date'])
                                df_series['value'] = pd.to_numeric(df_series['value'], errors='coerce')
                                df_series = df_series.set_index('date')
                                all_data[series_name] = df_series['value']
                                
                                logger.info(f"Fetched {len(df_series)} observations for {series_name}")
                        else:
                            logger.error(f"Failed to fetch {series_name}: HTTP {response.status}")
                            
            except Exception as e:
                logger.error(f"Error fetching {series_name}: {str(e)}")
                
        if all_data:
            df = pd.DataFrame(all_data)
            self.last_update = datetime.now()
            return df
        else:
            return self._generate_mock_data(start_date, end_date)
    
    def _generate_mock_data(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock FRED data for testing."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic economic indicators
        np.random.seed(42)
        data = {
            'unemployment_rate': 3.5 + np.random.normal(0, 0.5, len(date_range)).cumsum() * 0.01,
            'cpi': 100 + np.random.normal(0, 0.2, len(date_range)).cumsum() * 0.1,
            'gdp': 20000 + np.random.normal(0, 100, len(date_range)).cumsum(),
            'fed_funds_rate': 2.0 + np.random.normal(0, 0.1, len(date_range)).cumsum() * 0.01
        }
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock FRED data: {len(df)} observations")
        return df


class YahooFinanceDataSource(BaseDataSource):
    """Yahoo Finance market data source."""
    
    def __init__(self, symbols: List[str], update_frequency: str = "hourly"):
        super().__init__("Yahoo Finance", update_frequency)
        self.symbols = symbols
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch market data from Yahoo Finance."""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            all_data = {}
            
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval="1h"
                    )
                    
                    if not hist.empty:
                        # Use closing price
                        all_data[f"{symbol}_close"] = hist['Close']
                        all_data[f"{symbol}_volume"] = hist['Volume']
                        
                        logger.info(f"Fetched {len(hist)} observations for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {str(e)}")
            
            if all_data:
                df = pd.DataFrame(all_data)
                self.last_update = datetime.now()
                return df
            else:
                return self._generate_mock_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error: {str(e)}")
            return self._generate_mock_data(start_date, end_date)
    
    def _generate_mock_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate mock market data."""
        date_range = pd.date_range(start_date, end_date, freq='H')
        
        np.random.seed(42)
        data = {}
        
        for symbol in self.symbols:
            # Generate realistic price movements
            returns = np.random.normal(0, 0.02, len(date_range))
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(10, 1, len(date_range))
            
            data[f"{symbol}_close"] = prices
            data[f"{symbol}_volume"] = volumes
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock market data: {len(df)} observations")
        return df


class TwitterSentimentSource(BaseDataSource):
    """Twitter sentiment analysis source."""
    
    def __init__(self, keywords: List[str], sample_size: int = 1000, 
                 update_frequency: str = "hourly"):
        super().__init__("Twitter Sentiment", update_frequency)
        self.keywords = keywords
        self.sample_size = sample_size
        self.api_key = get_api_key("TWITTER_API_KEY")
        self.api_secret = get_api_key("TWITTER_API_SECRET")
        self.bearer_token = get_api_key("TWITTER_BEARER_TOKEN")
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch and analyze Twitter sentiment."""
        # For now, generate mock sentiment data since Twitter API requires authentication
        logger.warning("Using mock Twitter sentiment data - implement with actual API keys")
        return self._generate_mock_sentiment_data(start_date, end_date)
    
    def _generate_mock_sentiment_data(self, start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock sentiment data."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start_date, end_date, freq='H')
        
        np.random.seed(42)
        data = {}
        
        for keyword in self.keywords:
            # Generate sentiment scores (-1 to 1)
            sentiment_base = np.random.normal(0, 0.3, len(date_range))
            # Add some autocorrelation
            for i in range(1, len(sentiment_base)):
                sentiment_base[i] = 0.7 * sentiment_base[i-1] + 0.3 * sentiment_base[i]
            
            data[f"{keyword}_sentiment"] = np.clip(sentiment_base, -1, 1)
            data[f"{keyword}_volume"] = np.random.poisson(100, len(date_range))
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock sentiment data: {len(df)} observations")
        return df


class GoogleTrendsSource(BaseDataSource):
    """Google Trends data source."""
    
    def __init__(self, keywords: List[str], geo: str = "US", 
                 timeframe: str = "now 7-d", update_frequency: str = "daily"):
        super().__init__("Google Trends", update_frequency)
        self.keywords = keywords
        self.geo = geo
        self.timeframe = timeframe
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch Google Trends data."""
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            all_data = {}
            
            # Process keywords in batches (Google Trends API limitation)
            batch_size = 5
            for i in range(0, len(self.keywords), batch_size):
                batch_keywords = self.keywords[i:i+batch_size]
                
                try:
                    pytrends.build_payload(batch_keywords, cat=0, timeframe=self.timeframe, geo=self.geo)
                    interest_over_time = pytrends.interest_over_time()
                    
                    if not interest_over_time.empty:
                        # Remove 'isPartial' column if it exists
                        if 'isPartial' in interest_over_time.columns:
                            interest_over_time = interest_over_time.drop('isPartial', axis=1)
                        
                        for keyword in batch_keywords:
                            if keyword in interest_over_time.columns:
                                all_data[f"{keyword}_trends"] = interest_over_time[keyword]
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching trends for batch {batch_keywords}: {str(e)}")
            
            if all_data:
                df = pd.DataFrame(all_data)
                self.last_update = datetime.now()
                logger.info(f"Fetched Google Trends data: {len(df)} observations")
                return df
            else:
                return self._generate_mock_trends_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Google Trends fetch error: {str(e)}")
            return self._generate_mock_trends_data(start_date, end_date)
    
    def _generate_mock_trends_data(self, start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock Google Trends data."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        np.random.seed(42)
        data = {}
        
        for keyword in self.keywords:
            # Generate search interest (0-100)
            base_interest = np.random.uniform(20, 80)
            trend_data = base_interest + np.random.normal(0, 10, len(date_range))
            trend_data = np.clip(trend_data, 0, 100)
            data[f"{keyword}_trends"] = trend_data
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock Google Trends data: {len(df)} observations")
        return df


class WeatherDataSource(BaseDataSource):
    """OpenWeatherMap weather data source."""
    
    def __init__(self, locations: List[str], metrics: List[str] = ["temperature"],
                 update_frequency: str = "daily"):
        super().__init__("Weather", update_frequency)
        self.locations = locations
        self.metrics = metrics
        self.api_key = get_api_key("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch weather data from OpenWeatherMap."""
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not available, using mock data")
            return self._generate_mock_weather_data(start_date, end_date)
        
        all_data = {}
        
        async with aiohttp.ClientSession() as session:
            for location in self.locations:
                try:
                    # Get current weather
                    params = {
                        'q': location,
                        'appid': self.api_key,
                        'units': 'metric'
                    }
                    
                    url = f"{self.base_url}/weather"
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            location_name = location.replace(',', '_').replace(' ', '_')
                            all_data[f"{location_name}_temperature"] = data['main']['temp']
                            all_data[f"{location_name}_humidity"] = data['main']['humidity']
                            all_data[f"{location_name}_pressure"] = data['main']['pressure']
                            
                            logger.info(f"Fetched weather data for {location}")
                        else:
                            logger.error(f"Failed to fetch weather for {location}: HTTP {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching weather for {location}: {str(e)}")
        
        if all_data:
            # Create DataFrame with current timestamp
            df = pd.DataFrame([all_data], index=[datetime.now()])
            self.last_update = datetime.now()
            return df
        else:
            return self._generate_mock_weather_data(start_date, end_date)
    
    def _generate_mock_weather_data(self, start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock weather data."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        np.random.seed(42)
        data = {}
        
        for location in self.locations:
            location_name = location.replace(',', '_').replace(' ', '_')
            
            # Generate realistic weather patterns
            base_temp = np.random.uniform(15, 25)  # Celsius
            temp_data = base_temp + 10 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 365)
            temp_data += np.random.normal(0, 3, len(date_range))
            
            data[f"{location_name}_temperature"] = temp_data
            data[f"{location_name}_humidity"] = np.random.uniform(40, 80, len(date_range))
            data[f"{location_name}_pressure"] = np.random.normal(1013, 10, len(date_range))
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock weather data: {len(df)} observations")
        return df


class MockSocialSentimentSource(BaseDataSource):
    """Mock social sentiment data source for demonstration."""
    
    def __init__(self, keywords: List[str], update_frequency: str = "hourly"):
        super().__init__("Social Sentiment", update_frequency)
        self.keywords = keywords
        
    async def fetch_data(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock social sentiment data."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start_date, end_date, freq='H')
        
        np.random.seed(42)
        data = {}
        
        for keyword in self.keywords:
            # Generate sentiment with some correlation to economic events
            base_sentiment = np.random.normal(0, 0.3, len(date_range))
            
            # Add some autocorrelation and trending
            sentiment_series = np.zeros(len(date_range))
            sentiment_series[0] = base_sentiment[0]
            
            for i in range(1, len(date_range)):
                sentiment_series[i] = (0.8 * sentiment_series[i-1] + 
                                     0.2 * base_sentiment[i])
            
            # Clip to valid sentiment range
            sentiment_series = np.clip(sentiment_series, -1, 1)
            
            data[f"{keyword}_sentiment"] = sentiment_series
            data[f"{keyword}_mentions"] = np.random.poisson(50, len(date_range))
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Generated mock sentiment data: {len(df)} observations")
        return df


# Legacy alias for backward compatibility
TwitterSentimentSource = MockSocialSentimentSource