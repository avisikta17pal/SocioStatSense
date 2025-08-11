import os
import logging
from typing import Dict, Any

import requests
import pandas as pd
import tweepy
from dotenv import load_dotenv


# Initialize logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Load API keys from environment
TWITTER_BEARER_TOKEN: str | None = os.getenv("TWITTER_BEARER_TOKEN")
ALPHAVANTAGE_API_KEY: str | None = os.getenv("ALPHAVANTAGE_API_KEY")
OPENWEATHER_API_KEY: str | None = os.getenv("OPENWEATHER_API_KEY")


def _get_twitter_client() -> tweepy.Client:
    """Create and return a Tweepy Client using bearer token authentication.

    Raises
    ------
    RuntimeError
        If the TWITTER_BEARER_TOKEN is not set.
    """
    if not TWITTER_BEARER_TOKEN:
        raise RuntimeError("Environment variable TWITTER_BEARER_TOKEN is not set")

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    return client


def fetch_twitter_recent_tweets(keyword: str, max_results: int = 100) -> pd.DataFrame:
    """Fetch recent tweets matching keyword using Twitter API v2.

    Parameters
    ----------
    keyword : str
        Query keyword or query string compatible with Twitter search operators.
    max_results : int, optional
        Number of tweets to retrieve (1-100), by default 100.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: id, text, author_id, created_at, lang,
        retweet_count, reply_count, like_count, quote_count.

    Raises
    ------
    Exception
        Propagates Tweepy exceptions and HTTP errors.
    """
    client = _get_twitter_client()

    safe_max = max(10, min(int(max_results), 100))
    logger.info("Fetching recent tweets for keyword='%s' (max_results=%s)", keyword, safe_max)

    try:
        response = client.search_recent_tweets(
            query=keyword,
            max_results=safe_max,
            tweet_fields=[
                "id",
                "text",
                "author_id",
                "created_at",
                "lang",
                "public_metrics",
            ],
        )
    except tweepy.TooManyRequests as exc:
        logger.error("Twitter rate limit exceeded: %s", exc)
        raise
    except tweepy.TweepyException as exc:
        logger.error("Twitter API error: %s", exc)
        raise

    tweets = response.data or []
    records: list[dict[str, Any]] = []
    for tweet in tweets:
        metrics = getattr(tweet, "public_metrics", {}) or {}
        records.append(
            {
                "id": getattr(tweet, "id", None),
                "text": getattr(tweet, "text", None),
                "author_id": getattr(tweet, "author_id", None),
                "created_at": getattr(tweet, "created_at", None),
                "lang": getattr(tweet, "lang", None),
                "retweet_count": metrics.get("retweet_count"),
                "reply_count": metrics.get("reply_count"),
                "like_count": metrics.get("like_count"),
                "quote_count": metrics.get("quote_count"),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def fetch_alpha_vantage_stock(symbol: str, interval: str = "daily") -> pd.DataFrame:
    """Fetch stock time series data for a symbol using Alpha Vantage.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    interval : str, optional
        One of {'daily', 'weekly', 'monthly'}. Default is 'daily'.

    Returns
    -------
    pandas.DataFrame
        Time-indexed OHLCV DataFrame with numeric columns.

    Raises
    ------
    RuntimeError
        If the API key is missing or the API returns an error/rate-limit note.
    requests.RequestException
        For network-related errors.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("Environment variable ALPHAVANTAGE_API_KEY is not set")

    interval_lower = interval.lower().strip()
    function_map = {
        "daily": "TIME_SERIES_DAILY_ADJUSTED",
        "weekly": "TIME_SERIES_WEEKLY",
        "monthly": "TIME_SERIES_MONTHLY",
    }
    if interval_lower not in function_map:
        raise ValueError("interval must be one of {'daily', 'weekly', 'monthly'}")

    params: Dict[str, Any] = {
        "function": function_map[interval_lower],
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": "compact",
    }

    url = "https://www.alphavantage.co/query"
    logger.info("Fetching Alpha Vantage %s time series for symbol='%s'", interval_lower, symbol)

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Alpha Vantage HTTP error: %s", exc)
        raise

    payload = response.json()

    # Handle rate limits and API errors
    if isinstance(payload, dict) and payload.get("Note"):
        message = payload.get("Note")
        logger.error("Alpha Vantage rate limit hit: %s", message)
        raise RuntimeError(f"Alpha Vantage rate limit: {message}")
    if isinstance(payload, dict) and payload.get("Error Message"):
        message = payload.get("Error Message")
        logger.error("Alpha Vantage returned error: %s", message)
        raise RuntimeError(message)

    # Find the time series key dynamically
    time_series_key = next((k for k in payload.keys() if "Time Series" in k), None)
    if not time_series_key:
        logger.error("Unexpected Alpha Vantage response format: keys=%s", list(payload.keys()))
        raise RuntimeError("Unexpected Alpha Vantage response format")

    series = payload[time_series_key]
    frame = pd.DataFrame.from_dict(series, orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    # Normalize column names and types
    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "5. volume": "volume",
        "6. volume": "volume",
        "6. dividend amount": "dividend_amount",
        "7. dividend amount": "dividend_amount",
        "8. split coefficient": "split_coefficient",
    }
    frame = frame.rename(columns=rename_map)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def fetch_openweather_current(city: str) -> Dict[str, Any]:
    """Fetch current weather data for a city using OpenWeatherMap API.

    Parameters
    ----------
    city : str
        City name, optionally with country code (e.g., 'Paris,FR').

    Returns
    -------
    dict
        Parsed JSON response from OpenWeatherMap for the current weather.

    Raises
    ------
    RuntimeError
        If the API key is missing or the API returns an error.
    requests.RequestException
        For network-related errors.
    """
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("Environment variable OPENWEATHER_API_KEY is not set")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    logger.info("Fetching current weather for city='%s'", city)

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("OpenWeatherMap HTTP error: %s", exc)
        raise

    data = response.json()
    # OpenWeather can return cod as int or string
    cod = data.get("cod")
    if (isinstance(cod, int) and cod != 200) or (isinstance(cod, str) and cod != "200"):
        message = data.get("message", "Unknown error")
        logger.error("OpenWeatherMap API error (cod=%s): %s", cod, message)
        raise RuntimeError(f"OpenWeatherMap error (cod={cod}): {message}")

    return data