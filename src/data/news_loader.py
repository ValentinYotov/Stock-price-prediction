"""
News data loader module.

This module provides functions to fetch financial news from various sources.

API keys are loaded from environment variables (.env file).
See .env.example for template.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import time

import pandas as pd

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass


def fetch_yahoo_finance_news(
    ticker: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch news from Yahoo Finance using yfinance library.
    
    This is a free option with no API key required.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        limit: Maximum number of news items to return (None = all available)
    
    Returns:
        List of news dictionaries with keys:
        - 'title': News headline
        - 'publisher': Publisher name
        - 'providerPublishTime': Unix timestamp
        - 'link': URL to full article
        - 'uuid': Unique identifier
        - 'type': News type
        - 'relatedTickers': List of related tickers
    
    Example:
        >>> news = fetch_yahoo_finance_news('AAPL', limit=10)
        >>> print(f"Found {len(news)} news items")
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for Yahoo Finance news. "
            "Install with: pip install yfinance"
        )
    
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        # Apply limit if specified
        if limit is not None:
            news = news[:limit]
        
        # Convert to standardized format
        standardized_news = []
        for item in news:
            standardized_item = {
                'ticker': ticker,
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'published_time': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                'link': item.get('link', ''),
                'uuid': item.get('uuid', ''),
                'type': item.get('type', ''),
                'related_tickers': item.get('relatedTickers', []),
            }
            standardized_news.append(standardized_item)
        
        return standardized_news
        
    except Exception as e:
        raise RuntimeError(f"Error fetching Yahoo Finance news for {ticker}: {e}")


def fetch_alpha_vantage_news(
    ticker: str,
    api_key: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch news from Alpha Vantage News API.
    
    Requires free API key from https://www.alphavantage.co/support/#api-key
    Rate limit: 5 calls per minute, 500 calls per day
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        api_key: Alpha Vantage API key.
                 If None, will try to load from ALPHA_VANTAGE_API_KEY environment variable.
        limit: Maximum number of news items (default: 50)
    
    Returns:
        List of news dictionaries with sentiment scores
    
    Example:
        >>> news = fetch_alpha_vantage_news('AAPL', api_key='YOUR_KEY', limit=10)
    """
    import os
    import requests
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key not provided. "
                "Either pass api_key parameter or set ALPHA_VANTAGE_API_KEY environment variable. "
                "Get free API key at: https://www.alphavantage.co/support/#api-key"
            )
    
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'limit': limit,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'Information' in data:
            raise ValueError(f"API Info: {data['Information']}")
        
        if 'feed' not in data:
            raise ValueError(f"Unexpected response format: {list(data.keys())}")
        
        articles = data['feed']
        
        # Convert to standardized format
        standardized_news = []
        for article in articles:
            standardized_item = {
                'ticker': ticker,
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'published_time': datetime.fromisoformat(
                    article.get('time_published', '').replace('T', ' ')
                ),
                'url': article.get('url', ''),
                'summary': article.get('summary', ''),
                'sentiment_score': article.get('overall_sentiment_score', None),
                'sentiment_label': article.get('overall_sentiment_label', ''),
                'relevance_score': article.get('relevance_score', None),
            }
            standardized_news.append(standardized_item)
        
        return standardized_news
        
    except Exception as e:
        raise RuntimeError(f"Error fetching Alpha Vantage news for {ticker}: {e}")


def fetch_newsapi_news(
    ticker: str,
    api_key: Optional[str] = None,
    days_back: int = 7,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch news from NewsAPI.
    
    Requires free API key from https://newsapi.org/register
    Rate limit: 100 requests per day (free tier)
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        api_key: NewsAPI key.
                 If None, will try to load from NEWSAPI_KEY environment variable.
        days_back: How many days back to fetch (max 1 month for free tier)
        limit: Maximum number of articles (default: 100)
    
    Returns:
        List of news dictionaries
    """
    import os
    import requests
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            raise ValueError(
                "NewsAPI key not provided. "
                "Either pass api_key parameter or set NEWSAPI_KEY environment variable. "
                "Get free API key at: https://newsapi.org/register"
            )
    
    url = 'https://newsapi.org/v2/everything'
    
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    params = {
        'q': ticker,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': min(limit, 100),  # Max 100 per request
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
        
        articles = data.get('articles', [])
        
        # Convert to standardized format
        standardized_news = []
        for article in articles:
            published_time = None
            if article.get('publishedAt'):
                try:
                    published_time = datetime.fromisoformat(
                        article['publishedAt'].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            standardized_item = {
                'ticker': ticker,
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', ''),
                'published_time': published_time,
                'url': article.get('url', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
            }
            standardized_news.append(standardized_item)
        
        return standardized_news
        
    except Exception as e:
        raise RuntimeError(f"Error fetching NewsAPI news for {ticker}: {e}")


def news_to_dataframe(news_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of news dictionaries to pandas DataFrame.
    
    Args:
        news_list: List of news dictionaries
    
    Returns:
        DataFrame with standardized columns
    """
    if not news_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(news_list)
    
    # Ensure published_time is datetime
    if 'published_time' in df.columns:
        df['published_time'] = pd.to_datetime(df['published_time'])
        df['date'] = df['published_time'].dt.date
    
    # Sort by published time (newest first)
    if 'published_time' in df.columns:
        df = df.sort_values('published_time', ascending=False).reset_index(drop=True)
    
    return df


def fetch_stockdata_news(
    ticker: str,
    api_key: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch news from StockData.org API.
    
    Free tier: 100 requests/day, 7+ years historical data
    Get API key at: https://stockdata.org/
    
    Args:
        ticker: Stock ticker symbol
        api_key: StockData.org API key (free registration).
                 If None, will try to load from STOCKDATA_API_KEY environment variable.
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        limit: Maximum number of news items
    
    Returns:
        List of news dictionaries
    """
    import os
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('STOCKDATA_API_KEY')
        if not api_key:
            raise ValueError(
                "StockData.org API key not provided. "
                "Either pass api_key parameter or set STOCKDATA_API_KEY environment variable. "
                "Get free API key at: https://stockdata.org/"
            )
    import requests
    
    url = 'https://api.stockdata.org/v1/news/all'
    
    params = {
        'symbols': ticker,
        'api_token': api_key,
        'limit': limit,
    }
    
    # Add date filters if provided
    if start_date:
        params['date_from'] = start_date.strftime('%Y-%m-%d')
    if end_date:
        params['date_to'] = end_date.strftime('%Y-%m-%d')
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if 'data' in data:
            articles = data['data']
        elif isinstance(data, list):
            articles = data
        else:
            # Check for error messages
            if 'message' in data or 'error' in data:
                error_msg = data.get('message') or data.get('error', 'Unknown error')
                raise ValueError(f"API Error: {error_msg}")
            raise ValueError(f"Unexpected response format: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        # Free tier limitation: only 2 articles per request
        # Note: This is a limitation of the free tier
        # Convert to standardized format
        standardized_news = []
        for article in articles:
            published_time = None
            if article.get('published_at'):
                try:
                    published_time = datetime.fromisoformat(
                        article['published_at'].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            standardized_item = {
                'ticker': ticker,
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'published_time': published_time,
                'url': article.get('url', ''),
                'description': article.get('description', ''),
                'snippet': article.get('snippet', ''),
                'image_url': article.get('image_url', ''),
                'language': article.get('language', 'en'),
            }
            standardized_news.append(standardized_item)
        
        return standardized_news
        
    except Exception as e:
        raise RuntimeError(f"Error fetching StockData.org news for {ticker}: {e}")


__all__ = [
    'fetch_yahoo_finance_news',
    'fetch_alpha_vantage_news',
    'fetch_newsapi_news',
    'fetch_stockdata_news',
    'news_to_dataframe',
]
