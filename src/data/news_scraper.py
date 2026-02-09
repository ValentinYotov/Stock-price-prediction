"""
Web scraper for historical financial news.

This module provides functions to scrape historical news from various sources.
Note: Always respect robots.txt and terms of service.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_yahoo_finance_news_historical(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    delay: float = 1.0,  # Delay between requests to be respectful
) -> List[Dict[str, Any]]:
    """
    Scrape historical news from Yahoo Finance.
    
    Note: This uses web scraping. Be respectful:
    - Add delays between requests
    - Respect robots.txt
    - Don't overload their servers
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for news
        end_date: End date for news
        delay: Delay in seconds between requests
    
    Returns:
        List of news dictionaries
    """
    # Yahoo Finance news URL structure
    # Note: Yahoo Finance doesn't have a direct historical news API
    # This would require scraping their news pages
    
    print(f"⚠️  Yahoo Finance historical scraping is complex.")
    print(f"   Yahoo Finance doesn't provide easy historical news access.")
    print(f"   Consider using alternative sources for historical data.")
    
    # For now, return empty - would need to implement actual scraping
    return []


def scrape_seeking_alpha_news(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    delay: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Scrape news from Seeking Alpha.
    
    Note: Always check robots.txt and terms of service before scraping.
    Consider using their RSS feed instead if available.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        delay: Delay between requests
    
    Returns:
        List of news dictionaries
    """
    print(f"⚠️  Seeking Alpha scraping requires careful implementation.")
    print(f"   Check their terms of service and robots.txt first.")
    print(f"   Consider using RSS feeds if available.")
    
    return []


def get_rss_feed_news(
    ticker: str,
    source: str = "yahoo",
) -> List[Dict[str, Any]]:
    """
    Get news from RSS feeds (more reliable than scraping).
    
    Args:
        ticker: Stock ticker symbol
        source: Source name ('yahoo', 'google', etc.)
    
    Returns:
        List of news dictionaries
    """
    import feedparser
    
    # Yahoo Finance RSS feed
    if source == "yahoo":
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    
    try:
        feed = feedparser.parse(url)
        
        news_items = []
        for entry in feed.entries:
            news_item = {
                'ticker': ticker,
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published_time': None,
                'summary': entry.get('summary', ''),
                'source': 'yahoo_rss',
            }
            
            # Parse published date
            if 'published_parsed' in entry:
                try:
                    news_item['published_time'] = datetime(*entry.published_parsed[:6])
                except:
                    pass
            
            news_items.append(news_item)
        
        return news_items
        
    except Exception as e:
        print(f"❌ Error fetching RSS feed: {e}")
        return []


def fetch_historical_news_combined(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    sources: List[str] = ["yahoo_rss", "yfinance"],
) -> pd.DataFrame:
    """
    Fetch historical news from multiple sources.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        sources: List of sources to try
    
    Returns:
        DataFrame with news data
    """
    all_news = []
    
    # Try yfinance first (easiest, but limited history)
    if "yfinance" in sources:
        try:
            from src.data.news_loader import fetch_yahoo_finance_news
            print(f"Fetching recent news from Yahoo Finance (yfinance)...")
            yf_news = fetch_yahoo_finance_news(ticker, limit=None)
            
            if yf_news:
                # Filter by date
                filtered_news = [
                    n for n in yf_news
                    if start_date <= n.get('published_time', datetime.min) <= end_date
                ]
                print(f"  Found {len(filtered_news)} news items in date range")
                all_news.extend(filtered_news)
        except Exception as e:
            print(f"  ⚠️  Error with yfinance: {e}")
    
    # Try RSS feeds
    if "yahoo_rss" in sources:
        try:
            print(f"Fetching news from Yahoo Finance RSS feed...")
            rss_news = get_rss_feed_news(ticker, source="yahoo")
            
            if rss_news:
                # Filter by date
                filtered_news = [
                    n for n in rss_news
                    if n.get('published_time') and start_date <= n['published_time'] <= end_date
                ]
                print(f"  Found {len(filtered_news)} news items from RSS")
                all_news.extend(filtered_news)
        except Exception as e:
            print(f"  ⚠️  Error with RSS: {e}")
    
    # Remove duplicates (by title or link)
    seen = set()
    unique_news = []
    for item in all_news:
        identifier = item.get('link') or item.get('title', '')
        if identifier and identifier not in seen:
            seen.add(identifier)
            unique_news.append(item)
    
    print(f"\n✅ Total unique news items: {len(unique_news)}")
    
    if unique_news:
        df = pd.DataFrame(unique_news)
        df = df.sort_values('published_time', ascending=False)
        return df
    else:
        return pd.DataFrame()


__all__ = [
    'get_rss_feed_news',
    'fetch_historical_news_combined',
]
