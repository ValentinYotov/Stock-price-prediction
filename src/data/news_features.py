"""
News features extraction and caching module.

This module handles:
1. Fetching news from StockData API
2. Encoding news with FinBERT
3. Caching news embeddings for efficiency
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
import numpy as np

from src.data.news_loader import fetch_stockdata_news
from src.models.news_encoder import FinBERTEncoder


class NewsFeatureExtractor:
    """
    Extracts and caches news embeddings for stock data.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        encoder: Optional[FinBERTEncoder] = None,
    ):
        """
        Initialize news feature extractor.
        
        Args:
            api_key: StockData API key (or from env var)
            cache_dir: Directory to cache news embeddings
            encoder: FinBERT encoder (will create if None)
        """
        # Get API key
        if api_key is None:
            api_key = os.getenv('STOCKDATA_API_KEY')
        if not api_key:
            raise ValueError(
                "StockData API key required. Set STOCKDATA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.api_key = api_key
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path("data/processed/news_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        if encoder is None:
            print("Loading FinBERT encoder...")
            self.encoder = FinBERTEncoder()
        else:
            self.encoder = encoder
    
    def get_cache_path(self, ticker: str, start_date: datetime, end_date: datetime) -> Path:
        """Get cache file path for a ticker and date range."""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        filename = f"{ticker}_{start_str}_{end_str}.pkl"
        return self.cache_dir / filename
    
    def fetch_and_encode_news(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch news for a ticker and encode with FinBERT.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for news
            end_date: End date for news
            use_cache: Whether to use cached embeddings
            force_refresh: Force refresh even if cache exists
        
        Returns:
            DataFrame with columns: date, ticker, news_embedding_0, ..., news_embedding_767
        """
        cache_path = self.get_cache_path(ticker, start_date, end_date)
        
        # Check cache
        if use_cache and not force_refresh and cache_path.exists():
            print(f"📦 Loading cached news embeddings for {ticker} from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Fetch news
        print(f"📰 Fetching news for {ticker} from {start_date.date()} to {end_date.date()}...")
        news_list = fetch_stockdata_news(
            ticker=ticker,
            api_key=self.api_key,
            start_date=start_date,
            end_date=end_date,
            limit=100,  # Max for free tier is 2, but we request more
        )
        
        if not news_list:
            print(f"⚠️  No news found for {ticker}")
            return self._create_empty_news_df(ticker, start_date, end_date)
        
        # Convert to DataFrame
        news_df = pd.DataFrame(news_list)
        
        # Group news by date (one embedding per day)
        news_df['date'] = pd.to_datetime(news_df['published_time']).dt.date
        
        # Combine news text (title + description)
        news_df['text'] = (
            news_df['title'].fillna('') + ' ' + 
            news_df.get('description', pd.Series('')).fillna('')
        ).str.strip()
        
        # Group by date and combine texts
        daily_news = news_df.groupby('date')['text'].apply(
            lambda x: ' '.join(x.astype(str))
        ).reset_index()
        
        # Encode news with FinBERT
        print(f"🔢 Encoding {len(daily_news)} days of news with FinBERT...")
        texts = daily_news['text'].tolist()
        embeddings = self.encoder.encode_text(texts)
        
        # Create DataFrame with embeddings
        embedding_cols = [f'news_embedding_{i}' for i in range(embeddings.shape[1])]
        result_df = pd.DataFrame(embeddings, columns=embedding_cols)
        result_df['date'] = pd.to_datetime(daily_news['date'])
        result_df['ticker'] = ticker
        
        # Reorder columns
        result_df = result_df[['date', 'ticker'] + embedding_cols]
        
        # Cache result
        if use_cache:
            print(f"💾 Caching news embeddings to {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(result_df, f)
        
        return result_df
    
    def _create_empty_news_df(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create empty news DataFrame with correct structure."""
        embedding_cols = [f'news_embedding_{i}' for i in range(768)]
        df = pd.DataFrame(columns=['date', 'ticker'] + embedding_cols)
        df['date'] = pd.to_datetime([])
        df['ticker'] = ''
        df[embedding_cols] = 0.0
        return df
    
    def get_news_for_dataframe(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        symbol_column: str = "ticker",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get news embeddings for a DataFrame (returns only news embeddings, not merged).
        
        Args:
            df: DataFrame with date and ticker columns
            date_column: Name of date column
            symbol_column: Name of ticker/symbol column
            use_cache: Whether to use cache
        
        Returns:
            DataFrame with columns: date, ticker, news_embedding_0, ..., news_embedding_767
            (Not merged with input df - this is done separately in prepare_dataset_with_news)
        """
        # Get unique tickers and date range
        tickers = df[symbol_column].unique()
        dates = pd.to_datetime(df[date_column])
        start_date = dates.min().to_pydatetime()
        end_date = dates.max().to_pydatetime()
        
        # Fetch news for each ticker
        all_news = []
        for ticker in tickers:
            ticker_news = self.fetch_and_encode_news(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )
            all_news.append(ticker_news)
        
        if not all_news:
            return self._create_empty_news_df('', start_date, end_date)
        
        # Combine all tickers
        news_df = pd.concat(all_news, ignore_index=True)
        
        # Return only news embeddings DataFrame (not merged with input df)
        # This will be merged later in prepare_dataset_with_news
        return news_df


__all__ = [
    "NewsFeatureExtractor",
]
