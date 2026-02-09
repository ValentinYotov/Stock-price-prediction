"""
Script to fetch historical news for all tickers in the dataset.

This script:
1. Loads tickers from config
2. Fetches news for each ticker for the date range in config
3. Saves news to parquet files
4. Handles rate limits and errors gracefully
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time

from src.data.news_loader import fetch_yahoo_finance_news, news_to_dataframe
from src.data.news_scraper import fetch_historical_news_combined
from src.utils.config import load_config


def main():
    """Main function to fetch historical news."""
    config = load_config()
    
    # Get tickers
    tickers = config.data.tickers
    start_date = datetime.strptime(config.data.start_date, "%Y-%m-%d") if config.data.start_date else datetime(2010, 1, 1)
    end_date = datetime.now() if config.data.end_date is None else datetime.strptime(config.data.end_date, "%Y-%m-%d")
    
    print("=" * 80)
    print("HISTORICAL NEWS FETCHER")
    print("=" * 80)
    print(f"Tickers: {tickers}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("data/news")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_news = []
    
    for ticker in tqdm(tickers, desc="Fetching news"):
        print(f"\n{'='*80}")
        print(f"Processing {ticker}...")
        print(f"{'='*80}")
        
        try:
            # Try multiple sources
            df_news = fetch_historical_news_combined(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                sources=["yfinance", "yahoo_rss"],  # Start with free sources
            )
            
            if len(df_news) > 0:
                print(f"✅ Fetched {len(df_news)} news items for {ticker}")
                
                # Save individual ticker file
                ticker_file = output_dir / f"{ticker}_news.parquet"
                df_news.to_parquet(ticker_file, index=False)
                print(f"   Saved to: {ticker_file}")
                
                # Add to combined list
                all_news.append(df_news)
            else:
                print(f"⚠️  No news found for {ticker} in date range")
            
            # Be respectful - add delay
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue
    
    # Save combined file
    if all_news:
        combined_df = pd.concat(all_news, ignore_index=True)
        combined_file = output_dir / "all_news.parquet"
        combined_df.to_parquet(combined_file, index=False)
        print(f"\n✅ Combined news saved to: {combined_file}")
        print(f"   Total news items: {len(combined_df)}")
        print(f"   Date range: {combined_df['published_time'].min()} to {combined_df['published_time'].max()}")
    else:
        print("\n⚠️  No news data collected")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
