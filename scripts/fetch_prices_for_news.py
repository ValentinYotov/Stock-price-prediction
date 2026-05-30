"""
Fetch daily prices that overlap the news window.

The bundled HuggingFace price file covers 2005-2020, while the news file covers
2021-2023, so the two never overlap. To study news impact we need prices for the
news period. This script downloads daily OHLCV via yfinance for the tickers that
have news coverage and saves them in the same schema the pipeline expects:

    columns: date, symbol, open, high, low, close, volume

Run:
    python scripts/fetch_prices_for_news.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yfinance as yf

from src.data.news_sentiment import DEFAULT_SENTIMENT_PATH, load_daily_sentiment
from src.utils.config import load_config

# A few months of lead before the first news day so rolling indicators and the
# context window are valid by the time news coverage begins.
FETCH_START = "2021-06-01"
FETCH_END = "2023-03-15"
OUTPUT_PATH = Path("data/raw/prices_news_window.parquet")


def main() -> None:
    config = load_config()

    sentiment = load_daily_sentiment(config=config)
    tickers = sorted(sentiment["symbol"].unique())

    print("=" * 70)
    print("PRICE FETCHER (news window)")
    print("=" * 70)
    print(f"Tickers: {tickers}")
    print(f"Window:  {FETCH_START} -> {FETCH_END}")
    print("=" * 70)

    raw = yf.download(
        tickers,
        start=FETCH_START,
        end=FETCH_END,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            print(f"  ! no data for {ticker}")
            continue
        sub = raw[ticker].reset_index()
        sub = sub.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        sub["symbol"] = ticker
        sub = sub.dropna(subset=["close"])
        frames.append(sub[["date", "symbol", "open", "high", "low", "close", "volume"]])
        print(f"  {ticker:<6} {len(sub):>4} rows")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Total rows: {len(prices):,}")
    print(f"Symbols:    {prices['symbol'].nunique()}")
    print(f"Date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")
    print(f"Saved to:   {OUTPUT_PATH.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
