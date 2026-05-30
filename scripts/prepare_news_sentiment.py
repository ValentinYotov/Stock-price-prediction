"""
Materialize the daily news-sentiment table used by the news-enhanced model.

Reads `sp500_news_290k_articles.csv` from the project HuggingFace dataset,
aggregates per-article VADER sentiment to one row per (symbol, date) for the
configured ticker universe, and writes data/news/daily_sentiment.parquet.

Run:
    python scripts/prepare_news_sentiment.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.news_sentiment import DEFAULT_SENTIMENT_PATH, build_daily_sentiment
from src.utils.config import load_config


def main() -> None:
    config = load_config()

    print("=" * 70)
    print("DAILY NEWS SENTIMENT BUILDER")
    print("=" * 70)
    print(f"Dataset:  {config.data.dataset_name}")
    print(f"Tickers:  {len(config.data.tickers)} symbols")
    print(f"Output:   {DEFAULT_SENTIMENT_PATH}")
    print("=" * 70)

    df = build_daily_sentiment(config=config, save=True)

    print(f"Rows (symbol-day):   {len(df):,}")
    print(f"Symbols covered:     {df['symbol'].nunique()}")
    print(f"Date range:          {df['date'].min().date()} -> {df['date'].max().date()}")
    print()
    print("Articles per symbol:")
    counts = df.groupby("symbol")["news_count"].sum().sort_values(ascending=False)
    for symbol, total in counts.items():
        print(f"  {symbol:<6} {int(total):>7,} articles  ({(df['symbol'] == symbol).sum()} days)")
    print()
    print(f"Saved to: {DEFAULT_SENTIMENT_PATH.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
