
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import Config, load_config

# Columns added to the feature frame when news is enabled.
NEWS_FEATURE_COLUMNS = [
    "news_compound",
    "news_pos",
    "news_neg",
    "news_neu",
    "news_count",
    "news_has",
]

DEFAULT_SENTIMENT_PATH = Path("data/news/daily_sentiment_2015_2020.parquet")


def load_daily_sentiment(
    config: Optional[Config] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the cached FinBERT daily-sentiment table."""
    if path is None:
        if config is None:
            config = load_config()
        path = Path(getattr(config.data, "news_sentiment_file", DEFAULT_SENTIMENT_PATH))

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment table not found at {path}. Build it with:\n"
            "  py -3.11 scripts/fetch_fnspid_news.py\n"
            "  py -3.11 scripts/score_news_finbert.py"
        )

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_sentiment_features(
    df: pd.DataFrame,
    sentiment: pd.DataFrame,
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """
    Left-merge daily sentiment onto a price/feature frame.

    Days without news get neutral/zero sentiment so no rows are dropped.
    """
    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    sent = sentiment.copy()
    sent[date_column] = pd.to_datetime(sent[date_column])

    result = result.merge(sent, on=[symbol_column, date_column], how="left")

    fill_values = {
        "news_compound": 0.0,
        "news_pos": 0.0,
        "news_neg": 0.0,
        "news_neu": 0.0,
        "news_count": 0.0,
        "news_has": 0.0,
    }
    for col, value in fill_values.items():
        if col in result.columns:
            result[col] = result[col].fillna(value)

    return result


__all__ = [
    "NEWS_FEATURE_COLUMNS",
    "DEFAULT_SENTIMENT_PATH",
    "load_daily_sentiment",
    "merge_sentiment_features",
]
