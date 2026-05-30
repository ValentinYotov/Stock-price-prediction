"""
Daily news sentiment features from the project HuggingFace dataset.

The dataset `pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs` ships a file
`sp500_news_290k_articles.csv` that already contains per-article VADER sentiment
(`compound`, `neg`, `neu`, `pos`). This module aggregates those articles to one
row per (symbol, date) so the scores can be merged into the technical feature
frame as ordinary model inputs.

No scraping and no external API keys are needed.
"""
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

NEWS_CSV_FILENAME = "sp500_news_290k_articles.csv"
DEFAULT_SENTIMENT_PATH = Path("data/news/daily_sentiment.parquet")


def _download_news_csv(dataset_name: str) -> Path:
    """Fetch the raw news CSV from the HuggingFace dataset repo (cached)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=dataset_name,
        filename=NEWS_CSV_FILENAME,
        repo_type="dataset",
    )
    return Path(path)


def build_daily_sentiment(
    config: Optional[Config] = None,
    save: bool = True,
    output_path: Path = DEFAULT_SENTIMENT_PATH,
) -> pd.DataFrame:
    """
    Build a daily per-symbol sentiment table from the raw 290k-article CSV.

    Returns a DataFrame with columns:
        symbol, date, news_compound, news_pos, news_neg, news_neu,
        news_count, news_has
    """
    if config is None:
        config = load_config()

    csv_path = _download_news_csv(config.data.dataset_name)
    raw = pd.read_csv(csv_path)

    # Parse the 'Mon-DD-YY' date format; drop rows with unparseable dates.
    raw["date"] = pd.to_datetime(raw["date"], format="%b-%d-%y", errors="coerce")
    raw = raw.dropna(subset=["date"])

    raw = raw.rename(columns={"ticker": "symbol"})

    # Keep only tickers in the configured universe (smaller, faster, relevant).
    tickers = set(config.data.tickers)
    if tickers:
        raw = raw[raw["symbol"].isin(tickers)]

    for col in ["compound", "neg", "neu", "pos"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["compound", "neg", "neu", "pos"])

    grouped = (
        raw.groupby(["symbol", "date"])
        .agg(
            news_compound=("compound", "mean"),
            news_pos=("pos", "mean"),
            news_neg=("neg", "mean"),
            news_neu=("neu", "mean"),
            news_count=("compound", "size"),
        )
        .reset_index()
    )
    grouped["news_has"] = 1.0
    grouped["news_count"] = grouped["news_count"].astype(float)

    grouped = grouped.sort_values(["symbol", "date"]).reset_index(drop=True)

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_parquet(output_path, index=False)

    return grouped


def load_daily_sentiment(
    config: Optional[Config] = None,
    path: Path = DEFAULT_SENTIMENT_PATH,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load the cached daily sentiment table, building it on first use.
    """
    if not rebuild and Path(path).exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return build_daily_sentiment(config=config, save=True, output_path=Path(path))


def merge_sentiment_features(
    df: pd.DataFrame,
    sentiment: pd.DataFrame,
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """
    Left-merge daily sentiment onto a price/feature frame.

    Days without news get neutral/zero sentiment so no rows are dropped. Using
    same-day sentiment to predict the next-day return introduces no look-ahead:
    a row at day t only uses news published up to day t.
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
    "build_daily_sentiment",
    "load_daily_sentiment",
    "merge_sentiment_features",
]
