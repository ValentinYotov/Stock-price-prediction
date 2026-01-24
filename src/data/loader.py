from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import load_dataset

from src.utils.config import Config, load_config


def load_raw_dataset(
    config: Optional[Config] = None, split: Optional[str] = None
) -> pd.DataFrame:
    if config is None:
        config = load_config()
    dataset_name = config.data.dataset_name
    dataset = load_dataset(dataset_name)
    if split is None:
        split_name = list(dataset.keys())[0]
    else:
        split_name = split
    df = dataset[split_name].to_pandas()
    return df


def filter_by_tickers_and_dates(
    df: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol_column: str = "symbol",
    date_column: str = "date",
) -> pd.DataFrame:
    result = df
    if tickers is not None:
        result = result[result[symbol_column].isin(list(tickers))]
    if start_date is not None or end_date is not None:
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result = result.copy()
            result[date_column] = pd.to_datetime(result[date_column])
        if start_date is not None:
            result = result[result[date_column] >= pd.to_datetime(start_date)]
        if end_date is not None:
            result = result[result[date_column] <= pd.to_datetime(end_date)]
    return result


def load_and_filter_dataset(
    config: Optional[Config] = None,
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    split: Optional[str] = None,
) -> pd.DataFrame:
    if config is None:
        config = load_config()
    if tickers is None:
        tickers = config.data.tickers
    if start_date is None:
        start_date = config.data.start_date
    if end_date is None:
        end_date = config.data.end_date
    df = load_raw_dataset(config=config, split=split)
    df = filter_by_tickers_and_dates(
        df,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        symbol_column="symbol",
        date_column="date",
    )
    return df


__all__ = [
    "load_raw_dataset",
    "filter_by_tickers_and_dates",
    "load_and_filter_dataset",
]

