"""
Enhanced data pipeline with news embeddings support.

The base pipeline remains unchanged for the base model.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.pipeline import extract_dataset
from src.data.dataset_with_news import prepare_dataset_with_news
from src.data.news_features import NewsFeatureExtractor
from src.utils.config import Config


def extract_dataset_with_news(
    config: Optional[Config] = None,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
    volume_column: str = "volume",
    date_column: str = "date",
    symbol_column: str = "symbol",
    use_news_cache: bool = True,
    force_refresh_news: bool = False,
) -> tuple[pd.DataFrame, list[str], Optional[pd.DataFrame]]:
    """
    Extract dataset with news embeddings.
    
    Args:
        config: Configuration object
        price_column: Price column name
        high_column: High price column name
        low_column: Low price column name
        volume_column: Volume column name
        date_column: Date column name
        symbol_column: Symbol/ticker column name
        use_news_cache: Whether to use cached news embeddings
        force_refresh_news: Force refresh news even if cached
    
    Returns:
        Tuple of (df, feature_columns, news_embeddings_df)
    """
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    # Extract base dataset (technical features)
    df, feature_columns = extract_dataset(
        config=config,
        price_column=price_column,
        high_column=high_column,
        low_column=low_column,
        volume_column=volume_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )
    
    # Extract news embeddings
    news_embeddings_df = None
    if getattr(config.data, 'use_news', False):
        extractor = NewsFeatureExtractor()
        news_embeddings_df = extractor.get_news_for_dataframe(
            df,
            date_column=date_column,
            symbol_column=symbol_column,
            use_cache=use_news_cache,
        )
    
    return df, feature_columns, news_embeddings_df


def get_datasets_with_news(
    config: Optional[Config] = None,
    price_column: str = "close",
    use_news_cache: bool = True,
    force_refresh_news: bool = False,
) -> tuple:
 
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    # Extract dataset with news
    df, feature_columns, news_embeddings_df = extract_dataset_with_news(
        config=config,
        price_column=price_column,
        use_news_cache=use_news_cache,
        force_refresh_news=force_refresh_news,
    )
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_dataset_with_news(
        df,
        feature_columns=feature_columns,
        target_column=price_column,
        news_embeddings_df=news_embeddings_df,
        context_length=config.data.context_length,
        prediction_horizon=config.data.prediction_horizon,
    )
    
    return train_dataset, val_dataset, test_dataset, feature_columns


__all__ = [
    "extract_dataset_with_news",
    "get_datasets_with_news",
]
