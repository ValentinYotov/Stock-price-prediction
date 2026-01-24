from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.dataset import prepare_dataset, StockDataset
from src.data.feature_engineering import create_all_features
from src.data.loader import load_and_filter_dataset
from src.data.preprocessor import preprocess_data
from src.utils.config import Config


def extract_dataset(
    config: Optional[Config] = None,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
    volume_column: str = "volume",
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> tuple[pd.DataFrame, list[str]]:
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    df = load_and_filter_dataset(config=config)
    
    df, scaler = preprocess_data(
        df,
        handle_missing=True,
        missing_method="forward_fill",
        handle_outliers_flag=True,
        outliers_method="clip",
        normalize=False,
        date_column=date_column,
        symbol_column=symbol_column,
    )
    
    df = create_all_features(
        df,
        price_column=price_column,
        high_column=high_column,
        low_column=low_column,
        volume_column=volume_column,
        date_column=date_column,
        symbol_column=symbol_column,
        windows=config.data.features.windows,
        lags=[1, 2, 3, 5, 10] if config.data.features.lag_features else [],
        add_technical=config.data.features.technical_indicators,
        add_lags=config.data.features.lag_features,
        add_temporal=config.data.features.temporal_features,
        add_volume=True,
    )
    
    df = df.dropna()
    
    feature_columns = [
        col for col in df.columns
        if col not in [date_column, symbol_column, price_column]
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]
    
    return df, feature_columns


def get_datasets(
    config: Optional[Config] = None,
    price_column: str = "close",
) -> tuple[StockDataset, StockDataset, StockDataset, list[str]]:
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    df, feature_columns = extract_dataset(config=config, price_column=price_column)
    
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        df,
        feature_columns=feature_columns,
        target_column=price_column,
        context_length=config.data.context_length,
        prediction_horizon=config.data.prediction_horizon,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )
    
    return train_dataset, val_dataset, test_dataset, feature_columns


__all__ = [
    "extract_dataset",
    "get_datasets",
]
