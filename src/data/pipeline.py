from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.dataset import prepare_dataset, StockDataset, time_series_split
from src.data.feature_engineering import create_all_features, add_forward_log_return
from src.data.loader import load_and_filter_dataset
from src.data.preprocessor import preprocess_data
from src.utils.config import Config, split_per_symbol


def _exclude_columns_for_features(
    df: pd.DataFrame,
    date_column: str,
    symbol_column: str,
    price_column: str,
    target_column: str,
) -> list[str]:
    excluded = {date_column, symbol_column, price_column, target_column}
    return [
        col
        for col in df.columns
        if col not in excluded
        and df[col].dtype in ["float64", "int64", "float32", "int32"]
    ]


def extract_dataset(
    config: Optional[Config] = None,
    price_column: Optional[str] = None,
    high_column: str = "high",
    low_column: str = "low",
    volume_column: str = "volume",
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> tuple[pd.DataFrame, list[str]]:
    if config is None:
        from src.utils.config import load_config
        config = load_config()

    if price_column is None:
        price_column = config.data.price_column

    target_column = config.data.target_column

    df = load_and_filter_dataset(config=config)

    if config.data.hero_ticker:
        df = df[df[symbol_column] == config.data.hero_ticker]

    df, _scaler = preprocess_data(
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
        simplified=config.data.features.simplified,
    )

    if target_column == "log_return":
        df = add_forward_log_return(
            df,
            price_column=price_column,
            target_column=target_column,
            symbol_column=symbol_column,
        )
    elif target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not in dataframe; "
            "use 'log_return' or ensure the column exists after feature engineering"
        )

    df = df.dropna()

    feature_columns = _exclude_columns_for_features(
        df, date_column, symbol_column, price_column, target_column
    )

    if config.data.features.normalize:
        if config.data.features.normalize_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        per_symbol = split_per_symbol(config)
        train_df, _, _ = time_series_split(
            df,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            test_split=config.data.test_split,
            date_column=date_column,
            symbol_column=symbol_column,
            per_symbol=per_symbol,
        )

        cols_to_scale = list(feature_columns)
        if config.data.features.normalize_target and target_column in df.columns:
            cols_to_scale = cols_to_scale + [target_column]

        scaler.fit(train_df[cols_to_scale])
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, feature_columns


def get_datasets(
    config: Optional[Config] = None,
    price_column: Optional[str] = None,
) -> tuple[StockDataset, StockDataset, StockDataset, list[str]]:
    if config is None:
        from src.utils.config import load_config
        config = load_config()

    if price_column is None:
        price_column = config.data.price_column

    target_column = config.data.target_column

    df, feature_columns = extract_dataset(config=config, price_column=price_column)

    train_dataset, val_dataset, test_dataset = prepare_dataset(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
        context_length=config.data.context_length,
        prediction_horizon=config.data.prediction_horizon,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        per_symbol=split_per_symbol(config),
    )

    return train_dataset, val_dataset, test_dataset, feature_columns


__all__ = [
    "extract_dataset",
    "get_datasets",
]
