from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calculate_sma(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return df[column].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return df[column].ewm(span=window, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, column: str, window: int = 14) -> pd.Series:
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    df: pd.DataFrame, column: str, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calculate_ema(df, column, fast)
    ema_slow = calculate_ema(df, column, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(
    df: pd.DataFrame, column: str, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = calculate_sma(df, column, window)
    std = df[column].rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def calculate_atr(
    df: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int = 14
) -> pd.Series:
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def add_moving_averages(
    df: pd.DataFrame,
    column: str = "close",
    windows: list[int] = [5, 10, 20, 50],
    sma: bool = True,
    ema: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    
    for window in windows:
        if sma:
            result[f"sma_{window}"] = calculate_sma(result, column, window)
        if ema:
            result[f"ema_{window}"] = calculate_ema(result, column, window)
    
    return result


def add_rsi(df: pd.DataFrame, column: str = "close", window: int = 14) -> pd.DataFrame:
    result = df.copy()
    result[f"rsi_{window}"] = calculate_rsi(result, column, window)
    return result


def add_macd(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    result = df.copy()
    macd, signal, histogram = calculate_macd(result, column)
    result["macd"] = macd
    result["macd_signal"] = signal
    result["macd_histogram"] = histogram
    return result


def add_bollinger_bands(
    df: pd.DataFrame, column: str = "close", window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    result = df.copy()
    upper, middle, lower = calculate_bollinger_bands(result, column, window, num_std)
    result[f"bb_upper_{window}"] = upper
    result[f"bb_middle_{window}"] = middle
    result[f"bb_lower_{window}"] = lower
    result[f"bb_width_{window}"] = upper - lower
    result[f"bb_position_{window}"] = (result[column] - lower) / (upper - lower)
    return result


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 2, 3, 5, 10],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    result = df.copy()
    
    if group_by:
        for col in columns:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = result.groupby(group_by)[col].shift(lag)
    else:
        for col in columns:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)
    
    return result


def add_temporal_features(
    df: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    result = df.copy()
    
    if date_column not in result.columns:
        return result
    
    result[date_column] = pd.to_datetime(result[date_column])
    result["day_of_week"] = result[date_column].dt.dayofweek
    result["month"] = result[date_column].dt.month
    result["day_of_month"] = result[date_column].dt.day
    result["quarter"] = result[date_column].dt.quarter
    
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    
    return result


def add_volume_features(df: pd.DataFrame, volume_column: str = "volume") -> pd.DataFrame:
    result = df.copy()
    
    if volume_column not in result.columns:
        return result
    
    result["volume_sma_20"] = calculate_sma(result, volume_column, 20)
    result["volume_ratio"] = result[volume_column] / result["volume_sma_20"]
    
    return result


def create_all_features(
    df: pd.DataFrame,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
    volume_column: str = "volume",
    date_column: str = "date",
    symbol_column: Optional[str] = "symbol",
    windows: list[int] = [5, 10, 20, 50],
    lags: list[int] = [1, 2, 3, 5, 10],
    add_technical: bool = True,
    add_lags: bool = True,
    add_temporal: bool = True,
    add_volume: bool = True,
    simplified: bool = False,  # Нова опция за опростена версия
) -> pd.DataFrame:
    result = df.copy()
    
    if simplified:
        # Опростена версия с само най-важните features (~15-20 features)
        # Основни цени
        if price_column not in result.columns:
            result[price_column] = result[high_column]  # Fallback
        
        # Само най-важните moving averages
        result[f"sma_20"] = calculate_sma(result, price_column, 20)
        result[f"ema_20"] = calculate_ema(result, price_column, 20)
        
        # RSI за momentum
        result = add_rsi(result, price_column)
        
        # MACD за trend
        macd, signal, _ = calculate_macd(result, price_column)
        result["macd"] = macd
        result["macd_signal"] = signal
        
        # Bollinger Bands position (нормализирана позиция)
        upper, middle, lower = calculate_bollinger_bands(result, price_column, 20)
        result[f"bb_position_20"] = (result[price_column] - lower) / (upper - lower + 1e-8)
        
        # ATR за volatility
        if high_column in result.columns and low_column in result.columns:
            result["atr"] = calculate_atr(result, high_column, low_column, price_column)
        
        # Само 1 lag feature (Transformer attention покрива останалите)
        if add_lags:
            if symbol_column:
                result[f"{price_column}_lag_1"] = result.groupby(symbol_column)[price_column].shift(1)
            else:
                result[f"{price_column}_lag_1"] = result[price_column].shift(1)
        
        # Volume ratio
        if add_volume and volume_column in result.columns:
            result["volume_sma_20"] = calculate_sma(result, volume_column, 20)
            result["volume_ratio"] = result[volume_column] / (result["volume_sma_20"] + 1e-8)
        
        # Минимални temporal features
        if add_temporal and date_column in result.columns:
            result[date_column] = pd.to_datetime(result[date_column])
            result["day_of_week"] = result[date_column].dt.dayofweek
            result["month"] = result[date_column].dt.month
        
    else:
        # Пълна версия (оригинална)
        if add_technical:
            result = add_moving_averages(result, price_column, windows)
            result = add_rsi(result, price_column)
            result = add_macd(result, price_column)
            result = add_bollinger_bands(result, price_column)
            if high_column in result.columns and low_column in result.columns:
                result["atr"] = calculate_atr(result, high_column, low_column, price_column)
        
        if add_lags:
            lag_columns = [price_column]
            if volume_column in result.columns:
                lag_columns.append(volume_column)
            result = add_lag_features(result, lag_columns, lags, group_by=symbol_column)
        
        if add_temporal:
            result = add_temporal_features(result, date_column)
        
        if add_volume:
            result = add_volume_features(result, volume_column)
    
    return result


__all__ = [
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "add_moving_averages",
    "add_rsi",
    "add_macd",
    "add_bollinger_bands",
    "add_lag_features",
    "add_temporal_features",
    "add_volume_features",
    "create_all_features",
]
