from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _split_chronological(
    df: pd.DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
    date_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single chronologically sorted series by time (no shuffle)."""
    total_len = len(df)
    if total_len < 3:
        raise ValueError(f"Need at least 3 rows to split, got {total_len}")

    train_end = int(total_len * train_split)
    val_end = train_end + int(total_len * val_split)

    if train_end < 1 or val_end <= train_end or val_end >= total_len:
        raise ValueError(
            f"Split ratios leave an empty partition (n={total_len}, "
            f"train_end={train_end}, val_end={val_end})"
        )

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def time_series_split(
    df: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    date_column: str = "date",
    symbol_column: Optional[str] = "symbol",
    per_symbol: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val/test split.

    per_symbol=False: legacy global split on all rows (avoid with multiple tickers).
    per_symbol=True: split each ticker by its own timeline, then concatenate.
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")

    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    use_per_symbol = (
        per_symbol
        and symbol_column is not None
        and symbol_column in result.columns
        and result[symbol_column].nunique() > 1
    )

    if not use_per_symbol:
        sort_cols = [date_column]
        if symbol_column and symbol_column in result.columns:
            sort_cols = [symbol_column, date_column]
        result = result.sort_values(sort_cols)
        return _split_chronological(result, train_split, val_split, test_split, date_column)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in result.groupby(symbol_column, sort=False):
        group = group.sort_values(date_column)
        train_df, val_df, test_df = _split_chronological(
            group, train_split, val_split, test_split, date_column
        )
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def create_sequences(
    data: np.ndarray,
    context_length: int,
    prediction_horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    
    for i in range(0, len(data) - context_length - prediction_horizon + 1, stride):
        seq = data[i : i + context_length]
        target = data[i + context_length : i + context_length + prediction_horizon]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


class StockDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        context_length: int,
        prediction_horizon: int = 1,
    ):
        self.data = data
        self.targets = targets
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.data[idx])
        y_target = self.targets[idx]
        if isinstance(y_target, (int, float, np.number)):
            y = torch.FloatTensor([y_target])
        else:
            y = torch.FloatTensor(y_target)
        return x, y


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    context_length: int,
    prediction_horizon: int = 1,
    date_column: str = "date",
    symbol_column: Optional[str] = "symbol",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    per_symbol: bool = False,
) -> tuple[StockDataset, StockDataset, StockDataset]:
    train_df, val_df, test_df = time_series_split(
        df,
        train_split,
        val_split,
        test_split,
        date_column,
        symbol_column,
        per_symbol=per_symbol,
    )
    
    train_data = train_df[feature_columns].values
    train_targets = train_df[target_column].values.reshape(-1, 1)
    
    val_data = val_df[feature_columns].values
    val_targets = val_df[target_column].values.reshape(-1, 1)
    
    test_data = test_df[feature_columns].values
    test_targets = test_df[target_column].values.reshape(-1, 1)
    
    train_X, train_y = create_sequences(
        np.column_stack([train_data, train_targets]),
        context_length,
        prediction_horizon,
    )
    train_X = train_X[:, :, :-1]
    if prediction_horizon > 1:
        train_y = train_y[:, :, -1]
    else:
        train_y = train_y[:, -1, -1]
        if train_y.ndim == 0:
            train_y = train_y.reshape(-1, 1)
    
    val_X, val_y = create_sequences(
        np.column_stack([val_data, val_targets]),
        context_length,
        prediction_horizon,
    )
    val_X = val_X[:, :, :-1]
    if prediction_horizon > 1:
        val_y = val_y[:, :, -1]
    else:
        val_y = val_y[:, -1, -1]
        if val_y.ndim == 0:
            val_y = val_y.reshape(-1, 1)
    
    test_X, test_y = create_sequences(
        np.column_stack([test_data, test_targets]),
        context_length,
        prediction_horizon,
    )
    test_X = test_X[:, :, :-1]
    if prediction_horizon > 1:
        test_y = test_y[:, :, -1]
    else:
        test_y = test_y[:, -1, -1]
        if test_y.ndim == 0:
            test_y = test_y.reshape(-1, 1)
    
    train_dataset = StockDataset(train_X, train_y, context_length, prediction_horizon)
    val_dataset = StockDataset(val_X, val_y, context_length, prediction_horizon)
    test_dataset = StockDataset(test_X, test_y, context_length, prediction_horizon)
    
    return train_dataset, val_dataset, test_dataset


__all__ = [
    "time_series_split",
    "create_sequences",
    "StockDataset",
    "prepare_dataset",
]


