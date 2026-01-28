from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def time_series_split(
    df: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    date_column: str = "date",
    symbol_column: Optional[str] = "symbol",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    result = result.sort_values([symbol_column, date_column] if symbol_column else [date_column])
    
    total_len = len(result)
    train_end = int(total_len * train_split)
    val_end = train_end + int(total_len * val_split)
    
    train_df = result.iloc[:train_end].copy()
    val_df = result.iloc[train_end:val_end].copy()
    test_df = result.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


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
) -> tuple[StockDataset, StockDataset, StockDataset]:
    train_df, val_df, test_df = time_series_split(
        df, train_split, val_split, test_split, date_column, symbol_column
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
