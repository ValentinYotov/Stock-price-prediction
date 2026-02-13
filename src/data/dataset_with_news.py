"""
Enhanced dataset with news embeddings support.

This extends the base StockDataset to include news data.
The base StockDataset remains unchanged for comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import time_series_split, create_sequences, StockDataset


class StockDatasetWithNews(Dataset):
    """
    Enhanced dataset that includes news embeddings alongside technical features.
    
    Returns:
        - x: Technical features [seq_len, feature_dim]
        - news_emb: News embedding [news_embedding_dim] or None
        - y: Target value
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        news_embeddings: Optional[np.ndarray] = None,
        context_length: int = 60,
        prediction_horizon: int = 1,
    ):
        """
        Initialize dataset with news embeddings.
        
        Args:
            data: Technical features array [n_samples, feature_dim]
            targets: Target values array [n_samples]
            news_embeddings: News embeddings array [n_samples, news_embedding_dim]
                           If None, dataset behaves like base StockDataset
            context_length: Sequence length
            prediction_horizon: Number of steps to predict
        """
        self.data = data
        self.targets = targets
        self.news_embeddings = news_embeddings
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        # Validate shapes
        if news_embeddings is not None:
            if len(news_embeddings) != len(data):
                raise ValueError(
                    f"News embeddings length ({len(news_embeddings)}) "
                    f"must match data length ({len(data)})"
                )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Get item with news embeddings.
        
        Returns:
            (x, news_emb, y) where:
            - x: Technical features [seq_len, feature_dim]
            - news_emb: News embedding [news_embedding_dim] or None
            - y: Target value [prediction_horizon]
        """
        x = torch.FloatTensor(self.data[idx])
        
        # Get news embedding for the last timestep (when prediction is made)
        news_emb = None
        if self.news_embeddings is not None:
            # News embedding corresponds to the last timestep in the sequence
            news_emb = torch.FloatTensor(self.news_embeddings[idx])
        
        y_target = self.targets[idx]
        if isinstance(y_target, (int, float, np.number)):
            y = torch.FloatTensor([y_target])
        else:
            y = torch.FloatTensor(y_target)
        
        return x, news_emb, y


def prepare_dataset_with_news(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    news_embeddings_df: Optional[pd.DataFrame] = None,
    context_length: int = 60,
    prediction_horizon: int = 1,
    date_column: str = "date",
    symbol_column: Optional[str] = "symbol",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple[StockDatasetWithNews, StockDatasetWithNews, StockDatasetWithNews]:
    """
    Prepare datasets with news embeddings.
    
    Args:
        df: DataFrame with technical features
        feature_columns: List of feature column names
        target_column: Target column name
        news_embeddings_df: DataFrame with news embeddings (must have same index/date alignment)
        context_length: Sequence length
        prediction_horizon: Prediction horizon
        date_column: Date column name
        symbol_column: Symbol column name
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Split data
    train_df, val_df, test_df = time_series_split(
        df, train_split, val_split, test_split, date_column, symbol_column
    )
    
    # Prepare technical features
    train_data = train_df[feature_columns].values
    train_targets = train_df[target_column].values.reshape(-1, 1)
    
    val_data = val_df[feature_columns].values
    val_targets = val_df[target_column].values.reshape(-1, 1)
    
    test_data = test_df[feature_columns].values
    test_targets = test_df[target_column].values.reshape(-1, 1)
    
    # Create sequences for technical features
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
    
    # Prepare news embeddings if provided
    # News embeddings must be aligned with sequences, not with original DataFrame rows
    train_news = None
    val_news = None
    test_news = None
    
    if news_embeddings_df is not None and len(news_embeddings_df) > 0:
        # Align news embeddings with sequences
        # For each sequence ending at index i, use news embedding at index i
        train_news = _align_news_embeddings_with_sequences(
            train_df, train_X, news_embeddings_df, date_column, symbol_column, context_length
        )
        val_news = _align_news_embeddings_with_sequences(
            val_df, val_X, news_embeddings_df, date_column, symbol_column, context_length
        )
        test_news = _align_news_embeddings_with_sequences(
            test_df, test_X, news_embeddings_df, date_column, symbol_column, context_length
        )
    
    # Create datasets
    train_dataset = StockDatasetWithNews(
        train_X, train_y, train_news, context_length, prediction_horizon
    )
    val_dataset = StockDatasetWithNews(
        val_X, val_y, val_news, context_length, prediction_horizon
    )
    test_dataset = StockDatasetWithNews(
        test_X, test_y, test_news, context_length, prediction_horizon
    )
    
    return train_dataset, val_dataset, test_dataset


def _align_news_embeddings_with_sequences(
    df: pd.DataFrame,
    sequences: np.ndarray,
    news_embeddings_df: pd.DataFrame,
    date_column: str,
    symbol_column: Optional[str],
    context_length: int,
) -> Optional[np.ndarray]:
    """
    Align news embeddings with data sequences.
    
    For each sequence ending at index i in the original DataFrame,
    we use the news embedding at index i (the last timestep when prediction is made).
    
    Args:
        df: DataFrame with technical features (before sequence creation)
        sequences: Created sequences array [n_sequences, seq_len, feature_dim]
        news_embeddings_df: DataFrame with news embeddings
        date_column: Date column name
        symbol_column: Symbol column name
        context_length: Sequence length
    
    Returns:
        News embeddings array [n_sequences, news_embedding_dim] aligned with sequences
    """
    if news_embeddings_df is None or len(news_embeddings_df) == 0:
        return None
    
    # Get news embedding columns
    news_cols = [
        col for col in news_embeddings_df.columns
        if col.startswith('news_embedding_')
    ]
    
    if not news_cols:
        return None
    
    n_sequences = len(sequences)
    if n_sequences == 0:
        return None
    
    # Merge news embeddings with DataFrame by date and symbol
    df_merge = df.copy()
    df_merge[date_column] = pd.to_datetime(df_merge[date_column])
    
    news_df_merge = news_embeddings_df.copy()
    news_df_merge['date'] = pd.to_datetime(news_df_merge['date'])
    
    # Prepare merge columns
    merge_on_left = [date_column]
    merge_on_right = ['date']
    if symbol_column:
        merge_on_left.append(symbol_column)
        merge_on_right.append('ticker')
    
    # Merge news embeddings with DataFrame
    merged = df_merge.merge(
        news_df_merge[merge_on_right + news_cols],
        left_on=merge_on_left,
        right_on=merge_on_right,
        how='left',
        suffixes=('', '_news')
    )
    
    # Extract news embeddings aligned with DataFrame rows
    news_embeddings = merged[news_cols].values
    
    # Fill NaN with zeros (for dates without news)
    news_embeddings = np.nan_to_num(news_embeddings, nan=0.0)
    
    # Now align with sequences
    # For each sequence ending at index i, use news embedding at index i
    # Sequences are created starting from index 0, so:
    # - Sequence 0 ends at index context_length - 1
    # - Sequence 1 ends at index context_length
    # - Sequence k ends at index context_length - 1 + k
    aligned_news = []
    for seq_idx in range(n_sequences):
        # Index in original DataFrame where this sequence ends
        df_idx = context_length - 1 + seq_idx
        
        if df_idx < len(news_embeddings):
            news_emb = news_embeddings[df_idx]
        else:
            # Should not happen, but use zeros if it does
            news_emb = np.zeros(len(news_cols))
        
        aligned_news.append(news_emb)
    
    return np.array(aligned_news)


__all__ = [
    "StockDatasetWithNews",
    "prepare_dataset_with_news",
]
