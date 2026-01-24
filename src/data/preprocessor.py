from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "forward_fill",
    columns: Optional[list] = None,
) -> pd.DataFrame:
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == "forward_fill":
        result[columns] = result[columns].fillna(method="ffill")
        result[columns] = result[columns].fillna(method="bfill")
    elif method == "backward_fill":
        result[columns] = result[columns].fillna(method="bfill")
        result[columns] = result[columns].fillna(method="ffill")
    elif method == "interpolate":
        result[columns] = result[columns].interpolate(method="linear")
        result[columns] = result[columns].fillna(method="ffill")
        result[columns] = result[columns].fillna(method="bfill")
    elif method == "drop":
        result = result.dropna(subset=columns)
    
    return result


def handle_outliers(
    df: pd.DataFrame,
    method: str = "clip",
    columns: Optional[list] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.DataFrame:
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == "clip":
        for col in columns:
            lower = result[col].quantile(lower_percentile)
            upper = result[col].quantile(upper_percentile)
            result[col] = result[col].clip(lower=lower, upper=upper)
    elif method == "remove":
        for col in columns:
            lower = result[col].quantile(lower_percentile)
            upper = result[col].quantile(upper_percentile)
            mask = (result[col] >= lower) & (result[col] <= upper)
            result = result[mask]
    
    return result


def normalize_data(
    df: pd.DataFrame,
    method: str = "minmax",
    columns: Optional[list] = None,
    scaler: Optional[object] = None,
) -> tuple[pd.DataFrame, object]:
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if col not in ["symbol"] if "symbol" in result.columns]
    
    if scaler is None:
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        result[columns] = scaler.fit_transform(result[columns])
    else:
        result[columns] = scaler.transform(result[columns])
    
    return result, scaler


def preprocess_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    missing_method: str = "forward_fill",
    handle_outliers_flag: bool = True,
    outliers_method: str = "clip",
    normalize: bool = False,
    normalize_method: str = "minmax",
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> tuple[pd.DataFrame, Optional[object]]:
    result = df.copy()
    scaler = None
    
    if date_column in result.columns:
        result[date_column] = pd.to_datetime(result[date_column])
        result = result.sort_values([symbol_column, date_column] if symbol_column in result.columns else [date_column])
    
    if handle_missing:
        result = handle_missing_values(result, method=missing_method)
    
    if handle_outliers_flag:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if symbol_column in numeric_cols:
            numeric_cols.remove(symbol_column)
        result = handle_outliers(result, method=outliers_method, columns=numeric_cols)
    
    if normalize:
        result, scaler = normalize_data(result, method=normalize_method)
    
    return result, scaler


__all__ = [
    "handle_missing_values",
    "handle_outliers",
    "normalize_data",
    "preprocess_data",
]
