import numpy as np
import pandas as pd
import pytest

from src.data.dataset import time_series_split
from src.data.feature_engineering import add_forward_log_return


def test_time_series_split_per_symbol_preserves_ticker_boundaries():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["A"] * 100 + ["B"] * 100,
        "close": range(200),
    })

    train_df, val_df, test_df = time_series_split(
        df, per_symbol=True, symbol_column="symbol", date_column="date"
    )

    assert len(train_df) == 140  # 70 per symbol
    assert len(val_df) == 30
    assert len(test_df) == 30

    for part in (train_df, val_df, test_df):
        for symbol in ("A", "B"):
            sub = part[part["symbol"] == symbol]
            assert sub["date"].is_monotonic_increasing


def test_time_series_split_global_legacy():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "close": range(100),
    })
    train_df, val_df, test_df = time_series_split(df, per_symbol=False, symbol_column=None)
    assert len(train_df) + len(val_df) + len(test_df) == 100


def test_forward_log_return_aligns_with_next_close():
    df = pd.DataFrame({
        "symbol": ["X"] * 3,
        "close": [100.0, 110.0, 121.0],
    })
    out = add_forward_log_return(df, price_column="close", symbol_column="symbol")
    assert out["log_return"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert out["log_return"].iloc[1] == pytest.approx(np.log(121 / 110))
    assert pd.isna(out["log_return"].iloc[2])
