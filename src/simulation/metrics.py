"""
Backtest performance metrics: return, Sharpe ratio, max drawdown, buy-and-hold comparison.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .engine import BacktestResult


@dataclass
class SimulationMetrics:
    """Aggregated metrics from a backtest run."""
    total_return_pct: float
    sharpe_ratio_annual: float
    max_drawdown_pct: float
    num_trades: int
    buy_and_hold_return_pct: Optional[float] = None
    excess_return_vs_bh_pct: Optional[float] = None  # strategy return minus buy-and-hold


def _daily_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Log or simple returns; avoid div by zero."""
    eq = np.asarray(equity_curve, dtype=float).ravel()
    prev = eq[:-1]
    prev = np.where(prev <= 0, np.nan, prev)
    return (eq[1:] - prev) / prev


def total_return_pct(equity_curve: np.ndarray, initial_capital: float) -> float:
    """Total return in percent. Uses first and last equity values."""
    eq = np.asarray(equity_curve).ravel()
    if len(eq) == 0 or initial_capital <= 0:
        return 0.0
    final = float(eq[-1])
    return 100.0 * (final - initial_capital) / initial_capital


def sharpe_ratio_annual(
    equity_curve: np.ndarray,
    risk_free_rate_annual: float = 0.03,
    trading_days_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio from daily returns.
    Assumes risk-free rate is annual; converts to daily for excess return.
    """
    ret = _daily_returns(equity_curve)
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return 0.0
    rf_daily = (1 + risk_free_rate_annual) ** (1.0 / trading_days_per_year) - 1
    excess = ret - rf_daily
    std = np.nanstd(ret)
    if std <= 0:
        return 0.0
    return float(np.sqrt(trading_days_per_year) * np.nanmean(excess) / std)


def max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Maximum drawdown in percent (peak to trough)."""
    eq = np.asarray(equity_curve, dtype=float).ravel()
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    peak = np.where(peak <= 0, np.nan, peak)
    dd = (peak - eq) / peak
    return 100.0 * float(np.nanmax(dd))


def buy_and_hold_return_pct(
    initial_capital: float,
    prices: np.ndarray,
) -> float:
  
    p = np.asarray(prices).ravel()
    if len(p) < 2 or abs(p[0]) < 1e-9 or initial_capital <= 0:
        return 0.0
    
    pct_change = (p[-1] - p[0]) / abs(p[0]) * 100.0
    
    # Buy & hold return = same as price return (we buy and hold)
    return pct_change


def compute_metrics(
    result: BacktestResult,
    initial_capital: float,
    risk_free_rate_annual: float = 0.03,
    trading_days_per_year: int = 252,
    prices: Optional[np.ndarray] = None,
) -> SimulationMetrics:
    
    eq = result.equity_curve
    ret_pct = total_return_pct(eq, initial_capital)
    sharpe = sharpe_ratio_annual(eq, risk_free_rate_annual, trading_days_per_year)
    mdd = max_drawdown_pct(eq)
    n_trades = len(result.trades)

    bh_pct = None
    excess_pct = None
    if prices is not None and len(prices) == len(eq):
        bh_pct = buy_and_hold_return_pct(initial_capital, prices)
        excess_pct = ret_pct - bh_pct

    return SimulationMetrics(
        total_return_pct=ret_pct,
        sharpe_ratio_annual=sharpe,
        max_drawdown_pct=mdd,
        num_trades=n_trades,
        buy_and_hold_return_pct=bh_pct,
        excess_return_vs_bh_pct=excess_pct,
    )
