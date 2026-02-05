"""
Backtest engine: simulates trading on predictions, tracks portfolio and trade list.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .rules import signal_from_prediction, Signal


@dataclass
class Trade:
    """Single trade (buy or sell)."""
    date_idx: int
    date: Optional[Union[str, int]]
    side: Signal
    price: float
    quantity: float
    commission: float
    cash_after: float
    shares_after: float


@dataclass
class BacktestResult:
    """Backtest result: equity curve, trades, final state."""
    equity_curve: np.ndarray
    trades: List[Trade] = field(default_factory=list)
    final_cash: float = 0.0
    final_shares: float = 0.0
    dates: Optional[np.ndarray] = None


class BacktestEngine:
    """
    Backtest engine: given prices and predictions, applies rules,
    executes buy/sell and tracks portfolio.
    """

    def __init__(
        self,
        initial_capital: float,
        position_size_pct: float = 1.0,
        entry_threshold_pct: float = 0.5,
        exit_threshold_pct: float = -0.5,
        commission_pct: float = 0.1,
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = max(0.01, min(1.0, position_size_pct))
        self.entry_threshold_pct = entry_threshold_pct
        self.exit_threshold_pct = exit_threshold_pct
        self.commission_pct = commission_pct

    def run(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run the simulation for each day.

        Args:
            prices: 1D array of actual price per day (length N).
            predictions: 1D array of predicted price for next period (length N).
            dates: Optional date/label per day (length N).

        Returns:
            BacktestResult with equity_curve, trades, final_cash, final_shares.
        """
        prices = np.asarray(prices).ravel()
        predictions = np.asarray(predictions).ravel()
        n = len(prices)
        if len(predictions) != n:
            raise ValueError("prices and predictions must have the same length")

        if dates is None:
            dates = np.arange(n)
        dates = np.asarray(dates)

        cash = float(self.initial_capital)
        shares = 0.0
        equity_curve = np.zeros(n)
        trades: List[Trade] = []

        for i in range(n):
            price = float(prices[i])
            pred = float(predictions[i])
            in_position = shares > 0

            signal = signal_from_prediction(
                current_price=price,
                predicted_price=pred,
                in_position=in_position,
                entry_threshold_pct=self.entry_threshold_pct,
                exit_threshold_pct=self.exit_threshold_pct,
            )

            # Debug first few trades
            if i < 5:
                pct_chg = 100 * (pred - price) / abs(price) if abs(price) > 1e-9 else 0
                print(f"  Engine Day {i}: price={price:.4f}, pred={pred:.4f}, pct={pct_chg:.2f}%, "
                      f"in_pos={in_position}, signal={signal}, abs(price)={abs(price):.6f}")

            if signal == "buy" and not in_position and abs(price) > 1e-9:
                equity = cash
                amount = equity * self.position_size_pct
                commission = amount * (self.commission_pct / 100.0)
                cost = amount - commission
                qty = cost / price
                if qty > 0:
                    cash -= amount
                    shares = qty
                    trades.append(
                        Trade(
                            date_idx=i,
                            date=dates[i] if i < len(dates) else i,
                            side="buy",
                            price=price,
                            quantity=qty,
                            commission=commission,
                            cash_after=cash,
                            shares_after=shares,
                        )
                    )

            elif signal == "sell" and in_position and abs(price) > 1e-9:
                gross = shares * price
                commission = gross * (self.commission_pct / 100.0)
                cash += gross - commission
                qty = shares
                shares = 0.0
                trades.append(
                    Trade(
                        date_idx=i,
                        date=dates[i] if i < len(dates) else i,
                        side="sell",
                        price=price,
                        quantity=qty,
                        commission=commission,
                        cash_after=cash,
                        shares_after=0.0,
                    )
                )

            equity_curve[i] = cash + shares * price

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            final_cash=cash,
            final_shares=shares,
            dates=dates,
        )
