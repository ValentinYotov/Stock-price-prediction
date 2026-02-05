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
        shares = 0.0  # Positive = long, negative = short, 0 = cash
        equity_curve = np.zeros(n)
        trades: List[Trade] = []
        
        # Track position state for intelligent exit decisions
        entry_price = None  # Price when we entered the position
        entry_day = None  # Day when we entered
        peak_price = None  # Highest price since entry (for trailing stop)
        peak_prediction_momentum = None  # Best prediction momentum since entry

        for i in range(n):
            price = float(prices[i])
            pred = float(predictions[i])
            is_long = shares > 1e-9
            is_short = shares < -1e-9
            is_cash = abs(shares) < 1e-9

            # Calculate % change for signal logic
            pct_change = 100 * (pred - price) / abs(price) if abs(price) > 1e-9 else 0
            
            # Track position metrics when in long position
            if is_long:
                if entry_price is None:
                    # Just entered position
                    entry_price = price
                    entry_day = i
                    peak_price = price
                    peak_prediction_momentum = pct_change
                else:
                    # Update peak tracking
                    if abs(price) > abs(peak_price):
                        peak_price = price
                    if pct_change > peak_prediction_momentum:
                        peak_prediction_momentum = pct_change
            else:
                # Reset tracking when not in position
                entry_price = None
                entry_day = None
                peak_price = None
                peak_prediction_momentum = None
            
            # Determine signal based on position and prediction
            signal = None
            if is_cash:
                # In cash: buy if prediction shows good opportunity
                # More selective: need strong positive momentum
                if pct_change >= self.entry_threshold_pct:
                    signal = "buy"
                elif pct_change <= self.exit_threshold_pct:
                    signal = "sell"  # Short sell (if prediction is very negative)
            elif is_long:
                # Long position: sell when we see signs of decline
                # 1. Real price drop: if price dropped significantly from peak (trailing stop)
                if entry_price is not None and peak_price is not None:
                    # Calculate actual price change from entry and from peak
                    price_change_from_entry = 100 * (abs(price) - abs(entry_price)) / abs(entry_price)
                    price_change_from_peak = 100 * (abs(price) - abs(peak_price)) / abs(peak_price)
                    
                    # Sell if price dropped more than 3% from peak (trailing stop)
                    if price_change_from_peak < -3.0:
                        signal = "sell"
                    
                    # Sell if prediction momentum weakened significantly
                    # If we had strong momentum (>10%) and now it's much weaker
                    if peak_prediction_momentum is not None and peak_prediction_momentum > 10.0:
                        if pct_change < peak_prediction_momentum * 0.5:  # Momentum dropped to <50% of peak
                            signal = "sell"
                    
                    # Sell if prediction turns negative (price will drop)
                    if pct_change <= self.exit_threshold_pct:
                        signal = "sell"
                else:
                    # Fallback: sell if prediction is very negative
                    if pct_change <= self.exit_threshold_pct:
                        signal = "sell"
            else:  # is_short
                # Short position: buy to cover if prediction rises
                if pct_change >= -self.exit_threshold_pct:
                    signal = "buy"  # Cover short
            
            # Execute trades (no cooldown - let the strategy decide naturally)
            if signal == "buy" and abs(price) > 1e-9:
                if is_short:
                    # Cover short: buy back shares we sold
                    qty_to_cover = abs(shares)
                    cost = qty_to_cover * abs(price)
                    commission = cost * (self.commission_pct / 100.0)
                    total_cost = cost + commission
                    if cash >= total_cost:
                        cash -= total_cost
                        shares = 0.0  # Close short position
                        trades.append(
                            Trade(
                                date_idx=i,
                                date=dates[i] if i < len(dates) else i,
                                side="buy",
                                price=price,
                                quantity=qty_to_cover,
                                commission=commission,
                                cash_after=cash,
                                shares_after=0.0,
                            )
                        )
                elif is_cash:
                    # Open long position: use position_size_pct of available cash
                    equity = cash
                    amount = equity * self.position_size_pct
                    commission = amount * (self.commission_pct / 100.0)
                    cost = amount - commission
                    qty = cost / abs(price)
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

            elif signal == "sell" and abs(price) > 1e-9:
                if is_long:
                    # Close long position
                    gross = shares * abs(price)
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
                elif is_cash:
                    # Open short position: sell shares we don't own
                    equity = cash
                    amount = equity * self.position_size_pct
                    qty_to_sell = amount / abs(price)
                    gross = qty_to_sell * abs(price)
                    commission = gross * (self.commission_pct / 100.0)
                    cash += gross - commission  # Receive cash from short sale
                    shares = -qty_to_sell  # Negative = short position
                    trades.append(
                        Trade(
                            date_idx=i,
                            date=dates[i] if i < len(dates) else i,
                            side="sell",
                            price=price,
                            quantity=qty_to_sell,
                            commission=commission,
                            cash_after=cash,
                            shares_after=shares,
                        )
                    )

            # Calculate equity: cash + value of positions
            # For long: shares * price
            # For short: we owe shares, so equity decreases if price rises
            if shares > 0:
                equity_curve[i] = cash + shares * abs(price)
            elif shares < 0:
                # Short position: equity = cash - (shares_owed * current_price)
                # When we short, we receive cash, but owe shares
                equity_curve[i] = cash + shares * abs(price)  # shares is negative, so this subtracts
            else:
                equity_curve[i] = cash

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            final_cash=cash,
            final_shares=shares,
            dates=dates,
        )
