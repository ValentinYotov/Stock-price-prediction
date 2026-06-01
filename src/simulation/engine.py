"""
Backtest engine: simulates trading on predictions, tracks portfolio and trade list.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .rules import (
    signal_from_prediction,
    signal_from_return,
    signal_from_return_band,
    Signal,
)


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
    entry_thresholds: Optional[np.ndarray] = None
    exit_thresholds: Optional[np.ndarray] = None
    margin_call_count: int = 0
    account_wiped_at_idx: Optional[int] = None
    account_wiped_at_date: Optional[Union[str, int]] = None


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

    def run_from_log_returns(
        self,
        prices: np.ndarray,
        predicted_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        entry_threshold: float = 0.005,
        exit_threshold: float = -0.005,
        signal_mode: str = "default",
    ) -> BacktestResult:
        """
        Long-only backtest using predicted next-period log returns and real close prices.

        At day i: signal from predicted_returns[i]; trades execute at prices[i] (close).

        signal_mode:
            "default" - signal_from_return (sells aggressively when momentum weakens).
            "band"    - signal_from_return_band (buys above entry, sells below exit, holds in between).
        """
        prices = np.asarray(prices, dtype=float).ravel()
        predicted_returns = np.asarray(predicted_returns, dtype=float).ravel()
        n = len(prices)
        if len(predicted_returns) != n:
            raise ValueError("prices and predicted_returns must have the same length")
        if n == 0:
            raise ValueError("prices must not be empty")

        if dates is None:
            dates = np.arange(n)
        dates = np.asarray(dates)

        cash = float(self.initial_capital)
        shares = 0.0
        equity_curve = np.zeros(n)
        trades: List[Trade] = []

        for i in range(n):
            price = float(prices[i])
            pred_ret = float(predicted_returns[i])
            is_long = shares > 1e-9
            is_cash = abs(shares) < 1e-9

            if signal_mode == "band":
                signal = signal_from_return_band(
                    pred_ret,
                    in_position=is_long,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                )
            else:
                signal = signal_from_return(
                    pred_ret,
                    in_position=is_long,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                )

            if signal == "buy" and price > 1e-9 and is_cash:
                amount = cash * self.position_size_pct
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

            elif signal == "sell" and price > 1e-9 and is_long:
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

    def run_band_walk_forward(
        self,
        prices: np.ndarray,
        predicted_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        entry_quantile: float = 0.70,
        exit_quantile: float = 0.30,
        warmup: int = 60,
        smoothing_window: int = 1,
        confirmation_days: int = 1,
    ) -> BacktestResult:
        """
        Long-only band backtest with causal (walk-forward) quantile thresholds.

        At day t the entry/exit thresholds are computed as the requested quantiles
        of the *smoothed* predictions seen so far, i.e. they depend exclusively
        on information that would have been available at t. This removes the
        look-ahead bias of computing quantiles over the full test period.

        During the first ``warmup`` days no trading occurs; we just accumulate
        predictions so that the quantile estimates are stable. After that:
        - the raw next-day prediction is replaced by an EMA over the last
          ``smoothing_window`` days (span = smoothing_window),
        - we buy only when the smoothed prediction stays above the entry
          threshold for ``confirmation_days`` consecutive days, and sell when
          it stays below the exit threshold for the same number of days.

        ``smoothing_window = 1`` and ``confirmation_days = 1`` reproduce the
        original single-day band behaviour.
        """
        prices = np.asarray(prices, dtype=float).ravel()
        predicted_returns = np.asarray(predicted_returns, dtype=float).ravel()
        n = len(prices)
        if len(predicted_returns) != n:
            raise ValueError("prices and predicted_returns must have the same length")
        if n == 0:
            raise ValueError("prices must not be empty")
        if not 0.0 < exit_quantile < entry_quantile < 1.0:
            raise ValueError("require 0 < exit_quantile < entry_quantile < 1")
        warmup = max(1, int(warmup))
        smoothing_window = max(1, int(smoothing_window))
        confirmation_days = max(1, int(confirmation_days))

        if smoothing_window > 1:
            alpha = 2.0 / (smoothing_window + 1.0)
            smoothed = np.empty_like(predicted_returns)
            smoothed[0] = predicted_returns[0]
            for j in range(1, n):
                smoothed[j] = alpha * predicted_returns[j] + (1.0 - alpha) * smoothed[j - 1]
        else:
            smoothed = predicted_returns

        if dates is None:
            dates = np.arange(n)
        dates = np.asarray(dates)

        cash = float(self.initial_capital)
        shares = 0.0
        equity_curve = np.zeros(n)
        entry_thr_curve = np.full(n, np.nan)
        exit_thr_curve = np.full(n, np.nan)
        trades: List[Trade] = []
        consec_above = 0
        consec_below = 0

        for i in range(n):
            price = float(prices[i])
            pred_ret = float(smoothed[i])
            is_long = shares > 1e-9
            is_cash = abs(shares) < 1e-9

            history = smoothed[: i + 1]
            if len(history) >= warmup:
                entry_thr = float(np.quantile(history, entry_quantile))
                exit_thr = float(np.quantile(history, exit_quantile))
                entry_thr_curve[i] = entry_thr
                exit_thr_curve[i] = exit_thr

                if pred_ret >= entry_thr:
                    consec_above += 1
                    consec_below = 0
                elif pred_ret <= exit_thr:
                    consec_below += 1
                    consec_above = 0
                else:
                    consec_above = 0
                    consec_below = 0

                if is_cash and consec_above >= confirmation_days:
                    signal = "buy"
                elif is_long and consec_below >= confirmation_days:
                    signal = "sell"
                else:
                    signal = "hold"
            else:
                signal = "hold"
                consec_above = 0
                consec_below = 0

            if signal == "buy" and price > 1e-9 and is_cash:
                amount = cash * self.position_size_pct
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

            elif signal == "sell" and price > 1e-9 and is_long:
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
            entry_thresholds=entry_thr_curve,
            exit_thresholds=exit_thr_curve,
        )

    def run_smart_long_only(
        self,
        prices: np.ndarray,
        predicted_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        smoothing_span: int = 3,
        vol_window: int = 20,
        regime_window: int = 30,
        entry_z_quantile: float = 0.50,
        exit_z_quantile: float = 0.20,
        confirmation_days: int = 1,
        trailing_stop_pct: float = 10.0,
        cooldown_days: int = 1,
        stress_vol_multiplier: float = 2.5,
        leverage: float = 1.0,
    ) -> BacktestResult:
        """
        Long-biased opinionated strategy. Designed to actually participate
        in trends instead of waiting for rare "perfect" setups:

        1. EMA-smooth the model predictions (span=smoothing_span).
        2. Convert smoothed prediction into a risk-adjusted z-score via
           trailing realised return volatility (vol_window days).
        3. Calibrate entry/exit thresholds adaptively from each model's
           own past z-score distribution (walk-forward, no look-ahead).
           This makes the strategy model-agnostic so base and news models
           are compared on equal footing.
        4. Entry (only when in cash) requires ALL of:
             a. z-score >= past ``entry_z_quantile`` for ``confirmation_days``
                consecutive days (default: simply above the model's own median).
             b. realised volatility is not in an extreme stress regime
                (current vol < stress_vol_multiplier x median over the last
                ``regime_window`` days). Default 2.5x only filters true
                black-swan days (e.g. mid-March 2020) rather than ordinary
                volatility spikes.
             c. ``cooldown_days`` have elapsed since the last sell.
        5. While long, exit at the close of day t on the FIRST of:
             - Trailing stop: price drops trailing_stop_pct from post-entry peak.
             - Signal exit: z-score drops below past ``exit_z_quantile``.

        There is intentionally no fixed take-profit or max-hold rule: winners
        are allowed to keep running until the model signal weakens or price
        reverses materially from the post-entry peak.

        All calculations are causal -- only past prices and predictions used.
        """
        prices = np.asarray(prices, dtype=float).ravel()
        predicted_returns = np.asarray(predicted_returns, dtype=float).ravel()
        n = len(prices)
        if len(predicted_returns) != n:
            raise ValueError("prices and predicted_returns must have the same length")
        if n == 0:
            raise ValueError("prices must not be empty")
        if not 0.0 < exit_z_quantile < entry_z_quantile < 1.0:
            raise ValueError("require 0 < exit_z_quantile < entry_z_quantile < 1")

        smoothing_span = max(1, int(smoothing_span))
        vol_window = max(5, int(vol_window))
        regime_window = max(vol_window, int(regime_window))
        confirmation_days = max(1, int(confirmation_days))
        cooldown_days = max(0, int(cooldown_days))
        leverage = max(1.0, float(leverage))

        if smoothing_span > 1:
            alpha = 2.0 / (smoothing_span + 1.0)
            smoothed = np.empty_like(predicted_returns)
            smoothed[0] = predicted_returns[0]
            for j in range(1, n):
                smoothed[j] = alpha * predicted_returns[j] + (1.0 - alpha) * smoothed[j - 1]
        else:
            smoothed = predicted_returns.copy()

        realized = np.zeros(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            realized[1:] = np.log(np.where(prices[:-1] > 0, prices[1:] / prices[:-1], 1.0))
        realized = np.nan_to_num(realized, nan=0.0, posinf=0.0, neginf=0.0)

        warmup = max(vol_window, regime_window)
        if dates is None:
            dates = np.arange(n)
        dates = np.asarray(dates)

        cash = float(self.initial_capital)
        shares = 0.0
        borrowed = 0.0
        equity_curve = np.zeros(n)
        entry_thr_curve = np.full(n, np.nan)
        exit_thr_curve = np.full(n, np.nan)
        trades: List[Trade] = []
        z_history: List[float] = []

        consec_above = 0
        last_sell_idx = -10_000
        entry_price: Optional[float] = None
        peak_price: Optional[float] = None
        margin_call_count = 0
        account_wiped_idx: Optional[int] = None
        account_wiped_date: Optional[Union[str, int]] = None

        def total_equity(p: float) -> float:
            return cash + shares * p - borrowed

        def position_equity(p: float) -> float:
            return shares * p - borrowed

        for i in range(n):
            price = float(prices[i])
            is_long = shares > 1e-9
            is_cash = abs(shares) < 1e-9

            if account_wiped_idx is not None:
                equity_curve[i] = max(0.0, total_equity(price))
                continue

            past_vol = float(np.std(realized[max(0, i - vol_window):i], ddof=0)) if i > 0 else 0.0
            past_vol = past_vol if past_vol > 1e-6 else 1e-6
            z_score = float(smoothed[i]) / past_vol
            z_history.append(z_score)

            if i < warmup:
                equity_curve[i] = total_equity(price)
                consec_above = 0
                continue

            past_z = np.asarray(z_history[:-1], dtype=float)
            if len(past_z) < vol_window:
                equity_curve[i] = total_equity(price)
                consec_above = 0
                continue

            entry_z_thr = float(np.quantile(past_z, entry_z_quantile))
            exit_z_thr = float(np.quantile(past_z, exit_z_quantile))
            entry_thr_curve[i] = entry_z_thr
            exit_thr_curve[i] = exit_z_thr

            if z_score >= entry_z_thr:
                consec_above += 1
            else:
                consec_above = 0

            signal = "hold"
            if is_long:
                if entry_price is not None and peak_price is not None:
                    peak_price = max(peak_price, price)
                    drop_from_peak_pct = 100.0 * (price - peak_price) / peak_price

                    if drop_from_peak_pct <= -trailing_stop_pct:
                        signal = "sell"
                    elif z_score <= exit_z_thr:
                        signal = "sell"

            elif is_cash:
                regime_slice = realized[max(0, i - regime_window):i]
                median_vol = float(np.median(np.abs(regime_slice))) if len(regime_slice) else 0.0
                stress = median_vol > 1e-6 and past_vol > stress_vol_multiplier * median_vol
                cooled_down = (i - last_sell_idx) >= cooldown_days
                if (
                    consec_above >= confirmation_days
                    and not stress
                    and cooled_down
                ):
                    signal = "buy"

            if signal == "buy" and price > 1e-9 and is_cash and cash > 1e-9:
                margin = cash * self.position_size_pct
                notional = margin * leverage
                commission = notional * (self.commission_pct / 100.0)
                qty = (notional - commission) / price
                if qty > 0:
                    cash -= margin
                    borrowed = notional - margin
                    shares = qty
                    entry_price = price
                    peak_price = price
                    consec_above = 0
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
            elif signal == "sell" and price > 1e-9 and is_long:
                gross = shares * price
                commission = gross * (self.commission_pct / 100.0)
                cash += gross - commission - borrowed
                if cash < 0:
                    cash = 0.0
                qty = shares
                shares = 0.0
                borrowed = 0.0
                entry_price = None
                peak_price = None
                last_sell_idx = i
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

            if shares > 1e-9 and position_equity(price) <= 0.0:
                gross = shares * price
                commission = gross * (self.commission_pct / 100.0)
                cash += gross - commission - borrowed
                if cash < 0:
                    cash = 0.0
                qty = shares
                shares = 0.0
                borrowed = 0.0
                entry_price = None
                peak_price = None
                last_sell_idx = i
                margin_call_count += 1
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

            if account_wiped_idx is None and abs(shares) < 1e-9 and cash <= 1e-6:
                account_wiped_idx = i
                account_wiped_date = dates[i] if i < len(dates) else i

            equity_curve[i] = max(0.0, total_equity(price))

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            final_cash=cash,
            final_shares=shares,
            dates=dates,
            entry_thresholds=entry_thr_curve,
            exit_thresholds=exit_thr_curve,
            margin_call_count=margin_call_count,
            account_wiped_at_idx=account_wiped_idx,
            account_wiped_at_date=account_wiped_date,
        )
