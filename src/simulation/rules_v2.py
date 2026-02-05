"""
Improved trading signal rules: supports long, short, and partial positions.
"""

from typing import Literal

Signal = Literal["buy", "sell", "hold"]


def signal_from_prediction_v2(
    current_price: float,
    predicted_price: float,
    current_shares: float,  # Can be positive (long), negative (short), or 0 (cash)
    entry_threshold_pct: float = 0.5,
    exit_threshold_pct: float = -0.5,
) -> Signal:
    """
    Decide whether to buy, sell, or hold based on prediction.
    Supports both long and short positions.
    
    - If in cash (shares == 0):
      - BUY if prediction is significantly above current (long position)
      - SELL if prediction is significantly below current (short position)
    
    - If long (shares > 0):
      - SELL if prediction drops (close long position)
      - Can also SELL MORE to go short if prediction is very negative
    
    - If short (shares < 0):
      - BUY if prediction rises (close short position)
      - Can also BUY MORE to go long if prediction is very positive
    
    Args:
        current_price: Current asset price.
        predicted_price: Model-predicted price for next period.
        current_shares: Current position (positive=long, negative=short, 0=cash).
        entry_threshold_pct: Entry threshold in %.
        exit_threshold_pct: Exit threshold in %.
    
    Returns:
        "buy" | "sell" | "hold"
    """
    if abs(current_price) < 1e-9:
        return "hold"
    
    pct_change = 100 * (predicted_price - current_price) / abs(current_price)
    
    is_long = current_shares > 0
    is_short = current_shares < 0
    is_cash = abs(current_shares) < 1e-9
    
    if is_cash:
        # In cash: enter long if prediction is high, short if prediction is low
        if pct_change >= entry_threshold_pct:
            return "buy"  # Go long
        elif pct_change <= exit_threshold_pct:
            return "sell"  # Go short (sell without owning)
        return "hold"
    
    elif is_long:
        # Long position: exit if prediction drops
        if pct_change <= exit_threshold_pct:
            return "sell"  # Close long
        elif pct_change > 0 and pct_change < 3.0:
            return "sell"  # Momentum weakening
        return "hold"
    
    else:  # is_short
        # Short position: exit if prediction rises
        if pct_change >= -exit_threshold_pct:  # If prediction rises, close short
            return "buy"  # Close short (buy to cover)
        return "hold"
