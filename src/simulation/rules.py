"""
Trading signal rules from model predictions.
Return 'buy', 'sell' or 'hold' based on current price, predicted price and thresholds.
"""

from typing import Literal

Signal = Literal["buy", "sell", "hold"]


def signal_from_prediction(
    current_price: float,
    predicted_price: float,
    in_position: bool,
    entry_threshold_pct: float = 0.5,
    exit_threshold_pct: float = -0.5,
) -> Signal:
    """
    Decide whether to buy, sell or hold based on the prediction.

    - In cash: buy when pred >= current * (1 + entry_threshold_pct/100).
    - In position: sell when pred <= current * (1 + exit_threshold_pct/100).
    - Otherwise: hold.

    Works correctly for both positive and negative prices (normalized space).

    Args:
        current_price: Current (actual) asset price (can be negative in normalized space).
        predicted_price: Model-predicted price for the next period.
        in_position: Whether we currently hold shares (True) or are in cash (False).
        entry_threshold_pct: Entry threshold in % (e.g. 0.5 = enter if prediction is 0.5% above current).
        exit_threshold_pct: Exit threshold in % (e.g. -0.5 = exit if prediction is 0.5% below current).

    Returns:
        "buy" | "sell" | "hold"
    """
    if abs(current_price) < 1e-9:  # Avoid division by zero
        return "hold"

    # For normalized prices (can be negative), use relative change correctly
    # If current is negative, multiplying by (1 + pct) reverses the logic
    # So we compute the actual % change: (pred - current) / abs(current)
    pct_change = 100 * (predicted_price - current_price) / abs(current_price)
    
    if in_position:
        # Exit if predicted change drops significantly
        # exit_threshold_pct is negative (e.g. -5.0), meaning sell if pct_change <= -5.0
        # But if predictions are always positive, we need alternative logic:
        # Sell if pct_change drops below a minimum threshold (e.g. < 5% when it was higher)
        # For now, keep original logic: sell if prediction drops by threshold amount
        if pct_change <= exit_threshold_pct:
            return "sell"
        # Alternative: also sell if pct_change is positive but very small (< 1%)
        # This handles cases where prediction is always above price but momentum weakens
        if pct_change > 0 and pct_change < 1.0:
            return "sell"
        return "hold"
    
    # Entry if predicted change is above entry threshold
    if pct_change >= entry_threshold_pct:
        return "buy"
    return "hold"
