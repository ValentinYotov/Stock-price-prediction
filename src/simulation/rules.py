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

    Args:
        current_price: Current (actual) asset price.
        predicted_price: Model-predicted price for the next period.
        in_position: Whether we currently hold shares (True) or are in cash (False).
        entry_threshold_pct: Entry threshold in % (e.g. 0.5 = enter if prediction is 0.5% above current).
        exit_threshold_pct: Exit threshold in % (e.g. -0.5 = exit if prediction is 0.5% below current).

    Returns:
        "buy" | "sell" | "hold"
    """
    if current_price <= 0:
        return "hold"

    entry_bound = current_price * (1 + entry_threshold_pct / 100.0)
    exit_bound = current_price * (1 + exit_threshold_pct / 100.0)

    if in_position:
        if predicted_price <= exit_bound:
            return "sell"
        return "hold"

    if predicted_price >= entry_bound:
        return "buy"
    return "hold"
