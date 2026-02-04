"""
Правила за търговски сигнали от предсказания на модела.
Връщат 'buy', 'sell' или 'hold' според текуща цена, предсказана цена и прагове.
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
    Решава дали да купим, продадем или да държим според предсказанието.

    - Ако сме в кеш: купуваме (buy), когато pred >= current * (1 + entry_threshold_pct/100).
    - Ако сме в позиция: продаваме (sell), когато pred <= current * (1 + exit_threshold_pct/100).
    - Иначе: hold.

    Параметри:
        current_price: Текуща (реална) цена на актива.
        predicted_price: Предсказана от модела цена за следващия период.
        in_position: Дали в момента държим акции (True) или сме в кеш (False).
        entry_threshold_pct: Праг в % за влизане (напр. 0.5 = влизаме ако предсказанието е с 0.5% над текущата).
        exit_threshold_pct: Праг в % за излизане (напр. -0.5 = излизаме ако предсказанието е с 0.5% под текущата).

    Връща:
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

    # В кеш – търсим влизане
    if predicted_price >= entry_bound:
        return "buy"
    return "hold"
