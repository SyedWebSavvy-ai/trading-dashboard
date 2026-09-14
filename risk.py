"""
Risk management. Position sizing, stop-loss placement, and take-profit
targets are all derived from account equity, structure, and volatility —
never an arbitrary fixed percentage of price.
"""
import numpy as np


def calculate_stop_loss(direction: str, entry_price: float, atr: float,
                         structure_zones, atr_multiplier: float):
    """
    SL = structural invalidation point +/- ATR buffer.
    Falls back to a pure ATR-based stop if no usable structure zone exists.
    """
    buffer = atr * atr_multiplier

    if direction == "BUY":
        candidate_zones = [z for z in structure_zones if z["mid"] < entry_price]
        if candidate_zones:
            nearest = max(candidate_zones, key=lambda z: z["mid"])
            return round(nearest["low"] - buffer, 6), "structure + ATR buffer"
        return round(entry_price - buffer * 2, 6), "ATR-based (no structure zone available)"
    else:
        candidate_zones = [z for z in structure_zones if z["mid"] > entry_price]
        if candidate_zones:
            nearest = min(candidate_zones, key=lambda z: z["mid"])
            return round(nearest["high"] + buffer, 6), "structure + ATR buffer"
        return round(entry_price + buffer * 2, 6), "ATR-based (no structure zone available)"


def calculate_take_profits(direction: str, entry_price: float, stop_loss: float,
                            structure_zones):
    """
    TP1 = 1R, TP2 = 2R are scale-out levels, NOT the risk/reward gate — they are
    fixed R-multiples by construction, so measuring "risk/reward" against them
    is circular (it would always equal exactly 1.0 or 2.0 no matter the setup).

    The REAL profit target — used for the risk/reward quality gate — is the
    nearest meaningful opposing structure zone beyond entry (not the furthest
    one; picking the furthest zone as in the old version could select an
    unrealistically distant target). If no structure zone exists, we fall
    back to a conservative 3R projection and flag that explicitly so callers
    know the RR figure is a default assumption rather than market-derived.

    Returns (tp1, tp2, tp3, target_source) where target_source explains
    where tp3 (the primary/realistic target) came from.
    """
    risk_distance = abs(entry_price - stop_loss)

    if direction == "BUY":
        tp1 = entry_price + risk_distance * 1
        tp2 = entry_price + risk_distance * 2
        opposing = [z for z in structure_zones if z["mid"] > entry_price]
        if opposing:
            nearest_target = min(opposing, key=lambda z: z["mid"])
            tp3 = nearest_target["mid"]
            target_source = f"nearest opposing structure zone ({nearest_target['touches']} touches)"
        else:
            tp3 = entry_price + risk_distance * 3
            target_source = "3R fallback — no structure zone available, RR is an assumption"
    else:
        tp1 = entry_price - risk_distance * 1
        tp2 = entry_price - risk_distance * 2
        opposing = [z for z in structure_zones if z["mid"] < entry_price]
        if opposing:
            nearest_target = max(opposing, key=lambda z: z["mid"])
            tp3 = nearest_target["mid"]
            target_source = f"nearest opposing structure zone ({nearest_target['touches']} touches)"
        else:
            tp3 = entry_price - risk_distance * 3
            target_source = "3R fallback — no structure zone available, RR is an assumption"

    return round(tp1, 6), round(tp2, 6), round(tp3, 6), target_source


def calculate_position_size(equity: float, risk_pct: float, entry_price: float, stop_loss: float):
    """
    Position size = (equity * risk%) / stop-loss distance.
    Returns (position_size_in_units, position_value_usd, dollar_risk).
    """
    dollar_risk = equity * (risk_pct / 100)
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return 0.0, 0.0, dollar_risk
    position_size = dollar_risk / stop_distance
    position_value = position_size * entry_price
    return round(position_size, 6), round(position_value, 2), round(dollar_risk, 2)


def risk_reward_ratio(entry_price: float, stop_loss: float, target_price: float):
    """
    RR must be computed against a REAL target (the structure-derived tp3),
    never against TP1/TP2 — those are defined as fixed R-multiples of the
    stop distance, so RR against them is always exactly 1.0 / 2.0 by
    construction and tells you nothing about the actual setup quality.
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def correlation_matrix(price_series_dict):
    """
    price_series_dict: {symbol: pd.Series of closes}
    Returns a correlation matrix (pandas DataFrame) of returns.
    """
    import pandas as pd
    returns = {sym: s.pct_change().dropna() for sym, s in price_series_dict.items()}
    df = pd.DataFrame(returns)
    return df.corr()
