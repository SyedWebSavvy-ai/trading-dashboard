"""
Market structure analysis: swing points, trend structure (HH/HL vs LH/LL),
break of structure, and support/resistance ZONES (not exact prices).
Avoids simplistic 'price crossed EMA = signal' logic.
"""
import pandas as pd
import numpy as np


def find_swings(df: pd.DataFrame, lookback: int = 3):
    """
    Fractal-style swing detection: a candle is a swing high if its high is
    the max within +/- lookback candles (same logic for swing lows).
    Returns lists of (index, price) tuples.
    """
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = h[i - lookback: i + lookback + 1]
        window_l = l[i - lookback: i + lookback + 1]
        if h[i] == window_h.max() and np.argmax(window_h) == lookback:
            highs.append((i, h[i]))
        if l[i] == window_l.min() and np.argmin(window_l) == lookback:
            lows.append((i, l[i]))
    return highs, lows


def classify_structure(highs, lows):
    """
    Looks at the last two swing highs and last two swing lows to classify
    the structure as bullish (HH+HL), bearish (LH+LL), or mixed/unclear.
    """
    if len(highs) < 2 or len(lows) < 2:
        return "UNCLEAR", None

    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]

    last_swing_time = max(highs[-1][0], lows[-1][0])

    if hh and hl:
        return "BULLISH_STRUCTURE", last_swing_time
    if lh and ll:
        return "BEARISH_STRUCTURE", last_swing_time
    return "MIXED_STRUCTURE", last_swing_time


def detect_bos(df: pd.DataFrame, highs, lows):
    """
    Break of Structure: current close breaks beyond the most recent
    confirmed swing high (bullish BOS) or swing low (bearish BOS).
    """
    if not highs or not lows:
        return None
    last_close = df["close"].iloc[-1]
    last_high = highs[-1][1]
    last_low = lows[-1][1]

    if last_close > last_high:
        return "BULLISH_BOS"
    if last_close < last_low:
        return "BEARISH_BOS"
    return None


def support_resistance_zones(df: pd.DataFrame, highs, lows, zone_width_pct=0.0025, top_n=4):
    """
    Clusters recent swing points into zones. A zone's strength increases
    when multiple independent swings land close together (confluence),
    which is a stronger, more realistic construct than an exact price.
    """
    points = [p for _, p in highs[-15:]] + [p for _, p in lows[-15:]]
    if not points:
        return []

    points = sorted(points)
    zones = []
    current_zone = [points[0]]

    for p in points[1:]:
        if abs(p - current_zone[-1]) / current_zone[-1] <= zone_width_pct:
            current_zone.append(p)
        else:
            zones.append(current_zone)
            current_zone = [p]
    zones.append(current_zone)

    zone_summaries = [
        {"low": min(z), "high": max(z), "mid": sum(z) / len(z), "touches": len(z)}
        for z in zones
    ]
    zone_summaries.sort(key=lambda z: z["touches"], reverse=True)
    return zone_summaries[:top_n]


def nearest_zone_distance_pct(price, zones):
    """Distance (%) from current price to the nearest S/R zone midpoint."""
    if not zones:
        return None
    dists = [abs(price - z["mid"]) / price for z in zones]
    return min(dists) * 100


def analyze_structure(df: pd.DataFrame, swing_lookback: int = 3):
    highs, lows = find_swings(df, swing_lookback)
    structure, _ = classify_structure(highs, lows)
    bos = detect_bos(df, highs, lows)
    zones = support_resistance_zones(df, highs, lows)
    return {
        "highs": highs,
        "lows": lows,
        "structure": structure,
        "bos": bos,
        "zones": zones,
    }
