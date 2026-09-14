"""
Market regime detection. Classifies the CURRENT state of a market before
any signal is generated. Strategy behavior downstream must adapt to this.
"""
import numpy as np
import pandas as pd


def classify_regime(df: pd.DataFrame, cfg) -> dict:
    """
    Uses EMA structure, ADX, and ATR-based volatility percentile to assign
    one of: Strong Bull Trend, Weak Bull Trend, Strong Bear Trend,
    Weak Bear Trend, Range, High Volatility, Low Volatility, Unclear.
    """
    last = df.iloc[-1]
    close = last["close"]

    ema_cols = [c for c in df.columns if c.startswith("ema_")]
    if not ema_cols or df[ema_cols].iloc[-1].isnull().any() or pd.isna(last.get("adx")):
        return {"regime": "UNCLEAR", "reason": "insufficient indicator history"}

    ema_vals = {c: last[c] for c in ema_cols}
    sorted_emas = sorted(ema_vals.items(), key=lambda kv: int(kv[0].split("_")[1]))
    ema_prices = [v for _, v in sorted_emas]  # ordered short->long period

    bullish_stack = all(ema_prices[i] >= ema_prices[i + 1] for i in range(len(ema_prices) - 1)) and close > ema_prices[0]
    bearish_stack = all(ema_prices[i] <= ema_prices[i + 1] for i in range(len(ema_prices) - 1)) and close < ema_prices[0]

    adx = last["adx"]
    atr_pct = (df["atr"] / df["close"]).dropna()
    if atr_pct.empty:
        vol_percentile = 0.5
    else:
        vol_percentile = (atr_pct.rank(pct=True)).iloc[-1]

    volatility_flag = None
    if vol_percentile >= 0.85:
        volatility_flag = "HIGH_VOLATILITY"
    elif vol_percentile <= 0.15:
        volatility_flag = "LOW_VOLATILITY"

    if adx >= cfg.adx_strong_threshold and bullish_stack:
        regime = "STRONG_BULL_TREND"
    elif adx >= cfg.adx_trend_threshold and bullish_stack:
        regime = "WEAK_BULL_TREND"
    elif adx >= cfg.adx_strong_threshold and bearish_stack:
        regime = "STRONG_BEAR_TREND"
    elif adx >= cfg.adx_trend_threshold and bearish_stack:
        regime = "WEAK_BEAR_TREND"
    elif adx < cfg.adx_trend_threshold:
        regime = "RANGE"
    else:
        regime = "UNCLEAR"

    return {
        "regime": regime,
        "adx": round(float(adx), 1),
        "volatility_flag": volatility_flag,
        "vol_percentile": round(float(vol_percentile), 2),
        "ema_bullish_stack": bullish_stack,
        "ema_bearish_stack": bearish_stack,
    }
