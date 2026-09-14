"""
Technical analysis engine. Each function adds columns to the dataframe.
These are treated as EVIDENCE fed into the scoring engine (see scoring.py) —
never used standalone as "indicator X says Y therefore BUY".
"""
import pandas as pd
import numpy as np
import ta


def add_trend_indicators(df: pd.DataFrame, ema_periods) -> pd.DataFrame:
    df = df.copy()
    for p in ema_periods:
        df[f"ema_{p}"] = ta.trend.ema_indicator(df["close"], window=p)
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    return df


def add_momentum_indicators(df: pd.DataFrame, rsi_period=14, stoch_period=14) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_period).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"], window=stoch_period)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()
    return df


def add_volatility_indicators(df: pd.DataFrame, atr_period=14, bb_period=20, bb_std=2.0) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=atr_period
    ).average_true_range()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_period, window_dev=bb_std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


def add_volume_indicators(df: pd.DataFrame, sma_period=20) -> pd.DataFrame:
    df = df.copy()
    df["volume_sma"] = df["volume"].rolling(sma_period).mean()
    df["relative_volume"] = df["volume"] / df["volume_sma"]
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    return df


def compute_all_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = add_trend_indicators(df, cfg.ema_periods)
    df = add_momentum_indicators(df, cfg.rsi_period, cfg.stoch_rsi_period)
    df = add_volatility_indicators(df, cfg.atr_period, cfg.bb_period, cfg.bb_std)
    df = add_volume_indicators(df, cfg.volume_sma_period)
    return df
