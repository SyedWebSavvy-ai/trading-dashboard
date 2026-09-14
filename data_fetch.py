"""
Market data layer. Uses Binance's PUBLIC REST endpoints only — no API key,
no account access, no trading permissions required. This module never
places, modifies, or cancels orders; it only reads public candle data.
"""
import time
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


class DataFetchError(Exception):
    pass


def _parse_klines(raw) -> pd.DataFrame:
    """Shared parsing logic for both live and historical klines responses."""
    df = pd.DataFrame(raw, columns=COLUMNS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    return df.reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a symbol/interval from Binance public API.
    Cached for 30s so the dashboard doesn't hammer the API on every rerun.
    Raises DataFetchError on any network/data problem — callers must treat
    that as DATA INVALID and refuse to generate a signal.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        raise DataFetchError(f"{symbol} {interval}: fetch failed ({e})")

    if not raw or not isinstance(raw, list):
        raise DataFetchError(f"{symbol} {interval}: empty/invalid response")

    df = _parse_klines(raw)

    # Basic integrity checks
    if df.isnull().values.any():
        raise DataFetchError(f"{symbol} {interval}: contains missing values")
    if df["close_time"].duplicated().any():
        raise DataFetchError(f"{symbol} {interval}: duplicate candles detected")

    return df


def get_historical_klines(symbol: str, interval: str, start_dt, end_dt,
                           limit_per_call: int = 1000, sleep_between_calls: float = 0.25,
                           progress_cb=None) -> pd.DataFrame:
    """
    Paginated historical fetch for backtesting — Binance caps a single
    request at 1000 candles, so long backtest windows need multiple calls.
    NOT cached (each backtest window is typically different) and NOT rate-
    limit-optimized beyond a small sleep between calls — this is a research
    tool, not a high-frequency data pipeline.

    `progress_cb(fetched_count, chunk_start_time)` is called after each
    successful page, if provided, so callers (e.g. a Streamlit progress bar)
    can show status during long fetches.
    """
    start_ms = int(pd.Timestamp(start_dt).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_dt).timestamp() * 1000)
    cursor = start_ms
    chunks = []

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": limit_per_call,
        }
        try:
            resp = requests.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            raise DataFetchError(f"{symbol} {interval}: historical fetch failed at cursor {cursor} ({e})")

        if not raw:
            break

        chunk = _parse_klines(raw)
        chunks.append(chunk)
        last_close_ms = int(chunk["close_time"].iloc[-1].timestamp() * 1000)

        if progress_cb:
            progress_cb(sum(len(c) for c in chunks), chunk["open_time"].iloc[-1])

        if last_close_ms <= cursor:
            break  # safety: prevent infinite loop if API returns no progress
        cursor = last_close_ms + 1

        if len(raw) < limit_per_call:
            break  # reached the end of available data

        time.sleep(sleep_between_calls)

    if not chunks:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="close_time").sort_values("close_time").reset_index(drop=True)
    return df


def data_health_check(df: pd.DataFrame, interval: str) -> str:
    """
    Returns 'DATA OK', 'DATA DELAYED', or 'DATA INVALID'.
    Delay is judged against how stale the last CLOSED candle looks
    relative to the interval duration.
    """
    if df is None or df.empty:
        return "DATA INVALID"

    interval_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400,
    }.get(interval, 300)

    last_close = df["close_time"].iloc[-1]
    age = (pd.Timestamp.utcnow().tz_localize(None) - last_close).total_seconds()

    if age > interval_seconds * 3:
        return "DATA DELAYED"
    return "DATA OK"


def only_closed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the last candle if it has not closed yet, so signals are only ever
    generated from confirmed, closed candles (never an in-progress candle).
    """
    if df.empty:
        return df
    now = pd.Timestamp.utcnow().tz_localize(None)
    if df["close_time"].iloc[-1] > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def slice_as_of(df: pd.DataFrame, timestamp) -> pd.DataFrame:
    """
    Return only candles fully closed by `timestamp` — the core anti-look-
    ahead rule used throughout the backtester. A candle is only "known" once
    its close_time has passed; using anything later would leak future
    information into a historical decision.
    """
    if df is None or df.empty:
        return df
    return df[df["close_time"] <= timestamp].reset_index(drop=True)

