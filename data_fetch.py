"""
Market data layer. Uses Binance's PUBLIC REST endpoints only — no API key,
no account access, no trading permissions required. This module never
places, modifies, or cancels orders; it only reads public candle data.

Uses multiple Binance public API endpoints as fallbacks because a deployed
environment may receive HTTP 451 from one endpoint due to geographic/IP
restrictions.
"""

import time
import requests
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Binance public API endpoints
# ---------------------------------------------------------------------------
# Try the primary endpoint first, then Binance's documented alternatives.
BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

KLINES_ENDPOINT = "/api/v3/klines"


COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class DataFetchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Shared Binance request helper
# ---------------------------------------------------------------------------
def _request_klines(params, timeout=10):
    """
    Request Binance klines using multiple public endpoints.

    If an endpoint returns HTTP 451, we treat it as an endpoint/location
    restriction and immediately try the next Binance endpoint.

    Other request errors are also retried against the next endpoint.

    Returns:
        Parsed JSON response from Binance.

    Raises:
        DataFetchError if all endpoints fail.
    """
    errors = []

    for base_url in BINANCE_BASE_URLS:
        url = base_url + KLINES_ENDPOINT

        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
            )

            # HTTP 451 = unavailable for legal/restricted location.
            # Do not waste time retrying the same endpoint.
            if resp.status_code == 451:
                errors.append(
                    f"{base_url}: HTTP 451 (restricted/unavailable location)"
                )
                continue

            resp.raise_for_status()

            return resp.json()

        except requests.RequestException as e:
            errors.append(f"{base_url}: {e}")
            continue

        except ValueError as e:
            errors.append(f"{base_url}: invalid JSON response ({e})")
            continue

    error_summary = " | ".join(errors)

    raise DataFetchError(
        f"Binance API unavailable from all endpoints. "
        f"Errors: {error_summary}"
    )


# ---------------------------------------------------------------------------
# Kline parser
# ---------------------------------------------------------------------------
def _parse_klines(raw) -> pd.DataFrame:
    """Shared parsing logic for live and historical klines responses."""

    df = pd.DataFrame(raw, columns=COLUMNS)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
    )

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ]
    ]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live / recent klines
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_klines(
    symbol: str,
    interval: str,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a symbol/interval from Binance public API.

    Cached for 30 seconds so the dashboard doesn't hammer the API on
    every Streamlit rerun.

    Raises DataFetchError on any network/data problem — callers must
    treat that as DATA INVALID and refuse to generate a signal.
    """

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    try:
        raw = _request_klines(
            params=params,
            timeout=10,
        )

    except DataFetchError:
        raise

    except Exception as e:
        raise DataFetchError(
            f"{symbol} {interval}: fetch failed ({e})"
        )

    if not raw or not isinstance(raw, list):
        raise DataFetchError(
            f"{symbol} {interval}: empty/invalid response"
        )

    try:
        df = _parse_klines(raw)

    except Exception as e:
        raise DataFetchError(
            f"{symbol} {interval}: failed to parse response ({e})"
        )

    # Basic integrity checks
    if df.isnull().values.any():
        raise DataFetchError(
            f"{symbol} {interval}: contains missing values"
        )

    if df["close_time"].duplicated().any():
        raise DataFetchError(
            f"{symbol} {interval}: duplicate candles detected"
        )

    return df


# ---------------------------------------------------------------------------
# Historical klines for backtesting
# ---------------------------------------------------------------------------
def get_historical_klines(
    symbol: str,
    interval: str,
    start_dt,
    end_dt,
    limit_per_call: int = 1000,
    sleep_between_calls: float = 0.25,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Paginated historical fetch for backtesting.

    Binance caps a single request at 1000 candles, so long backtest
    windows need multiple calls.

    NOT cached (each backtest window is typically different).

    `progress_cb(fetched_count, chunk_start_time)` is called after each
    successful page, if provided.
    """

    start_ms = int(
        pd.Timestamp(start_dt).timestamp() * 1000
    )

    end_ms = int(
        pd.Timestamp(end_dt).timestamp() * 1000
    )

    cursor = start_ms
    chunks = []

    while cursor < end_ms:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit_per_call,
        }

        try:
            raw = _request_klines(
                params=params,
                timeout=15,
            )

        except DataFetchError as e:
            raise DataFetchError(
                f"{symbol} {interval}: historical fetch failed "
                f"at cursor {cursor} ({e})"
            )

        except Exception as e:
            raise DataFetchError(
                f"{symbol} {interval}: historical fetch failed "
                f"at cursor {cursor} ({e})"
            )

        if not raw:
            break

        try:
            chunk = _parse_klines(raw)

        except Exception as e:
            raise DataFetchError(
                f"{symbol} {interval}: failed to parse historical "
                f"response at cursor {cursor} ({e})"
            )

        if chunk.empty:
            break

        chunks.append(chunk)

        last_close_ms = int(
            chunk["close_time"].iloc[-1].timestamp() * 1000
        )

        if progress_cb:
            progress_cb(
                sum(len(c) for c in chunks),
                chunk["open_time"].iloc[-1],
            )

        # Safety: prevent infinite loop if API returns no progress.
        if last_close_ms <= cursor:
            break

        cursor = last_close_ms + 1

        # Reached the end of available data.
        if len(raw) < limit_per_call:
            break

        time.sleep(sleep_between_calls)

    # No data
    if not chunks:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
            ]
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(subset="close_time")
        .sort_values("close_time")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------------
# Data health check
# ---------------------------------------------------------------------------
def data_health_check(
    df: pd.DataFrame,
    interval: str,
) -> str:
    """
    Returns:
        'DATA OK'
        'DATA DELAYED'
        'DATA INVALID'

    Delay is judged against how stale the last CLOSED candle looks
    relative to the interval duration.
    """

    if df is None or df.empty:
        return "DATA INVALID"

    interval_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }.get(interval, 300)

    last_close = df["close_time"].iloc[-1]

    age = (
        pd.Timestamp.utcnow().tz_localize(None)
        - last_close
    ).total_seconds()

    if age > interval_seconds * 3:
        return "DATA DELAYED"

    return "DATA OK"


# ---------------------------------------------------------------------------
# Closed candles only
# ---------------------------------------------------------------------------
def only_closed_candles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop the last candle if it has not closed yet, so signals are only
    ever generated from confirmed, closed candles.
    """

    if df.empty:
        return df

    now = pd.Timestamp.utcnow().tz_localize(None)

    if df["close_time"].iloc[-1] > now:
        return df.iloc[:-1].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Historical point-in-time slicing
# ---------------------------------------------------------------------------
def slice_as_of(
    df: pd.DataFrame,
    timestamp,
) -> pd.DataFrame:
    """
    Return only candles fully closed by `timestamp`.

    Core anti-look-ahead rule used throughout the backtester.
    """

    if df is None or df.empty:
        return df

    return df[
        df["close_time"] <= timestamp
    ].reset_index(drop=True)
