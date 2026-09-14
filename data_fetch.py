"""
Market data layer.

Uses Binance PUBLIC market-data REST endpoints only.
No API key, no account access, and no trading permissions are required.

This module ONLY reads public candle data.
It never places, modifies, or cancels orders.

Primary endpoint:
    https://data-api.binance.vision

Fallback endpoints:
    https://api.binance.com
    https://api1.binance.com
    https://api2.binance.com
    https://api3.binance.com
    https://api4.binance.com

The Binance market-data mirror is preferred because it provides the same
Binance Spot kline format while avoiding the geo-restriction that may affect
api.binance.com from hosted environments.
"""

import time

import requests
import pandas as pd
import streamlit as st


# ============================================================================
# BINANCE PUBLIC MARKET-DATA ENDPOINTS
# ============================================================================

# IMPORTANT:
# Put Binance's dedicated public market-data endpoint FIRST.
#
# Binance officially documents this host for public market data including
# /api/v3/klines.
#
# If this endpoint is unavailable, we fall back to the normal Binance
# endpoints.
BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

KLINES_ENDPOINT = "/api/v3/klines"


# ============================================================================
# EXPECTED BINANCE KLINE COLUMNS
# ============================================================================

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


# ============================================================================
# CUSTOM ERROR
# ============================================================================

class DataFetchError(Exception):
    pass


# ============================================================================
# BINANCE REQUEST HELPER
# ============================================================================

def _request_klines(params, timeout=10):
    """
    Fetch Binance klines using the dedicated Binance market-data endpoint
    first, followed by fallback endpoints.

    The function returns the raw JSON response.

    It raises DataFetchError only if every endpoint fails.

    IMPORTANT:
    HTTP 451 is NOT retried repeatedly against the same endpoint.
    We immediately move to the next endpoint.
    """

    errors = []

    for base_url in BINANCE_BASE_URLS:

        url = base_url + KLINES_ENDPOINT

        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    "User-Agent": "TradingDashboard/1.0",
                    "Accept": "application/json",
                },
            )

            # ---------------------------------------------------------------
            # 451 = restricted/unavailable location
            # ---------------------------------------------------------------
            if response.status_code == 451:
                errors.append(
                    f"{base_url}: HTTP 451 (restricted/unavailable location)"
                )
                continue

            # ---------------------------------------------------------------
            # Rate limiting
            # ---------------------------------------------------------------
            if response.status_code in (418, 429):
                retry_after = response.headers.get("Retry-After")

                errors.append(
                    f"{base_url}: HTTP {response.status_code}"
                    + (
                        f" (Retry-After: {retry_after}s)"
                        if retry_after
                        else ""
                    )
                )

                # Do not hammer the endpoint.
                continue

            # ---------------------------------------------------------------
            # Other HTTP errors
            # ---------------------------------------------------------------
            response.raise_for_status()

            # ---------------------------------------------------------------
            # Parse JSON
            # ---------------------------------------------------------------
            try:
                return response.json()

            except ValueError as e:
                errors.append(
                    f"{base_url}: invalid JSON response ({e})"
                )
                continue

        except requests.Timeout:
            errors.append(
                f"{base_url}: request timed out"
            )
            continue

        except requests.ConnectionError as e:
            errors.append(
                f"{base_url}: connection error ({e})"
            )
            continue

        except requests.RequestException as e:
            errors.append(
                f"{base_url}: request error ({e})"
            )
            continue

        except Exception as e:
            errors.append(
                f"{base_url}: unexpected error ({e})"
            )
            continue

    error_summary = " | ".join(errors)

    raise DataFetchError(
        "Binance API unavailable from all endpoints. "
        f"Errors: {error_summary}"
    )


# ============================================================================
# KLINE PARSER
# ============================================================================

def _parse_klines(raw) -> pd.DataFrame:
    """
    Convert Binance raw kline response into the DataFrame structure expected
    by the rest of the application.
    """

    if not isinstance(raw, list):
        raise ValueError("Kline response is not a list")

    if not raw:
        return pd.DataFrame(columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ])

    df = pd.DataFrame(
        raw,
        columns=COLUMNS,
    )

    # ------------------------------------------------------------------------
    # Convert numeric market-data columns
    # ------------------------------------------------------------------------

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # ------------------------------------------------------------------------
    # Convert timestamps
    # ------------------------------------------------------------------------

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        errors="coerce",
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # Keep only columns used by the application
    # ------------------------------------------------------------------------

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


# ============================================================================
# LIVE / RECENT KLINES
# ============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def get_klines(
    symbol: str,
    interval: str,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a Binance Spot symbol and interval.

    The result is cached for 30 seconds to avoid repeatedly requesting the
    same candles during Streamlit reruns.

    IMPORTANT:
    If market data cannot be fetched or fails integrity checks, this function
    raises DataFetchError. Callers should treat that as DATA INVALID and
    must not generate a trading signal from invalid data.
    """

    symbol = str(symbol).upper().strip()
    interval = str(interval).strip()

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
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

    # ------------------------------------------------------------------------
    # Basic response validation
    # ------------------------------------------------------------------------

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

    if df.empty:
        raise DataFetchError(
            f"{symbol} {interval}: no candle data returned"
        )

    # ------------------------------------------------------------------------
    # Data integrity checks
    # ------------------------------------------------------------------------

    if df.isnull().values.any():
        raise DataFetchError(
            f"{symbol} {interval}: contains missing values"
        )

    if df["close_time"].duplicated().any():
        raise DataFetchError(
            f"{symbol} {interval}: duplicate candles detected"
        )

    # OHLC sanity checks
    if (df["high"] < df["low"]).any():
        raise DataFetchError(
            f"{symbol} {interval}: invalid high/low values"
        )

    if (df["high"] < df["open"]).any():
        raise DataFetchError(
            f"{symbol} {interval}: high below open"
        )

    if (df["high"] < df["close"]).any():
        raise DataFetchError(
            f"{symbol} {interval}: high below close"
        )

    if (df["low"] > df["open"]).any():
        raise DataFetchError(
            f"{symbol} {interval}: low above open"
        )

    if (df["low"] > df["close"]).any():
        raise DataFetchError(
            f"{symbol} {interval}: low above close"
        )

    if (df["volume"] < 0).any():
        raise DataFetchError(
            f"{symbol} {interval}: negative volume detected"
        )

    return df


# ============================================================================
# HISTORICAL KLINES FOR BACKTESTING
# ============================================================================

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
    Paginated historical Binance Spot kline fetch for backtesting.

    Binance allows up to 1000 candles per request, so longer backtest
    windows require multiple requests.

    This function is intentionally NOT cached because different backtest
    windows normally require different data.

    progress_cb(fetched_count, last_open_time) is called after each
    successful page if supplied.
    """

    symbol = str(symbol).upper().strip()
    interval = str(interval).strip()

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)

    # ------------------------------------------------------------------------
    # Convert to milliseconds
    # ------------------------------------------------------------------------

    start_ms = int(
        start_ts.timestamp() * 1000
    )

    end_ms = int(
        end_ts.timestamp() * 1000
    )

    if start_ms >= end_ms:
        raise DataFetchError(
            f"{symbol} {interval}: start_dt must be before end_dt"
        )

    cursor = start_ms
    chunks = []

    while cursor < end_ms:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": int(limit_per_call),
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

        # No more data
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

        # --------------------------------------------------------------------
        # Validate historical chunk
        # --------------------------------------------------------------------

        if chunk.isnull().values.any():
            raise DataFetchError(
                f"{symbol} {interval}: historical data contains "
                f"missing values at cursor {cursor}"
            )

        if chunk["close_time"].duplicated().any():
            raise DataFetchError(
                f"{symbol} {interval}: duplicate historical candles "
                f"detected at cursor {cursor}"
            )

        chunks.append(chunk)

        # --------------------------------------------------------------------
        # Determine next cursor
        # --------------------------------------------------------------------

        last_close_ms = int(
            chunk["close_time"].iloc[-1].timestamp() * 1000
        )

        # Progress callback
        if progress_cb:
            progress_cb(
                sum(len(c) for c in chunks),
                chunk["open_time"].iloc[-1],
            )

        # Safety against infinite loops
        if last_close_ms <= cursor:
            break

        cursor = last_close_ms + 1

        # If Binance returned fewer candles than requested,
        # we've reached the end of the available range.
        if len(raw) < int(limit_per_call):
            break

        # Small delay between historical requests.
        time.sleep(sleep_between_calls)

    # ------------------------------------------------------------------------
    # No data
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Combine pages
    # ------------------------------------------------------------------------

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    # Remove any accidental duplicate candles and sort chronologically.
    df = (
        df
        .drop_duplicates(
            subset="close_time",
        )
        .sort_values(
            "close_time",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================================
# DATA HEALTH CHECK
# ============================================================================

def data_health_check(
    df: pd.DataFrame,
    interval: str,
) -> str:
    """
    Returns:

        DATA OK
        DATA DELAYED
        DATA INVALID

    Delay is judged against the age of the last closed candle relative
    to the interval duration.
    """

    if df is None or df.empty:
        return "DATA INVALID"

    interval_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }.get(
        interval,
        300,
    )

    last_close = df["close_time"].iloc[-1]

    now = pd.Timestamp.utcnow().tz_localize(None)

    age = (
        now - last_close
    ).total_seconds()

    if age > interval_seconds * 3:
        return "DATA DELAYED"

    return "DATA OK"


# ============================================================================
# CLOSED CANDLES ONLY
# ============================================================================

def only_closed_candles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove the last candle if it is still in progress.

    Signals should only use fully closed candles.
    """

    if df is None or df.empty:
        return df

    now = pd.Timestamp.utcnow().tz_localize(None)

    if df["close_time"].iloc[-1] > now:
        return df.iloc[:-1].reset_index(drop=True)

    return df


# ============================================================================
# POINT-IN-TIME BACKTEST SLICING
# ============================================================================

def slice_as_of(
    df: pd.DataFrame,
    timestamp,
) -> pd.DataFrame:
    """
    Return only candles fully closed by `timestamp`.

    This is the anti-look-ahead rule for historical backtesting.
    """

    if df is None or df.empty:
        return df

    return df[
        df["close_time"] <= timestamp
    ].reset_index(drop=True)
