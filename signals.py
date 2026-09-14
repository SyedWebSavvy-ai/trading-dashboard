"""
Final decision engine. Orchestrates data -> indicators -> structure ->
regime -> scoring -> risk into a single signal object.

Pipeline order (mirrors spec section 39):
data validation -> regime -> HTF analysis -> structure -> indicator
confirmation -> volume confirmation -> directional-edge gate ->
risk/reward filter -> final score.

Any critical validation failure short-circuits straight to NO TRADE.

IMPORTANT: generate_signal_from_frames() is the single source of truth for
the decision logic. Both the live app (generate_signal, which fetches fresh
data) and the backtester (backtest.py, which slices historical data at each
point in time) call this SAME function. If live and backtest ever used
separate copies of this logic, a backtest result would tell you nothing
about what the live system actually does — that mismatch is one of the most
common ways trading research quietly lies to you.
"""
from datetime import datetime, timedelta

from data_fetch import get_klines, data_health_check, only_closed_candles, DataFetchError
from indicators import compute_all_indicators
from market_structure import analyze_structure
from regime import classify_regime
from scoring import (
    score_trend, score_structure, score_momentum, score_volume,
    score_support_resistance, score_volatility, score_mtf_confirmation,
    combine_scores, score_band_label,
)
from risk import calculate_stop_loss, calculate_take_profits, calculate_position_size, risk_reward_ratio


def _empty_result(symbol, now, cfg):
    return {
        "symbol": symbol,
        "timestamp": now,
        "direction": "NO TRADE",
        "score": 0.0,
        "bull_score": 0.0,
        "bear_score": 0.0,
        "directional_edge": 0.0,
        "confidence_label": "NO TRADE",
        "data_status": "DATA OK",
        "reasons_supporting": [],
        "reasons_against": [],
        "invalidation": None,
        "entry_zone": None,
        "stop_loss": None,
        "tp1": None, "tp2": None, "tp3": None,
        "target_source": None,
        "rr": None,
        "regime": None,
        "expiry": now + timedelta(minutes=cfg.signal_expiry_minutes),
        "position_size": None,
        "position_value": None,
        "dollar_risk": None,
        "structure_zones": [],
        "current_price": None,
    }


def generate_signal_from_frames(symbol: str, cfg, htf_df, mtf_df, ltf_df, entry_df,
                                 as_of=None) -> dict:
    """
    Pure decision function: takes already-fetched, already-indicator-computed
    dataframes for each timeframe and returns a signal dict. Does no network
    I/O, so it works identically whether called live or from history.

    `as_of` is used only for the timestamp field (live: now; backtest: the
    simulated decision time) — it never affects which rows are looked at;
    that's entirely the caller's responsibility (only pass rows that would
    actually have been visible/closed by that point in time).
    """
    now = as_of or datetime.utcnow()
    result = _empty_result(symbol, now, cfg)

    for label, df in (("htf", htf_df), ("mtf", mtf_df), ("ltf", ltf_df), ("entry", entry_df)):
        if df is None or len(df) < 60:
            result["data_status"] = "DATA INVALID"
            result["reasons_against"] = [f"Insufficient closed-candle history on {label}"]
            return result

    current_price = entry_df["close"].iloc[-1]
    result["current_price"] = round(float(current_price), 6)

    # ---- 2. Market regime (on HTF) ----
    regime_info = classify_regime(htf_df, cfg)
    result["regime"] = regime_info["regime"]

    if regime_info["regime"] == "UNCLEAR":
        result["reasons_against"] = ["Higher-timeframe regime is unclear — insufficient confluence"]
        return result

    # ---- 3. Market structure (on LTF, the "setup confirmation" timeframe) ----
    structure_info = analyze_structure(ltf_df, cfg.swing_lookback)
    result["structure_zones"] = structure_info["zones"]

    # ---- 4. Multi-timeframe bias agreement ----
    htf_regime_bias = "bull" if regime_info["regime"] in ("STRONG_BULL_TREND", "WEAK_BULL_TREND") else (
        "bear" if regime_info["regime"] in ("STRONG_BEAR_TREND", "WEAK_BEAR_TREND") else "neutral"
    )
    mtf_regime = classify_regime(mtf_df, cfg)
    mtf_bias = "bull" if mtf_regime["regime"] in ("STRONG_BULL_TREND", "WEAK_BULL_TREND") else (
        "bear" if mtf_regime["regime"] in ("STRONG_BEAR_TREND", "WEAK_BEAR_TREND") else "neutral"
    )
    ltf_bias = "bull" if structure_info["structure"] == "BULLISH_STRUCTURE" else (
        "bear" if structure_info["structure"] == "BEARISH_STRUCTURE" else "neutral"
    )
    entry_last = entry_df.iloc[-1]
    entry_bias = "bull" if entry_last.get("macd_hist", 0) > 0 else "bear"

    tf_biases = {"4h_htf": htf_regime_bias, "1h_mtf": mtf_bias, "15m_ltf": ltf_bias, "5m_entry": entry_bias}

    # ---- 4b. Critical conflict gate ----
    # "Higher timeframes conflict" is a first-class NO-TRADE reason (spec
    # section 13) and must be a HARD gate, not just a note. Gated on HTF vs
    # MTF (the primary trend-defining timeframes); the entry timeframe is
    # allowed to differ without invalidating a swing setup.
    critical_conflict = (
        htf_regime_bias != "neutral"
        and mtf_bias != "neutral"
        and htf_regime_bias != mtf_bias
    )
    if critical_conflict:
        result["direction"] = "NO TRADE"
        result["regime"] = regime_info["regime"]
        result["reasons_against"] = [
            f"Higher-timeframe conflict: {cfg.htf} is {htf_regime_bias} while "
            f"{cfg.mtf_confirm} is {mtf_bias} — insufficient confluence"
        ]
        result["confirmations"] = tf_biases
        return result

    # OBV trend for volume scoring
    obv_series = ltf_df["obv"].dropna()
    obv_trend = None
    if len(obv_series) >= 10:
        obv_trend = "up" if obv_series.iloc[-1] > obv_series.iloc[-10] else "down"
    ltf_last = ltf_df.iloc[-1].copy()
    ltf_last["obv_trend"] = obv_trend

    # ---- 5. Scoring engine ----
    components = {
        "trend": score_trend(regime_info, mtf_regime),
        "structure": score_structure(structure_info),
        "momentum": score_momentum(ltf_last),
        "volume": score_volume(ltf_last),
        "support_resistance": score_support_resistance(current_price, structure_info["zones"], None),
        "volatility": score_volatility(regime_info),
        "mtf_confirmation": score_mtf_confirmation(tf_biases),
    }
    combined = combine_scores(components, cfg.weights)
    result["direction"] = combined["direction"]
    result["score"] = combined["score"]
    result["bull_score"] = combined["bull_score"]
    result["bear_score"] = combined["bear_score"]
    result["directional_edge"] = round(abs(combined["bull_score"] - combined["bear_score"]), 1)
    result["reasons_supporting"] = combined["reasons_supporting"]
    result["reasons_against"] = combined["reasons_against"]
    result["confidence_label"] = score_band_label(combined["score"], cfg.score_bands)

    # ---- 6. Score threshold gate ----
    if combined["score"] < cfg.min_signal_score:
        result["direction"] = "NO TRADE"
        result["reasons_against"].append(
            f"Score {combined['score']:.0f} is below the minimum threshold ({cfg.min_signal_score:.0f})"
        )
        return result

    # ---- 6b. Directional-edge gate ----
    # A score of e.g. 68 can happen with Bull=68/Bear=64 (a 4-point lean) just
    # as easily as Bull=68/Bear=20 (a decisive lean) — the headline score
    # alone can't tell those apart, but they are very different setups. This
    # requires the winning side to actually beat the losing side by a
    # meaningful, configurable margin, not just be non-negative.
    if result["directional_edge"] < cfg.min_directional_edge:
        result["direction"] = "NO TRADE"
        result["reasons_against"].append(
            f"Directional edge too thin (bull {combined['bull_score']} vs bear {combined['bear_score']}, "
            f"edge {result['directional_edge']} < minimum {cfg.min_directional_edge}) — evidence doesn't "
            f"clearly favor one side"
        )
        return result

    # ---- 7. Secondary (non-critical) conflict note ----
    if components["mtf_confirmation"].get("conflict"):
        result["reasons_against"].append("Some timeframes disagree — reduced confluence")

    direction = combined["direction"]

    # ---- 8. Risk / reward calculation ----
    atr = ltf_last.get("atr")
    if atr is None or atr != atr:  # NaN check
        result["direction"] = "NO TRADE"
        result["reasons_against"].append("ATR unavailable — cannot size stop-loss safely")
        return result

    stop_loss, sl_method = calculate_stop_loss(direction, current_price, atr, structure_info["zones"], cfg.atr_sl_multiplier)
    tp1, tp2, tp3, target_source = calculate_take_profits(direction, current_price, stop_loss, structure_info["zones"])
    # RR is measured against tp3 (the structure-derived realistic target),
    # never tp1/tp2 (fixed R-multiples by construction — RR against those is
    # always exactly 1.0/2.0 and tells you nothing about the actual setup).
    rr = risk_reward_ratio(current_price, stop_loss, tp3)

    if rr < cfg.min_rr:
        result["direction"] = "NO TRADE"
        result["reasons_against"].append(
            f"Risk/reward to the nearest realistic target ({rr}) is below minimum acceptable ({cfg.min_rr}) "
            f"[target source: {target_source}]"
        )
        result["score"] = combined["score"]
        return result

    pos_size, pos_value, dollar_risk = calculate_position_size(cfg.account_equity, cfg.risk_per_trade_pct, current_price, stop_loss)

    # ---- 9. Populate final signal ----
    entry_buffer = atr * 0.15
    result.update({
        "direction": direction,
        "entry_zone": (round(current_price - entry_buffer, 6), round(current_price + entry_buffer, 6)),
        "stop_loss": stop_loss,
        "sl_method": sl_method,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "target_source": target_source,
        "rr": rr,
        "position_size": pos_size,
        "position_value": pos_value,
        "dollar_risk": dollar_risk,
        "invalidation": (
            f"{cfg.ltf_confirm} candle closes beyond {stop_loss} "
            f"({'below support' if direction == 'BUY' else 'above resistance'})"
        ),
        "confirmations": tf_biases,
        "current_price": round(float(current_price), 6),
    })
    return result


def generate_signal(symbol: str, cfg) -> dict:
    """
    Live wrapper: fetches fresh data for all 4 timeframes, validates it, then
    delegates to generate_signal_from_frames() for the actual decision.
    """
    now = datetime.utcnow()
    timeframes = {"htf": cfg.htf, "mtf": cfg.mtf_confirm, "ltf": cfg.ltf_confirm, "entry": cfg.entry_tf}
    frames = {}
    data_status = "DATA OK"

    for label, tf in timeframes.items():
        try:
            raw = get_klines(symbol, tf, cfg.candles_per_tf)
        except DataFetchError as e:
            result = _empty_result(symbol, now, cfg)
            result["data_status"] = "DATA INVALID"
            result["reasons_against"] = [str(e)]
            return result

        status = data_health_check(raw, tf)
        raw = only_closed_candles(raw)
        if len(raw) < 60:
            result = _empty_result(symbol, now, cfg)
            result["data_status"] = "DATA INVALID"
            result["reasons_against"] = [f"Insufficient closed-candle history on {tf}"]
            return result

        frames[label] = compute_all_indicators(raw, cfg)
        if status != "DATA OK":
            data_status = status

    result = generate_signal_from_frames(
        symbol, cfg, frames["htf"], frames["mtf"], frames["ltf"], frames["entry"], as_of=now
    )
    if result["data_status"] == "DATA OK":
        result["data_status"] = data_status
    return result
