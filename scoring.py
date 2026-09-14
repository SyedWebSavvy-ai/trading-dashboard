"""
Weighted scoring engine. Combines evidence from trend, structure, momentum,
volume, S/R proximity, volatility, and multi-timeframe confirmation into a
single 0-100 score. Weights are configurable (config.py), not assumed
optimal — they should be validated via backtesting/walk-forward testing.

Direction-agnostic sub-scores are computed for BOTH the bull case and the
bear case; the stronger, more consistent side wins, and inconsistent /
conflicting evidence pulls the score toward NO TRADE rather than forcing
a direction.
"""
import pandas as pd


def _clip(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def score_trend(htf_regime: dict, mtf_regime: dict) -> dict:
    """
    Both args are regime dicts from regime.classify_regime() — this is where
    'ema_bullish_stack' / 'ema_bearish_stack' actually live. (Previously this
    read those keys off a raw indicator dataframe row, where they never
    existed, so the confirmation-timeframe EMA-stack check was silently a
    no-op.)
    """
    bull, bear = 0, 0
    reasons_bull, reasons_bear = [], []

    if htf_regime["regime"] in ("STRONG_BULL_TREND", "WEAK_BULL_TREND"):
        bull += 70 if htf_regime["regime"] == "STRONG_BULL_TREND" else 45
        reasons_bull.append(f"Higher-timeframe regime is {htf_regime['regime'].replace('_', ' ').title()}")
    if htf_regime["regime"] in ("STRONG_BEAR_TREND", "WEAK_BEAR_TREND"):
        bear += 70 if htf_regime["regime"] == "STRONG_BEAR_TREND" else 45
        reasons_bear.append(f"Higher-timeframe regime is {htf_regime['regime'].replace('_', ' ').title()}")

    if mtf_regime.get("ema_bullish_stack"):
        bull += 30
        reasons_bull.append("EMA stack aligned bullish on confirmation timeframe")
    if mtf_regime.get("ema_bearish_stack"):
        bear += 30
        reasons_bear.append("EMA stack aligned bearish on confirmation timeframe")

    return {"bull": _clip(bull), "bear": _clip(bear), "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_structure(structure_info: dict) -> dict:
    bull, bear = 0, 0
    reasons_bull, reasons_bear = [], []

    if structure_info["structure"] == "BULLISH_STRUCTURE":
        bull += 60
        reasons_bull.append("Market structure shows higher highs and higher lows")
    if structure_info["structure"] == "BEARISH_STRUCTURE":
        bear += 60
        reasons_bear.append("Market structure shows lower highs and lower lows")

    if structure_info["bos"] == "BULLISH_BOS":
        bull += 40
        reasons_bull.append("Bullish break of structure confirmed")
    if structure_info["bos"] == "BEARISH_BOS":
        bear += 40
        reasons_bear.append("Bearish break of structure confirmed")

    return {"bull": _clip(bull), "bear": _clip(bear), "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_momentum(row) -> dict:
    bull, bear = 0, 0
    reasons_bull, reasons_bear = [], []

    rsi = row.get("rsi")
    if pd.notna(rsi):
        if 50 < rsi < 70:
            bull += 50
            reasons_bull.append(f"RSI ({rsi:.0f}) confirms upward momentum without being overbought")
        elif rsi >= 70:
            bull += 20
            reasons_bear.append(f"RSI ({rsi:.0f}) is overbought — momentum may be exhausted")
        if 30 < rsi < 50:
            bear += 50
            reasons_bear.append(f"RSI ({rsi:.0f}) confirms downward momentum without being oversold")
        elif rsi <= 30:
            bear += 20
            reasons_bull.append(f"RSI ({rsi:.0f}) is oversold — downside momentum may be exhausted")

    macd_hist = row.get("macd_hist")
    if pd.notna(macd_hist):
        if macd_hist > 0:
            bull += 40
            reasons_bull.append("MACD histogram positive")
        else:
            bear += 40
            reasons_bear.append("MACD histogram negative")

    return {"bull": _clip(bull), "bear": _clip(bear), "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_volume(row) -> dict:
    bull, bear = 0, 0
    reasons_bull, reasons_bear = [], []
    rel_vol = row.get("relative_volume")

    if pd.notna(rel_vol):
        if rel_vol >= 1.5:
            bull += 60
            bear += 60
            reasons_bull.append(f"Relative volume elevated ({rel_vol:.1f}x average)")
            reasons_bear.append(f"Relative volume elevated ({rel_vol:.1f}x average)")
        elif rel_vol < 0.7:
            reasons_bull.append("Volume confirmation is weak/below average")
            reasons_bear.append("Volume confirmation is weak/below average")

    obv_trend = row.get("obv_trend")
    if obv_trend == "up":
        bull += 40
        reasons_bull.append("On-balance volume trending up")
    elif obv_trend == "down":
        bear += 40
        reasons_bear.append("On-balance volume trending down")

    return {"bull": _clip(bull), "bear": _clip(bear), "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_support_resistance(price, zones, direction_hint) -> dict:
    bull, bear = 0, 0
    reasons_bull, reasons_bear = [], []
    if not zones:
        return {"bull": 0, "bear": 0, "reasons_bull": [], "reasons_bear": []}

    nearest = min(zones, key=lambda z: abs(price - z["mid"]))
    dist_pct = abs(price - nearest["mid"]) / price * 100
    confluence_bonus = min(nearest["touches"] * 10, 40)

    if dist_pct < 0.5:
        if price >= nearest["mid"]:
            bull += 20 + confluence_bonus
            reasons_bull.append(f"Price holding above a {nearest['touches']}-touch support zone")
            bear += 10
            reasons_bear.append("Price is close to a resistance/support zone — reduced room to run")
        else:
            bear += 20 + confluence_bonus
            reasons_bear.append(f"Price rejected at a {nearest['touches']}-touch resistance zone")
            bull += 10
            reasons_bull.append("Price is close to a resistance/support zone — reduced room to run")

    return {"bull": _clip(bull), "bear": _clip(bear), "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_volatility(regime_info: dict) -> dict:
    bull, bear = 50, 50  # neutral baseline; volatility itself isn't directional
    reasons_bull, reasons_bear = [], []
    flag = regime_info.get("volatility_flag")
    if flag == "HIGH_VOLATILITY":
        bull, bear = 20, 20
        reasons_bull.append("Volatility is abnormally high — wider stops / lower conviction")
        reasons_bear.append("Volatility is abnormally high — wider stops / lower conviction")
    elif flag == "LOW_VOLATILITY":
        reasons_bull.append("Volatility is low — breakouts may lack follow-through")
        reasons_bear.append("Volatility is low — breakouts may lack follow-through")
    return {"bull": bull, "bear": bear, "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score_mtf_confirmation(tf_biases: dict) -> dict:
    """
    tf_biases: dict like {'4h': 'bull', '1h': 'bull', '15m': 'bull', '5m': 'neutral'}
    Rewards agreement across timeframes; penalizes conflict heavily.
    """
    bull_count = sum(1 for v in tf_biases.values() if v == "bull")
    bear_count = sum(1 for v in tf_biases.values() if v == "bear")
    total = len(tf_biases)

    bull = _clip((bull_count / total) * 100) if total else 0
    bear = _clip((bear_count / total) * 100) if total else 0

    reasons_bull = [f"{tf.upper()} bias bullish" for tf, v in tf_biases.items() if v == "bull"]
    reasons_bear = [f"{tf.upper()} bias bearish" for tf, v in tf_biases.items() if v == "bear"]

    conflict = bull_count > 0 and bear_count > 0
    return {
        "bull": bull, "bear": bear,
        "reasons_bull": reasons_bull, "reasons_bear": reasons_bear,
        "conflict": conflict,
    }


def combine_scores(components: dict, weights: dict) -> dict:
    """
    components: {category: {"bull": x, "bear": y, "reasons_bull": [...], "reasons_bear": [...]}}
    Returns final direction, score (0-100), and combined reasons.
    """
    bull_total, bear_total = 0.0, 0.0
    reasons_bull, reasons_bear = [], []

    for cat, weight in weights.items():
        comp = components.get(cat)
        if not comp:
            continue
        bull_total += comp["bull"] * weight
        bear_total += comp["bear"] * weight
        reasons_bull.extend(comp.get("reasons_bull", []))
        reasons_bear.extend(comp.get("reasons_bear", []))

    if bull_total >= bear_total:
        direction = "BUY"
        score = bull_total
        supporting = reasons_bull
        opposing = reasons_bear
    else:
        direction = "SELL"
        score = bear_total
        supporting = reasons_bear
        opposing = reasons_bull

    return {
        "direction": direction,
        "score": round(score, 1),
        "bull_score": round(bull_total, 1),
        "bear_score": round(bear_total, 1),
        "reasons_supporting": supporting,
        "reasons_against": opposing,
    }


def score_band_label(score: float, bands) -> str:
    for lo, hi, label in bands:
        if lo <= score < hi:
            return label
    return "NO TRADE"
