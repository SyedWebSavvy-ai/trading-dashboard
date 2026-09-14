"""
Event-driven backtester.

This calls the EXACT SAME decision function the live app uses
(signals.generate_signal_from_frames) and asks "if this signal fired at
candle T, what actually happened next?" — but critically, it does NOT
assume the trade fills at the signal candle's close. It simulates the full
causal entry lifecycle the live dashboard actually implies:

    SIGNAL GENERATED -> WAITING FOR ENTRY -> ENTRY TRIGGERED -> OPEN -> EXIT

Earlier versions of this backtester used `entry_price = sig["current_price"]`
directly — i.e. assumed an instant fill at the decision candle's close. But
the live app displays an `entry_zone` (current_price +/- 0.15*ATR) that the
user is meant to wait for. Those are two different strategies. This version
fixes that mismatch: a signal only becomes a trade once price actually
touches the advertised entry zone on a subsequent candle, and if it never
does before the signal expires, that's recorded as EXPIRED_NO_ENTRY, not a
trade.

============================== EXECUTION MODEL ==============================

1. SIGNAL GENERATED at decision_time (a closed entry-timeframe candle),
   using generate_signal_from_frames() with data sliced as-of that moment.
   The signal stores: direction, entry_zone (low/high), stop_loss, tp1/2/3,
   score, regime, and an expiry time (decision_time + signal_expiry_minutes).
   No trade exists yet.

2. WAITING FOR ENTRY: subsequent candles (open_time > decision_time) are
   scanned in order for a zone touch. The signal's OWN generation candle is
   NEVER used for fill — even though price is trivially inside the zone at
   that instant (the zone is centered on that candle's close) — because
   execution cannot happen before the signal was actually available. A
   fresh touch on a later candle is required. This is a deliberate design
   choice for the "price already in the zone" edge case.

3. ENTRY TRIGGER / FILL RULE: a candle "touches" the zone if its
   [low, high] range overlaps [zone_low, zone_high]. Only candles that
   close at or before the signal's expiry are eligible — a touch on a
   candle closing after expiry does not count. Fill price:
     - open already inside the zone            -> fill = open
     - open below the zone, high reaches it     -> fill = zone_low
     - open above the zone, low reaches it      -> fill = zone_high
   This is a deterministic "filled at the boundary first crossed" rule. It
   deliberately does NOT assume a more optimistic fill deeper inside the
   zone, and does NOT assume a mid-zone fill on a gap-through candle.

4. If no candle touches the zone before expiry -> EXPIRED_NO_ENTRY. No
   trade is created; this is tracked separately from completed trades so
   headline stats aren't diluted or inflated by setups that never filled.

5. SAME-CANDLE / ORDERING AMBIGUITY has two distinct forms, handled
   differently and deliberately:
     a) Within the TRIGGER candle itself (entry happens intrabar on this
        candle): only the stop-loss is checked against this candle's range.
        Take-profit hits within this same candle are NOT evaluated and
        cannot produce a win — we cannot tell from OHLC data whether a TP
        was reached before or after price actually crossed into the entry
        zone, and crediting it would risk counting a win that happened
        before the trade even existed. TP evaluation begins from the next
        candle onward, where entry has unambiguously already occurred.
     b) On any LATER candle (after entry is already established): if that
        candle's range reaches both the stop-loss and a take-profit, the
        stop-loss is assumed to hit first (conservative/worst-case) — this
        ambiguity is only about intra-candle order between two prices,
        which is a different and simpler question than (a).

6. AFTER ENTRY, the signal's expiry no longer applies — the position is
   open and is only closed by SL, TP3, or the max-hold-candles limit
   (renamed MAX_HOLD_EXIT, distinct from EXPIRED_NO_ENTRY which is a
   pre-entry-only outcome).

7. TP1/TP2 remain MILESTONES ONLY in this version — they are recorded for
   hit-rate reporting but do NOT trigger a partial close. The full
   simulated position remains open until SL, TP3, or max-hold. Do not read
   "TP1 hit rate" as "money banked at TP1."

8. Fees and slippage are applied to the ACTUAL simulated fill price and the
   actual simulated exit price (not the original signal price) — BUY fills
   worse (higher) and BUY exits worse (lower) by slippage; SELL is
   mirrored. Fees are charged once per side (entry + exit), not doubled.

Anti-look-ahead design (unchanged from the previous version, re-verified
here): every timeframe is sliced to only candles closed by decision_time
(data_fetch.slice_as_of) before generate_signal_from_frames() ever sees it.
Structure/swing detection runs on that already-sliced data, so it cannot see
future candles either. See tests/test_backtest_entry_model.py for automated
checks of the entry-trigger logic specifically.

Known limitations still present after this fix (unchanged, not addressed in
this pass on purpose — see the review that prompted this fix):
  - No partial position scale-out at TP1/TP2 (see point 7 above).
  - No portfolio-level correlation/max-eligible-signals simulation across
    symbols — each symbol is backtested independently. This is the next
    planned phase, not this one.
  - One pending signal (waiting-for-entry OR open) per symbol at a time: a
    new signal is not evaluated for a symbol while a previous one is either
    waiting to fill or already open.
  - RR/position sizing basis: stop_loss/tp1/tp2/tp3 (and the live RR gate
    in signals.py) are all computed from `signal_price` (the price at
    signal generation), not from the eventual fill price — this matches
    what a live user would actually see and act on, but means the R
    multiple realized in backtest can differ slightly from `rr_planned`.
    Both `rr_planned` (signal-price basis, matches the live gate) and
    `rr_realized` (actual-fill-price basis) are now recorded on every
    Trade so this is visible rather than silently absorbed. The gap is
    bounded by the entry-zone width (+/- 0.15*ATR), so it's small, but not
    zero, and position sizing is likewise still based on signal price, not
    recalculated after the fill.
"""
from dataclasses import dataclass, asdict
from datetime import timedelta
import pandas as pd
import numpy as np

from data_fetch import get_historical_klines, slice_as_of, only_closed_candles
from indicators import compute_all_indicators
from signals import generate_signal_from_frames
from risk import risk_reward_ratio


TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


@dataclass
class Trade:
    symbol: str
    direction: str
    signal_time: object          # when the signal was generated (decision candle close)
    signal_price: float          # current_price at signal generation (zone center)
    entry_zone_low: float
    entry_zone_high: float
    entry_trigger_time: object   # candle close_time on which the zone was touched
    entry_delay_minutes: float   # signal_time -> entry_trigger_time
    entry_price: float           # ACTUAL simulated fill price (boundary-crossed rule)
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    score: float
    directional_edge: float
    regime: str
    rr_planned: float             # RR computed at signal time, from signal_price (matches the live gate)
    rr_realized: float            # RR recomputed from the ACTUAL fill price vs the same TP3/SL
    fill_slippage: float          # entry_price (fill) - signal_price, signed; how far the real fill drifted from what the live RR gate saw
    exit_time: object = None
    exit_price: float = None
    exit_reason: str = None      # STOPPED | TP3 | MAX_HOLD_EXIT | DATA_END
    r_multiple: float = None
    r_multiple_net: float = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    mae_r: float = None
    mfe_r: float = None


@dataclass
class UnfilledSignal:
    symbol: str
    direction: str
    signal_time: object
    signal_price: float
    entry_zone_low: float
    entry_zone_high: float
    expiry_time: object
    score: float
    regime: str
    reason: str = "EXPIRED_NO_ENTRY"


def fetch_backtest_data(symbol: str, cfg, start_dt, end_dt, buffer_candles: int = 250, progress_cb=None):
    """
    Fetches and indicator-computes all 4 timeframes for [start_dt, end_dt],
    with extra history before start_dt so slow indicators (EMA200 etc.) are
    warmed up from the very first evaluated bar rather than starting as NaN.
    """
    tf_map = {"htf": cfg.htf, "mtf": cfg.mtf_confirm, "ltf": cfg.ltf_confirm, "entry": cfg.entry_tf}
    frames = {}
    for label, tf in tf_map.items():
        buffer_start = pd.Timestamp(start_dt) - timedelta(minutes=TF_MINUTES[tf] * buffer_candles)
        raw = get_historical_klines(symbol, tf, buffer_start, end_dt, progress_cb=progress_cb)
        raw = only_closed_candles(raw)
        if raw.empty:
            frames[label] = raw
            continue
        frames[label] = compute_all_indicators(raw, cfg)
    return frames


def find_entry_trigger(zone_low: float, zone_high: float, future_candles: pd.DataFrame, expiry_time):
    """
    Scans candles STRICTLY AFTER the signal's generation candle (the caller
    guarantees this — the generation candle itself must never be passed in
    here) for the first one whose [low, high] range overlaps the entry
    zone, and returns the deterministic fill described in the module
    docstring. Only candles closing at or before `expiry_time` are eligible.

    Returns a dict: {"status": "FILLED"|"EXPIRED_NO_ENTRY", "fill_price",
    "trigger_time" (candle close_time), "trigger_iloc" (positional index
    into future_candles, so the caller can resume simulation from there)}.
    """
    for pos in range(len(future_candles)):
        row = future_candles.iloc[pos]
        if row["close_time"] > expiry_time:
            break  # this and all subsequent candles close after expiry

        o, h, l = row["open"], row["high"], row["low"]
        touched = (l <= zone_high) and (h >= zone_low)
        if not touched:
            continue

        if zone_low <= o <= zone_high:
            fill = o
        elif o < zone_low:
            fill = zone_low
        else:
            fill = zone_high

        return {
            "status": "FILLED", "fill_price": fill,
            "trigger_time": row["close_time"], "trigger_iloc": pos,
        }

    return {"status": "EXPIRED_NO_ENTRY", "fill_price": None, "trigger_time": None, "trigger_iloc": None}


def _simulate_after_entry(direction, entry_price, stop_loss, tp1, tp2, tp3,
                           candles_from_trigger: pd.DataFrame, max_hold_candles: int):
    """
    Walks forward starting AT the triggering candle itself (inclusive).

    CRITICAL ORDERING RULE for the trigger candle specifically (count == 0):
    entry itself happened intrabar on this candle (price crossed into the
    zone somewhere within its [low, high] range). If this SAME candle's
    range also reaches a take-profit, we have NO way to know from OHLC data
    whether that TP was reached before or after the zone was actually
    touched — e.g. a candle that opens above the zone, spikes up through
    TP3, then falls back down through the zone could be mistaken for "enter
    then immediately hit TP3," when the true order may have been "hit TP3
    first, then the zone was touched on the way back down" — which is not
    a valid entry+win at all.

    So, on the trigger candle ONLY:
      - Stop-loss IS still checked and honored. This is conservative in
        both possible orderings: if SL happened before the zone was even
        touched, there was never a valid trade to begin with, and if it
        happened after entry, it's a real loss — either way, "assume the
        loss" is the safe assumption.
      - Take-profit hits (TP1/TP2/TP3) are NOT evaluated on the trigger
        candle. No milestone is marked and no win is declared from that
        candle's range. Evaluation of TP levels begins from the NEXT
        candle onward, where entry has unambiguously already happened.

    From the second candle onward, the existing "SL wins same-candle ties"
    convention applies as before — that ambiguity is only about which of
    SL/TP happened first within a single bar, which is a different and
    already-correctly-handled question from "had we even entered yet."

    Signal expiry does NOT apply here — the position is already open.
    """
    mae, mfe = 0.0, 0.0
    tp1_hit = tp2_hit = tp3_hit = False

    for count in range(len(candles_from_trigger)):
        row = candles_from_trigger.iloc[count]
        if count >= max_hold_candles:
            return "MAX_HOLD_EXIT", row["close"], row["close_time"], tp1_hit, tp2_hit, tp3_hit, mae, mfe

        high, low = row["high"], row["low"]
        if direction == "BUY":
            adverse = entry_price - low
            favorable = high - entry_price
        else:
            adverse = high - entry_price
            favorable = entry_price - low
        mae = max(mae, adverse)
        mfe = max(mfe, favorable)

        hit_sl = (low <= stop_loss) if direction == "BUY" else (high >= stop_loss)

        if hit_sl:
            return "STOPPED", stop_loss, row["close_time"], tp1_hit, tp2_hit, tp3_hit, mae, mfe

        if count == 0:
            # Trigger candle: SL checked above, but TP evaluation is
            # deliberately skipped here — see docstring. Move to the next
            # candle without marking any TP milestone or exit.
            continue

        hit_tp1 = (high >= tp1) if direction == "BUY" else (low <= tp1)
        hit_tp2 = (high >= tp2) if direction == "BUY" else (low <= tp2)
        hit_tp3 = (high >= tp3) if direction == "BUY" else (low <= tp3)

        if hit_tp1:
            tp1_hit = True
        if hit_tp2:
            tp2_hit = True
        if hit_tp3:
            return "TP3", tp3, row["close_time"], tp1_hit, tp2_hit, True, mae, mfe

    if len(candles_from_trigger):
        last = candles_from_trigger.iloc[-1]
        return "DATA_END", last["close"], last["close_time"], tp1_hit, tp2_hit, tp3_hit, mae, mfe
    return "DATA_END", entry_price, None, tp1_hit, tp2_hit, tp3_hit, mae, mfe


def _apply_costs(direction, raw_entry_price, raw_exit_price, slippage_pct, fee_pct):
    """
    Applies slippage adversely on both fills, then fees on both notional
    legs once each (never doubled). Returns (net_pnl_per_unit, fill_entry,
    fill_exit) so callers can also report gross figures if needed.
    """
    slip = slippage_pct / 100
    fee = fee_pct / 100
    if direction == "BUY":
        fill_entry = raw_entry_price * (1 + slip)   # worse (higher) fill for a buyer
        fill_exit = raw_exit_price * (1 - slip)     # worse (lower) fill on exit
        raw_pnl = fill_exit - fill_entry
    else:
        fill_entry = raw_entry_price * (1 - slip)   # worse (lower) fill for a seller/short
        fill_exit = raw_exit_price * (1 + slip)     # worse (higher) fill on exit
        raw_pnl = fill_entry - fill_exit

    fee_cost = (fill_entry + fill_exit) * fee
    net_pnl = raw_pnl - fee_cost
    return raw_pnl, net_pnl, fill_entry, fill_exit


def run_backtest(symbol: str, cfg, start_dt, end_dt, progress_cb=None):
    """
    Runs the event-driven backtest for one symbol over [start_dt, end_dt].
    Returns (trades, unfilled_signals) — trades is a list of Trade objects
    (signals that actually got filled and resolved), unfilled_signals is a
    list of UnfilledSignal objects (setups that expired before price ever
    touched the entry zone). Headline trade statistics should be computed
    from `trades` only; `unfilled_signals` is for funnel/diagnostic
    reporting (see compute_metrics).
    """
    frames = fetch_backtest_data(symbol, cfg, start_dt, end_dt, progress_cb=progress_cb)
    entry_df_full = frames.get("entry")
    if entry_df_full is None or entry_df_full.empty:
        return [], []

    eval_df = entry_df_full[
        (entry_df_full["close_time"] >= pd.Timestamp(start_dt)) &
        (entry_df_full["close_time"] <= pd.Timestamp(end_dt))
    ].reset_index(drop=True)

    trades = []
    unfilled_signals = []
    busy_until = None  # timestamp until which no new signal is evaluated for this symbol
    min_history = 60

    for i in range(len(eval_df)):
        decision_time = eval_df["close_time"].iloc[i]

        if busy_until is not None and decision_time < busy_until:
            continue  # a prior signal is still waiting-for-entry or open

        htf_slice = slice_as_of(frames["htf"], decision_time)
        mtf_slice = slice_as_of(frames["mtf"], decision_time)
        ltf_slice = slice_as_of(frames["ltf"], decision_time)
        entry_slice = slice_as_of(entry_df_full, decision_time)

        if min(len(htf_slice) if htf_slice is not None else 0,
               len(mtf_slice) if mtf_slice is not None else 0,
               len(ltf_slice) if ltf_slice is not None else 0,
               len(entry_slice) if entry_slice is not None else 0) < min_history:
            continue

        sig = generate_signal_from_frames(
            symbol, cfg, htf_slice, mtf_slice, ltf_slice, entry_slice, as_of=decision_time
        )

        if sig["direction"] not in ("BUY", "SELL"):
            continue

        signal_price = sig["current_price"]
        zone_low, zone_high = sig["entry_zone"]
        stop_loss = sig["stop_loss"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        expiry_time = sig["expiry"]  # decision_time + signal_expiry_minutes, set inside signals.py

        # Candles strictly AFTER the generation candle — the generation
        # candle itself is never eligible for fill (see docstring point 2).
        future_candles = entry_df_full[entry_df_full["open_time"] > decision_time].reset_index(drop=True)

        trigger = find_entry_trigger(zone_low, zone_high, future_candles, expiry_time)

        if trigger["status"] == "EXPIRED_NO_ENTRY":
            unfilled_signals.append(UnfilledSignal(
                symbol=symbol, direction=sig["direction"], signal_time=decision_time,
                signal_price=signal_price, entry_zone_low=zone_low, entry_zone_high=zone_high,
                expiry_time=expiry_time, score=sig["score"], regime=sig["regime"],
            ))
            busy_until = expiry_time  # don't re-evaluate this symbol until the expired setup's window passes
            continue

        fill_price = trigger["fill_price"]
        trigger_time = trigger["trigger_time"]
        trigger_iloc = trigger["trigger_iloc"]
        entry_delay_minutes = (trigger_time - decision_time).total_seconds() / 60

        # ---- RR/risk accounting note ----
        # stop_loss/tp1/tp2/tp3 were all computed at signal time relative to
        # `signal_price` (current_price at generation) — that's also what
        # the LIVE risk/reward gate in signals.py checks against. The actual
        # fill can differ slightly (bounded by the entry-zone width, since
        # fill is only ever taken within it), so the R multiple actually
        # realized is computed against `fill_price`, while `rr_planned`
        # (already on `sig`) reflects what the live gate saw. Both are kept
        # on the Trade record rather than only reporting one, so this
        # inconsistency is visible instead of silently absorbed.
        rr_realized = risk_reward_ratio(fill_price, stop_loss, tp3)
        fill_slippage = round(fill_price - signal_price, 6)

        # Simulate SL/TP/max-hold starting AT the triggering candle (inclusive).
        candles_from_trigger = future_candles.iloc[trigger_iloc:].reset_index(drop=True)
        (exit_reason, raw_exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit,
         mae, mfe) = _simulate_after_entry(
            sig["direction"], fill_price, stop_loss, tp1, tp2, tp3,
            candles_from_trigger, cfg.max_hold_candles,
        )

        risk_distance = abs(fill_price - stop_loss)
        raw_pnl, net_pnl, fill_entry, fill_exit = _apply_costs(
            sig["direction"], fill_price, raw_exit_price, cfg.slippage_pct, cfg.taker_fee_pct
        )

        r_multiple = round(raw_pnl / risk_distance, 3) if risk_distance else 0.0
        r_multiple_net = round(net_pnl / risk_distance, 3) if risk_distance else 0.0
        mae_r = round(mae / risk_distance, 3) if risk_distance else 0.0
        mfe_r = round(mfe / risk_distance, 3) if risk_distance else 0.0

        trade = Trade(
            symbol=symbol, direction=sig["direction"],
            signal_time=decision_time, signal_price=signal_price,
            entry_zone_low=zone_low, entry_zone_high=zone_high,
            entry_trigger_time=trigger_time, entry_delay_minutes=round(entry_delay_minutes, 1),
            entry_price=fill_price, stop_loss=stop_loss, tp1=tp1, tp2=tp2, tp3=tp3,
            score=sig["score"], directional_edge=sig.get("directional_edge", 0),
            regime=sig["regime"], rr_planned=sig["rr"], rr_realized=rr_realized,
            fill_slippage=fill_slippage,
            exit_time=exit_time, exit_price=raw_exit_price, exit_reason=exit_reason,
            r_multiple=r_multiple, r_multiple_net=r_multiple_net,
            tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
            mae_r=mae_r, mfe_r=mfe_r,
        )
        trades.append(trade)

        # One pending/open signal per symbol at a time: nothing new is
        # evaluated until this trade's exit_time.
        busy_until = exit_time if exit_time is not None else decision_time

    return trades, unfilled_signals


def compute_metrics(trades: list, unfilled_signals: list = None) -> dict:
    """
    Aggregates results into the honesty-forcing metrics that tell you
    whether the scoring model has any demonstrated edge — using net
    (post-fee, post-slippage) R multiples throughout, computed ONLY from
    actual filled trades. Unfilled signals are reported as funnel/diagnostic
    counts, not folded into win rate or expectancy.
    """
    unfilled_signals = unfilled_signals or []
    n_unfilled = len(unfilled_signals)
    n_signals_total = len(trades) + n_unfilled

    if not trades:
        return {
            "total_trades": 0,
            "signals_generated": n_signals_total,
            "entries_triggered": 0,
            "expired_no_entry": n_unfilled,
            "fill_rate_pct": 0.0 if n_signals_total == 0 else round(0 / n_signals_total * 100, 1),
        }

    df = pd.DataFrame([asdict(t) for t in trades])
    r = df["r_multiple_net"]

    wins = r[r > 0]
    losses = r[r <= 0]
    win_rate = len(wins) / len(r) * 100 if len(r) else 0
    avg_r = r.mean()
    expectancy = avg_r  # expectancy per trade, in R, net of costs
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    equity_curve = r.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown_r = drawdown.min() if len(drawdown) else 0.0

    std_r = r.std()
    sharpe_like = (avg_r / std_r) if std_r and std_r > 0 else 0.0

    metrics = {
        "signals_generated": n_signals_total,
        "entries_triggered": len(trades),
        "expired_no_entry": n_unfilled,
        "fill_rate_pct": round(len(trades) / n_signals_total * 100, 1) if n_signals_total else 0.0,
        "avg_entry_delay_minutes": round(df["entry_delay_minutes"].mean(), 1),

        "total_trades": len(df),
        "win_rate_pct": round(win_rate, 1),
        "avg_r_net": round(avg_r, 3),
        "expectancy_r": round(expectancy, 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losers)",
        "max_drawdown_r": round(max_drawdown_r, 2),
        "sharpe_like": round(sharpe_like, 2),
        "avg_winner_r": round(wins.mean(), 3) if len(wins) else 0.0,
        "avg_loser_r": round(losses.mean(), 3) if len(losses) else 0.0,
        "tp1_hit_rate_pct": round(df["tp1_hit"].mean() * 100, 1),
        "tp2_hit_rate_pct": round(df["tp2_hit"].mean() * 100, 1),
        "tp3_hit_rate_pct": round(df["tp3_hit"].mean() * 100, 1),
        "stopped_rate_pct": round((df["exit_reason"] == "STOPPED").mean() * 100, 1),
        "max_hold_exit_rate_pct": round((df["exit_reason"].isin(["MAX_HOLD_EXIT", "DATA_END"])).mean() * 100, 1),
        "equity_curve_r": equity_curve.tolist(),
    }

    by_regime = df.groupby("regime")["r_multiple_net"].agg(["count", "mean"]).round(3)
    metrics["by_regime"] = by_regime.to_dict(orient="index")

    by_symbol = df.groupby("symbol")["r_multiple_net"].agg(["count", "mean"]).round(3)
    metrics["by_symbol"] = by_symbol.to_dict(orient="index")

    bins = [0, 65, 70, 75, 80, 85, 100]
    labels = ["<65", "65-70", "70-75", "75-80", "80-85", "85+"]
    df["score_bucket"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)
    by_score = df.groupby("score_bucket", observed=True)["r_multiple_net"].agg(["count", "mean"]).round(3)
    metrics["by_score_bucket"] = by_score.to_dict(orient="index")

    metrics["trades_df"] = df
    return metrics
