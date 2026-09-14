"""
Automated tests for the backtester's entry-lifecycle model
(find_entry_trigger + _simulate_after_entry), added specifically to verify
the entry-model fix requested after the 4th review pass.

Run with:  python -m pytest tests/test_backtest_entry_model.py -v
or, without pytest installed:  python tests/test_backtest_entry_model.py

These are deliberately unit-level (synthetic candles constructed by hand),
not full run_backtest() integration tests, so they run instantly with no
network access and pin down exact expected behavior for each rule described
in backtest.py's module docstring.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtest import find_entry_trigger, _simulate_after_entry


def _candle(open_, high, low, close, minutes_from_epoch):
    t0 = pd.Timestamp("2025-01-01")
    return {
        "open_time": t0 + pd.Timedelta(minutes=minutes_from_epoch),
        "open": open_, "high": high, "low": low, "close": close,
        "close_time": t0 + pd.Timedelta(minutes=minutes_from_epoch + 5),
    }


def _candles(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. BUY zone touched -> entry occurs
# ---------------------------------------------------------------------------
def test_buy_zone_touched_triggers_entry():
    zone_low, zone_high = 99.0, 101.0
    future = _candles([
        _candle(105, 106, 104, 105, 0),   # far away, no touch
        _candle(103, 104, 100.5, 101.5, 5),  # dips into zone (low <= zone_high)
    ])
    expiry = future["close_time"].iloc[-1] + pd.Timedelta(minutes=100)
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["status"] == "FILLED", "expected a fill when the zone is touched"
    assert result["trigger_iloc"] == 1
    # open (103) is above zone_high (101) and low (100.5) reaches into the zone
    # -> fill at zone_high per the documented boundary-crossed rule
    assert result["fill_price"] == 101.0


# ---------------------------------------------------------------------------
# 2. SELL zone touched -> entry occurs (touch/fill logic is direction-agnostic)
# ---------------------------------------------------------------------------
def test_sell_zone_touched_triggers_entry():
    zone_low, zone_high = 99.0, 101.0
    future = _candles([
        _candle(95, 96, 94, 95, 0),        # far away, no touch
        _candle(97, 99.5, 96, 98, 5),      # rises into zone from below
    ])
    expiry = future["close_time"].iloc[-1] + pd.Timedelta(minutes=100)
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["status"] == "FILLED"
    # open (97) is below zone_low (99) and high (99.5) reaches into the zone
    # -> fill at zone_low
    assert result["fill_price"] == 99.0


# ---------------------------------------------------------------------------
# 3. Zone never touched -> no trade, EXPIRED_NO_ENTRY
# ---------------------------------------------------------------------------
def test_zone_never_touched_expires():
    zone_low, zone_high = 99.0, 101.0
    future = _candles([
        _candle(110, 111, 109, 110, 0),
        _candle(112, 113, 111, 112, 5),
        _candle(115, 116, 114, 115, 10),
    ])
    expiry = future["close_time"].iloc[-1] + pd.Timedelta(minutes=100)
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["status"] == "EXPIRED_NO_ENTRY"
    assert result["fill_price"] is None


# ---------------------------------------------------------------------------
# 4. Price moves away before zone touch -> no fake entry
# ---------------------------------------------------------------------------
def test_price_runs_away_no_fake_entry():
    zone_low, zone_high = 99.0, 101.0
    # Price starts near the zone but immediately runs away without ever
    # actually overlapping [99, 101] in any candle's [low, high] range.
    future = _candles([
        _candle(102, 103, 101.5, 102.5, 0),   # low (101.5) stays above zone_high (101)
        _candle(104, 106, 103, 105, 5),
    ])
    expiry = future["close_time"].iloc[-1] + pd.Timedelta(minutes=100)
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["status"] == "EXPIRED_NO_ENTRY", "must not fabricate a touch that never happened"


# ---------------------------------------------------------------------------
# 5. Entry occurs, then SL -> correct loss
# ---------------------------------------------------------------------------
def test_entry_then_stop_loss():
    fill_price = 100.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 101.0, 102.0, 104.0
    candles_from_trigger = _candles([
        _candle(100, 100.5, 99.5, 100, 0),   # triggering candle itself, no SL/TP hit yet
        _candle(99.5, 99.8, 97.5, 98.0, 5),  # drops through stop loss
    ])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "STOPPED"
    assert exit_price == stop_loss
    assert tp1_hit is False and tp2_hit is False and tp3_hit is False


# ---------------------------------------------------------------------------
# 6. Entry occurs, then TP3 -> correct winner
# ---------------------------------------------------------------------------
def test_entry_then_tp3():
    fill_price = 100.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 101.0, 102.0, 104.0
    candles_from_trigger = _candles([
        _candle(100, 101.2, 99.8, 101, 0),    # TP1 hit
        _candle(101, 102.5, 100.9, 102, 5),   # TP2 hit
        _candle(102, 104.5, 101.9, 104, 10),  # TP3 hit -> full close
    ])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "TP3"
    assert exit_price == tp3
    assert tp1_hit is True and tp2_hit is True and tp3_hit is True


# ---------------------------------------------------------------------------
# 7. Entry zone touched only after signal expiry -> no entry
# ---------------------------------------------------------------------------
def test_zone_touched_after_expiry_is_not_an_entry():
    zone_low, zone_high = 99.0, 101.0
    future = _candles([
        _candle(110, 111, 109, 110, 0),    # before expiry, no touch
        _candle(105, 106, 104, 105, 5),    # before expiry, no touch
        _candle(103, 104, 100.5, 101.5, 10),  # touches the zone, but AFTER expiry
    ])
    # Expiry occurs before the third candle closes
    expiry = future["close_time"].iloc[1]
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["status"] == "EXPIRED_NO_ENTRY", "a touch after expiry must not count as an entry"


# ---------------------------------------------------------------------------
# 8. Costs affect actual execution correctly (direction-correct slippage/fees)
# ---------------------------------------------------------------------------
def test_costs_apply_to_actual_fill_and_exit():
    from backtest import _apply_costs

    # BUY: entry should be marked UP by slippage, exit marked DOWN by slippage
    raw_pnl, net_pnl, fill_entry, fill_exit = _apply_costs("BUY", 100.0, 105.0, slippage_pct=0.1, fee_pct=0.05)
    assert fill_entry > 100.0, "BUY entry fill should be worse (higher) due to slippage"
    assert fill_exit < 105.0, "BUY exit fill should be worse (lower) due to slippage"
    assert net_pnl < raw_pnl, "fees must reduce net pnl relative to raw pnl"

    # SELL: entry should be marked DOWN by slippage, exit marked UP by slippage
    raw_pnl_s, net_pnl_s, fill_entry_s, fill_exit_s = _apply_costs("SELL", 100.0, 95.0, slippage_pct=0.1, fee_pct=0.05)
    assert fill_entry_s < 100.0, "SELL entry fill should be worse (lower) due to slippage"
    assert fill_exit_s > 95.0, "SELL exit fill should be worse (higher) due to slippage"
    assert net_pnl_s < raw_pnl_s


# ---------------------------------------------------------------------------
# 9. Same-candle ambiguous entry/SL/TP -> conservative rule (SL first)
# ---------------------------------------------------------------------------
def test_same_candle_sl_and_tp_resolves_conservatively_to_sl():
    fill_price = 100.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 101.0, 102.0, 104.0
    # The very first (triggering) candle's range spans BOTH the stop-loss
    # and TP3 — true intra-candle order is unknowable, so SL must win.
    candles_from_trigger = _candles([
        _candle(100, 105.0, 97.0, 99.0, 0),
    ])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "STOPPED", "same-candle SL/TP ambiguity must resolve to SL (conservative)"
    assert exit_price == stop_loss


# ---------------------------------------------------------------------------
# 10. No future-candle information affects the signal / entry search
# ---------------------------------------------------------------------------
def test_entry_trigger_never_looks_before_its_own_candle_list():
    """
    Structural invariant check: find_entry_trigger only ever considers
    candles in the exact list it's given, in order, and never returns a
    trigger_iloc that would require information beyond what was passed in.
    Combined with run_backtest() only ever constructing `future_candles`
    from rows with open_time > decision_time (see backtest.py), this is the
    concrete guarantee against look-ahead in the entry search specifically.
    """
    zone_low, zone_high = 99.0, 101.0
    future = _candles([
        _candle(110, 111, 109, 110, 0),
        _candle(100, 100.5, 99.5, 100, 5),  # touch here, iloc=1
        _candle(50, 51, 49, 50, 10),        # irrelevant later candle
    ])
    expiry = future["close_time"].iloc[-1] + pd.Timedelta(minutes=100)
    result = find_entry_trigger(zone_low, zone_high, future, expiry)
    assert result["trigger_iloc"] == 1, "must trigger on the first touching candle, not scan ahead for a 'better' one"


# ---------------------------------------------------------------------------
# 11. TP reached within the TRIGGER candle must NOT be counted as a win
#     (order-ambiguous: could have happened before the zone was even touched)
# ---------------------------------------------------------------------------
def test_tp_in_trigger_candle_is_not_counted_as_immediate_win():
    fill_price = 101.0   # e.g. filled at zone_high, price approached from above
    stop_loss = 98.0
    tp1, tp2, tp3 = 102.0, 104.0, 108.0
    # The triggering candle itself spikes way past TP3 (high=110) — this
    # must NOT be credited as an immediate TP3 win on count==0.
    trigger_candle = _candle(103, 110, 100, 105, 0)  # open above zone, low touches zone, high blows past TP3
    next_candle = _candle(101, 101.5, 100.5, 101.2, 5)  # genuinely unremarkable: stays below TP1 (102)
    candles_from_trigger = _candles([trigger_candle, next_candle])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason != "TP3", "a TP hit within the trigger candle itself must not be credited as an immediate win"
    assert tp3_hit is False
    assert tp1_hit is False and tp2_hit is False, "TP milestones must not be marked from the trigger candle either"


# ---------------------------------------------------------------------------
# 12. TP reached before entry (same idea, phrased as the general case) ->
#     must not be a winner even when the trigger candle ALSO has no SL hit
# ---------------------------------------------------------------------------
def test_tp_before_entry_general_case_not_a_winner():
    fill_price = 100.0
    stop_loss = 95.0
    tp1, tp2, tp3 = 101.0, 102.0, 103.0
    # Trigger candle's range spans well past TP3 with no SL touch at all —
    # under the OLD (buggy) logic this would have been an instant TP3 win.
    trigger_candle = _candle(100, 106, 99.5, 100.5, 0)
    candles_from_trigger = _candles([trigger_candle])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "DATA_END", "with only the trigger candle available and TP-checks skipped on it, the trade should still be open at data end, not a declared TP3 win"
    assert tp3_hit is False


# ---------------------------------------------------------------------------
# 13. Zone touch -> then a LATER candle reaches TP -> valid win
# ---------------------------------------------------------------------------
def test_zone_touch_then_later_candle_tp_is_a_valid_win():
    fill_price = 100.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 101.0, 102.0, 103.0
    trigger_candle = _candle(100, 100.2, 99.8, 100.0, 0)   # entry candle, nothing else happens
    winning_candle = _candle(100, 103.5, 99.9, 103.0, 5)   # TP3 reached on a LATER candle
    candles_from_trigger = _candles([trigger_candle, winning_candle])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "TP3", "a TP reached on a candle AFTER the trigger candle is a legitimate win"
    assert exit_price == tp3


# ---------------------------------------------------------------------------
# 14. Zone touch -> then a LATER candle hits SL -> valid loss
# ---------------------------------------------------------------------------
def test_zone_touch_then_later_candle_sl_is_a_valid_loss():
    fill_price = 100.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 101.0, 102.0, 103.0
    trigger_candle = _candle(100, 100.2, 99.8, 100.0, 0)
    losing_candle = _candle(99.5, 99.8, 97.5, 98.0, 5)
    candles_from_trigger = _candles([trigger_candle, losing_candle])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "STOPPED"
    assert exit_price == stop_loss


# ---------------------------------------------------------------------------
# 15. SL within the trigger candle IS still honored (conservative both ways)
# ---------------------------------------------------------------------------
def test_sl_in_trigger_candle_is_still_honored():
    fill_price = 101.0
    stop_loss = 98.0
    tp1, tp2, tp3 = 102.0, 104.0, 108.0
    # Trigger candle's range also reaches the stop-loss -> must exit STOPPED
    # right there, regardless of true intra-candle order (safe assumption).
    trigger_candle = _candle(103, 105, 97.5, 99, 0)
    candles_from_trigger = _candles([trigger_candle])
    exit_reason, exit_price, exit_time, tp1_hit, tp2_hit, tp3_hit, mae, mfe = _simulate_after_entry(
        "BUY", fill_price, stop_loss, tp1, tp2, tp3, candles_from_trigger, max_hold_candles=50
    )
    assert exit_reason == "STOPPED"
    assert exit_price == stop_loss


# ---------------------------------------------------------------------------
# 16. Actual fill differing from signal price -> rr_realized reported
#     correctly and distinctly from rr_planned
# ---------------------------------------------------------------------------
def test_realized_rr_differs_from_planned_when_fill_differs_from_signal():
    from risk import risk_reward_ratio

    signal_price = 100.0
    stop_loss = 98.0
    tp3 = 106.0
    rr_planned = risk_reward_ratio(signal_price, stop_loss, tp3)  # (106-100)/(100-98) = 3.0

    fill_price = 100.15  # actual fill drifted to the top of a tight entry zone
    rr_realized = risk_reward_ratio(fill_price, stop_loss, tp3)   # (106-100.15)/(100.15-98) = ~2.72

    assert rr_planned == 3.0
    assert rr_realized != rr_planned, "realized RR must reflect the actual fill, not silently equal the planned RR"
    assert abs(rr_realized - 2.72) < 0.01


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
