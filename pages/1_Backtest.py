"""
Backtest page. Runs the EXACT SAME decision logic as the live dashboard
(signals.generate_signal_from_frames) against real historical Binance data,
so you can see whether this scoring model has ever actually made money —
instead of just trusting that a high score looks convincing.
"""
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st

from config import Config
import backtest

st.set_page_config(page_title="Backtest — Signal Research", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0f14; }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .metric-box {
        background: #131a24; border: 1px solid #1f2937; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .disclaimer {
        background: #1a1207; border: 1px solid #4a3510; color: #fbbf24;
        border-radius: 10px; padding: 12px 16px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Backtest — Does This Thing Actually Work?")
st.markdown(
    '<div class="disclaimer">This runs the identical decision logic used by the live dashboard '
    '(same file, same function — <code>signals.generate_signal_from_frames</code>) against real '
    'historical candles. A signal only becomes a trade once price actually touches the advertised '
    'entry zone on a <em>later</em> candle (never assumed to fill instantly at the signal candle\'s '
    'close) — if it never touches before expiry, that\'s recorded as an expired, unfilled signal, not '
    'a trade. Known simplifications: no partial TP1/TP2 scale-out (trade closes fully at SL, TP3, or '
    'max-hold), no cross-symbol correlation/portfolio limits during the backtest, and same-candle '
    'SL-vs-TP ties are resolved conservatively (SL assumed to hit first).</div>',
    unsafe_allow_html=True,
)
st.write("")

cfg = Config()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 Backtest Settings")

symbol = st.sidebar.text_input("Symbol", value="BTCUSDT").strip().upper()

col_a, col_b = st.sidebar.columns(2)
default_end = date.today() - timedelta(days=1)
default_start = default_end - timedelta(days=90)
start_date = col_a.date_input("Start date", value=default_start, max_value=default_end)
end_date = col_b.date_input("End date", value=default_end, max_value=default_end)

with st.sidebar.expander("Timeframes", expanded=False):
    tf_options = ["1m", "5m", "15m", "1h", "4h"]
    cfg.htf = st.selectbox("Higher timeframe", tf_options, index=tf_options.index("4h"))
    cfg.mtf_confirm = st.selectbox("Confirmation timeframe", tf_options, index=tf_options.index("1h"))
    cfg.ltf_confirm = st.selectbox("Setup timeframe", tf_options, index=tf_options.index("15m"))
    cfg.entry_tf = st.selectbox("Entry timing timeframe", tf_options, index=tf_options.index("5m"))

with st.sidebar.expander("Signal thresholds", expanded=True):
    cfg.min_signal_score = st.slider("Minimum signal score", 0, 100, int(cfg.min_signal_score))
    cfg.min_rr = st.slider("Minimum risk/reward", 1.0, 4.0, float(cfg.min_rr), 0.1)
    cfg.min_directional_edge = st.slider("Minimum directional edge", 0, 50, int(cfg.min_directional_edge))

with st.sidebar.expander("Costs & realism", expanded=True):
    cfg.taker_fee_pct = st.number_input("Taker fee per side (%)", value=cfg.taker_fee_pct, step=0.01, format="%.2f")
    cfg.slippage_pct = st.number_input("Assumed slippage per fill (%)", value=cfg.slippage_pct, step=0.01, format="%.2f")
    cfg.max_hold_candles = st.number_input("Max hold (entry-tf candles)", value=cfg.max_hold_candles, step=10)

with st.sidebar.expander("Risk sizing (for reference only)", expanded=False):
    cfg.account_equity = st.number_input("Account equity (USD)", value=cfg.account_equity, step=100.0)
    cfg.risk_per_trade_pct = st.slider("Risk per trade (%)", 0.1, 5.0, cfg.risk_per_trade_pct, 0.1)
    cfg.atr_sl_multiplier = st.slider("ATR stop-loss buffer (x ATR)", 0.5, 3.0, cfg.atr_sl_multiplier, 0.1)

run = st.sidebar.button("▶️ Run Backtest", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Fetches real historical Binance candles across 4 timeframes with pagination. "
    "Longer windows and lower timeframes mean more API calls — expect this to take "
    "anywhere from several seconds to a couple of minutes."
)

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if run:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    if (end_date - start_date).days > 400:
        st.warning("Windows beyond ~400 days can take a long time to fetch on lower timeframes. Consider narrowing the range.")

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    progress_bar = st.progress(0.0, text="Fetching historical data...")
    status_text = st.empty()

    def progress_cb(count, last_time):
        status_text.caption(f"Fetched {count} candles so far — up to {last_time}")

    try:
        with st.spinner(f"Running event-driven backtest for {symbol}..."):
            trades, unfilled_signals = backtest.run_backtest(symbol, cfg, start_dt, end_dt, progress_cb=progress_cb)
        progress_bar.progress(1.0, text="Done")
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

    progress_bar.empty()
    status_text.empty()

    if not trades and not unfilled_signals:
        st.warning(
            f"No signals were generated for {symbol} between {start_date} and {end_date} at these "
            f"thresholds. That's a valid, honest result — it means the strategy found no qualifying "
            f"setups in this window. Try relaxing thresholds or widening the date range."
        )
        st.stop()

    metrics = backtest.compute_metrics(trades, unfilled_signals)

    st.subheader(f"📊 Results — {symbol} — {start_date} to {end_date}")

    st.markdown("**Signal funnel** — how many setups actually became trades, not just signals")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Signals Generated", metrics["signals_generated"])
    f2.metric("Entries Triggered", metrics["entries_triggered"])
    f3.metric("Expired (No Entry)", metrics["expired_no_entry"])
    f4.metric("Fill Rate", f"{metrics['fill_rate_pct']}%")
    if trades:
        f5.metric("Avg Entry Delay", f"{metrics['avg_entry_delay_minutes']} min")
    st.caption(
        "A signal only becomes a trade once price actually touches the advertised entry zone on a "
        "later candle — entries are never assumed to fill instantly at the signal candle's close."
    )

    if not trades:
        st.warning(
            f"{metrics['signals_generated']} signal(s) were generated but none were ever filled — "
            f"price never touched the advertised entry zone before expiry in this window. This is a "
            f"legitimate result: it means live execution of this exact setup would likely have missed "
            f"every one of these too."
        )
        st.stop()

    st.write("")
    st.markdown("**Trade performance** (computed from filled trades only)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Trades", metrics["total_trades"])
    m2.metric("Win Rate", f"{metrics['win_rate_pct']}%")
    m3.metric("Expectancy (R, net)", f"{metrics['expectancy_r']:+.3f}")
    m4.metric("Profit Factor", metrics["profit_factor"])
    m5.metric("Max Drawdown (R)", f"{metrics['max_drawdown_r']:.2f}")
    m6.metric("Sharpe-like", metrics["sharpe_like"])

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Avg Winner (R)", f"+{metrics['avg_winner_r']}")
    n2.metric("Avg Loser (R)", f"{metrics['avg_loser_r']}")
    n3.metric("TP1 Hit Rate", f"{metrics['tp1_hit_rate_pct']}%")
    n4.metric("TP3 Hit Rate", f"{metrics['tp3_hit_rate_pct']}%")
    n5.metric("Stopped Rate", f"{metrics['stopped_rate_pct']}%")

    if metrics["expectancy_r"] > 0:
        st.success(
            f"✅ Positive net expectancy ({metrics['expectancy_r']:+.3f}R/trade) over {metrics['total_trades']} "
            f"trades in this window, after fees and slippage. This is evidence — not proof — of an edge. "
            f"A small sample can easily be noise; validate across more symbols, periods, and out-of-sample "
            f"data (walk-forward) before trusting it."
        )
    else:
        st.error(
            f"🛑 Negative or zero net expectancy ({metrics['expectancy_r']:+.3f}R/trade) over "
            f"{metrics['total_trades']} trades in this window, after fees and slippage. This is a "
            f"legitimate and useful research result — it means this configuration would have lost money "
            f"here. Don't paper over it; use it to reconsider the scoring weights, thresholds, or strategy "
            f"logic itself."
        )

    st.write("")
    st.subheader("📈 Equity Curve (cumulative R, net of costs)")
    equity_df = pd.DataFrame({
        "Trade #": list(range(1, len(metrics["equity_curve_r"]) + 1)),
        "Cumulative R": metrics["equity_curve_r"],
    })
    st.line_chart(equity_df.set_index("Trade #"))

    st.write("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Performance by Regime")
        by_regime_df = pd.DataFrame(metrics["by_regime"]).T.rename(columns={"count": "Trades", "mean": "Avg R (net)"})
        st.dataframe(by_regime_df, use_container_width=True)

    with col_right:
        st.subheader("Performance by Score Bucket")
        by_score_df = pd.DataFrame(metrics["by_score_bucket"]).T.rename(columns={"count": "Trades", "mean": "Avg R (net)"})
        st.dataframe(by_score_df, use_container_width=True)
        st.caption("If higher score buckets don't show higher average R, the score isn't currently predictive.")

    st.write("")
    st.subheader("📋 Trade Log")
    trades_df = metrics["trades_df"][[
        "symbol", "direction", "signal_time", "signal_price",
        "entry_zone_low", "entry_zone_high", "entry_trigger_time", "entry_delay_minutes",
        "entry_price", "fill_slippage", "exit_time", "exit_reason",
        "score", "directional_edge", "regime", "rr_planned", "rr_realized",
        "r_multiple", "r_multiple_net", "tp1_hit", "tp2_hit", "tp3_hit",
    ]].copy()
    trades_df.columns = [
        "Symbol", "Direction", "Signal Time", "Signal Price",
        "Zone Low", "Zone High", "Entry Trigger Time", "Entry Delay (min)",
        "Fill Price", "Fill vs Signal (Δ)", "Exit Time", "Exit Reason",
        "Score", "Edge", "Regime", "Planned RR", "Realized RR",
        "R (gross)", "R (net)", "TP1", "TP2", "TP3",
    ]
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    st.caption(
        "Planned RR is what the live dashboard's risk/reward gate actually saw (based on the signal "
        "price). Realized RR recomputes the same ratio using the actual fill price — they can differ "
        "slightly since the fill is only ever taken within the entry zone."
    )

    if unfilled_signals:
        with st.expander(f"📭 {len(unfilled_signals)} expired signal(s) that never got filled"):
            unfilled_df = pd.DataFrame([{
                "Symbol": u.symbol, "Direction": u.direction, "Signal Time": u.signal_time,
                "Signal Price": u.signal_price, "Zone Low": u.entry_zone_low, "Zone High": u.entry_zone_high,
                "Expiry": u.expiry_time, "Score": u.score, "Regime": u.regime,
            } for u in unfilled_signals])
            st.dataframe(unfilled_df, use_container_width=True, hide_index=True)

    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download trade log as CSV", csv, f"{symbol}_backtest_trades.csv", "text/csv")

else:
    st.info(
        "Set a symbol and date range in the sidebar, then click **Run Backtest**. "
        "This fetches real historical Binance candles and replays the exact live decision logic "
        "candle-by-candle, with no look-ahead."
    )
