"""
Streamlit dashboard for the Binance signal-only research system.

IMPORTANT: This app never places, modifies, or cancels any order. It only
reads PUBLIC Binance market data and displays probabilistic, manually-
executed trade setups. No API key is required or used anywhere.
"""
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from config import Config
from data_fetch import get_klines, DataFetchError
from signals import generate_signal
from risk import correlation_matrix

st.set_page_config(
    page_title="Binance Signal Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0b0f14; }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: #131a24; border: 1px solid #1f2937; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .signal-card {
        background: #10151d; border-radius: 14px; padding: 20px 22px;
        margin-bottom: 18px; border: 1px solid #1e2733;
    }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.78rem; letter-spacing: 0.03em;
    }
    .badge-buy { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.4);}
    .badge-sell { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4);}
    .badge-notrade { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.4);}
    .score-pill {
        font-size: 1.6rem; font-weight: 800; color: #e2e8f0;
    }
    .reason-good { color: #4ade80; font-size: 0.88rem; }
    .reason-bad { color: #f87171; font-size: 0.88rem; }
    .tag {
        display:inline-block; background:#1a2230; border:1px solid #2a3444;
        color:#94a3b8; border-radius:6px; padding:1px 8px; font-size:0.75rem;
        margin-right:6px; margin-bottom:4px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .disclaimer {
        background: #1a1207; border: 1px solid #4a3510; color: #fbbf24;
        border-radius: 10px; padding: 12px 16px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — CONFIGURATION
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

cfg = Config()

with st.sidebar.expander("Watchlist", expanded=True):
    default_symbols = ", ".join(cfg.symbols)
    symbols_input = st.text_area("Symbols (comma-separated)", value=default_symbols, height=70)
    cfg.symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

with st.sidebar.expander("Timeframes", expanded=False):
    tf_options = ["1m", "5m", "15m", "1h", "4h"]
    cfg.htf = st.selectbox("Higher timeframe (regime)", tf_options, index=tf_options.index("4h"))
    cfg.mtf_confirm = st.selectbox("Confirmation timeframe", tf_options, index=tf_options.index("1h"))
    cfg.ltf_confirm = st.selectbox("Setup timeframe", tf_options, index=tf_options.index("15m"))
    cfg.entry_tf = st.selectbox("Entry timing timeframe", tf_options, index=tf_options.index("5m"))

with st.sidebar.expander("Signal thresholds", expanded=True):
    cfg.min_signal_score = st.slider("Minimum signal score", 0, 100, int(cfg.min_signal_score))
    cfg.min_rr = st.slider("Minimum risk/reward", 1.0, 4.0, float(cfg.min_rr), 0.1)
    cfg.min_directional_edge = st.slider(
        "Minimum directional edge (bull vs bear score gap)", 0, 50, int(cfg.min_directional_edge),
        help="Blocks trades where bull/bear evidence is nearly tied (e.g. 68 vs 64) even if the "
             "headline score looks high. Not a proven-optimal value — validate via backtesting.",
    )

with st.sidebar.expander("Risk management", expanded=True):
    cfg.account_equity = st.number_input("Account equity (USD)", min_value=0.0, value=cfg.account_equity, step=100.0)
    cfg.risk_per_trade_pct = st.slider("Risk per trade (%)", 0.1, 5.0, cfg.risk_per_trade_pct, 0.1)
    cfg.atr_sl_multiplier = st.slider("ATR stop-loss buffer (x ATR)", 0.5, 3.0, cfg.atr_sl_multiplier, 0.1)
    cfg.max_open_setups = st.number_input(
        "Max eligible signals per scan", min_value=1, value=cfg.max_open_setups, step=1,
        help="This limits how many setups are shown as eligible in a single scan. It is NOT a "
             "persistent open-position tracker — the app has no memory of trades you've already "
             "taken, so it can't know if a signal it showed you 10 minutes ago is 'still open.'",
    )

with st.sidebar.expander("Daily loss limit", expanded=True):
    cfg.max_daily_loss_pct = st.slider("Max daily loss (%)", 0.5, 10.0, cfg.max_daily_loss_pct, 0.5)
    todays_realized_pnl = st.number_input(
        "Today's realized P&L so far (USD, negative if loss)",
        value=0.0, step=10.0,
        help="Manual input — this signal-only app has no account access to read your actual P&L. "
             "Enter it yourself so the daily loss limit can actually block new setups.",
    )

with st.sidebar.expander("Scoring weights", expanded=False):
    st.caption("Must reflect relative importance — not assumed optimal. Validate via backtesting.")
    new_weights = {}
    for k, v in cfg.weights.items():
        new_weights[k] = st.slider(k.replace("_", " ").title(), 0.0, 0.5, v, 0.01)
    total_w = sum(new_weights.values())
    if total_w > 0:
        cfg.weights = {k: v / total_w for k, v in new_weights.items()}

auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)
run_button = st.sidebar.button("🔄 Generate Signals", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "🔒 Signal-only mode. This app reads public Binance market data and "
    "never places, modifies, or cancels trades. All execution is manual."
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📊 Binance Signal Research Dashboard")
st.markdown(
    '<div class="disclaimer">⚠️ <b>No signal here is a guarantee of profit.</b> '
    'Scores are probabilistic estimates of setup quality, not win probabilities. '
    'Always paper-trade and backtest before risking real capital.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------------------------
def band_class(direction):
    if direction == "BUY":
        return "badge-buy"
    if direction == "SELL":
        return "badge-sell"
    return "badge-notrade"


@st.cache_data(ttl=25, show_spinner=False)
def run_all_signals(_cfg_dict, symbols):
    cfg_local = Config(**{k: v for k, v in _cfg_dict.items() if k in Config.__dataclass_fields__})
    results = []
    for sym in symbols:
        try:
            sig = generate_signal(sym, cfg_local)
        except Exception as e:
            sig = {"symbol": sym, "direction": "NO TRADE", "score": 0,
                   "data_status": "DATA INVALID", "reasons_against": [f"Unexpected error: {e}"],
                   "confidence_label": "NO TRADE"}
        results.append(sig)
    return results


should_run = run_button or auto_refresh or "last_signals" not in st.session_state

if should_run:
    with st.spinner("Fetching market data and computing signals..."):
        cfg_dict = {k: getattr(cfg, k) for k in Config.__dataclass_fields__}
        signals = run_all_signals(cfg_dict, tuple(cfg.symbols))
        st.session_state["last_signals"] = signals
        st.session_state["last_run"] = datetime.utcnow()
else:
    signals = st.session_state.get("last_signals", [])

signals = st.session_state.get("last_signals", [])
last_run = st.session_state.get("last_run")

if last_run:
    st.caption(f"Last updated: {last_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")

if not signals:
    st.info("Configure your watchlist and click **Generate Signals** in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# RANKING TABLE
# ---------------------------------------------------------------------------
st.subheader("🏆 Signal Ranking")

ranked = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)
table_rows = []
for s in ranked:
    table_rows.append({
        "Symbol": s["symbol"],
        "Direction": s["direction"],
        "Score": s.get("score", 0),
        "Confidence": s.get("confidence_label", "NO TRADE"),
        "Regime": s.get("regime", "-"),
        "R:R": s.get("rr", "-"),
        "Data": s.get("data_status", "-"),
    })
df_table = pd.DataFrame(table_rows)

def style_direction(val):
    if val == "BUY":
        return "color: #4ade80; font-weight: 700;"
    if val == "SELL":
        return "color: #f87171; font-weight: 700;"
    return "color: #94a3b8;"

styler = df_table.style
if hasattr(styler, "map"):
    styler = styler.map(style_direction, subset=["Direction"])
else:
    styler = styler.applymap(style_direction, subset=["Direction"])

st.dataframe(
    styler,
    use_container_width=True,
    hide_index=True,
)

actionable = [s for s in ranked if s["direction"] in ("BUY", "SELL")]

# ---------------------------------------------------------------------------
# RISK ENFORCEMENT (previously these were warnings only — now they actually
# change which setups are marked eligible for execution)
# ---------------------------------------------------------------------------
daily_loss_limit_usd = cfg.account_equity * (cfg.max_daily_loss_pct / 100)
daily_loss_breached = todays_realized_pnl <= -daily_loss_limit_usd

for s in ranked:
    s["eligible"] = s["direction"] in ("BUY", "SELL")
    s["exclusion_reason"] = None

if daily_loss_breached:
    for s in ranked:
        if s["eligible"]:
            s["eligible"] = False
            s["exclusion_reason"] = "Daily loss limit reached — no new setups until tomorrow"
    st.error(
        f"🛑 **Daily loss limit reached.** Today's realized P&L (${todays_realized_pnl:,.2f}) has hit "
        f"the {cfg.max_daily_loss_pct:.1f}% limit (${daily_loss_limit_usd:,.2f}) of account equity. "
        f"All new setups are marked ineligible below — this is a hard stop, not a suggestion."
    )

# ---------------------------------------------------------------------------
# CORRELATION ENFORCEMENT
# ---------------------------------------------------------------------------
high_corr_pairs = []
if not daily_loss_breached and len(actionable) >= 2:
    try:
        price_data = {}
        for s in actionable:
            df_close = get_klines(s["symbol"], cfg.mtf_confirm, 100)
            price_data[s["symbol"]] = df_close["close"]
        corr = correlation_matrix(price_data)
        syms = list(corr.columns)
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                c = corr.iloc[i, j]
                if abs(c) >= cfg.correlation_warning_threshold:
                    high_corr_pairs.append((syms[i], syms[j], round(c, 2)))
    except Exception:
        pass

    # For each highly-correlated pair, keep only the higher-scored symbol
    # eligible; the weaker one is excluded rather than just warned about,
    # since two correlated BUYs are effectively one position, not two.
    by_symbol = {s["symbol"]: s for s in ranked}
    for a, b, c in high_corr_pairs:
        sig_a, sig_b = by_symbol.get(a), by_symbol.get(b)
        if not sig_a or not sig_b or not sig_a["eligible"] or not sig_b["eligible"]:
            continue
        weaker = sig_a if sig_a["score"] < sig_b["score"] else sig_b
        stronger = sig_b if weaker is sig_a else sig_a
        weaker["eligible"] = False
        weaker["exclusion_reason"] = f"Correlated with {stronger['symbol']} (r={c}) — kept the higher-scored setup only"

    if high_corr_pairs:
        pairs_str = ", ".join([f"{a}/{b} ({c})" for a, b, c in high_corr_pairs])
        st.warning(f"🔗 **High correlated exposure detected:** {pairs_str} — "
                   f"the lower-scored setup in each pair has been excluded below, not just flagged.")

# ---------------------------------------------------------------------------
# MAX OPEN SETUPS ENFORCEMENT
# ---------------------------------------------------------------------------
still_eligible = [s for s in ranked if s["eligible"]]
still_eligible.sort(key=lambda s: s["score"], reverse=True)
if len(still_eligible) > cfg.max_open_setups:
    for s in still_eligible[cfg.max_open_setups:]:
        s["eligible"] = False
        s["exclusion_reason"] = f"Exceeds max eligible signals per scan ({cfg.max_open_setups}) — lower score than others in this scan"
    st.warning(
        f"⚠️ {len(still_eligible)} setups cleared score/RR/correlation filters, but max eligible signals per scan is "
        f"{cfg.max_open_setups}. Only the top {cfg.max_open_setups} by score are marked eligible below. "
        f"Note: this limits *this scan's* results — it doesn't track positions you've actually opened."
    )

st.write("")

# ---------------------------------------------------------------------------
# DETAILED SIGNAL CARDS
# ---------------------------------------------------------------------------
st.subheader("📋 Detailed Signals")

for s in ranked:
    direction = s["direction"]
    badge = band_class(direction)

    with st.container():
        st.markdown('<div class="signal-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(
                f"### {s['symbol']} &nbsp; <span class='badge {badge}'>{direction}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Regime: {s.get('regime', '-')}  •  Data: {s.get('data_status', '-')}")
            if direction in ("BUY", "SELL"):
                if s.get("eligible"):
                    st.markdown("✅ **Eligible** — passes all portfolio-level risk controls")
                else:
                    st.markdown(f"🚫 **Excluded:** {s.get('exclusion_reason', 'Not eligible')}")

        with col2:
            st.markdown(f"<div class='score-pill'>{s.get('score', 0):.0f}/100</div>", unsafe_allow_html=True)
            st.caption(s.get("confidence_label", "NO TRADE"))
            if s.get("bull_score") is not None:
                st.caption(f"Bull {s.get('bull_score', 0):.0f} vs Bear {s.get('bear_score', 0):.0f} "
                           f"(edge: {s.get('directional_edge', 0):.0f})")

        with col3:
            if s.get("rr"):
                st.metric("Risk : Reward", f"1 : {s['rr']}")

        if direction in ("BUY", "SELL"):
            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            entry_low, entry_high = s["entry_zone"]
            c1.metric("Entry Zone", f"{entry_low} – {entry_high}")
            c2.metric("Stop Loss", f"{s['stop_loss']}")
            c3.metric("TP1 / TP2", f"{s['tp1']} / {s['tp2']}")
            c4.metric("TP3 (RR target)", f"{s['tp3']}")
            st.caption(f"TP3/RR basis: {s.get('target_source', '-')} — TP1/TP2 are fixed 1R/2R scale-out levels, "
                       f"not the risk/reward gate.")

            c5, c6, c7 = st.columns(3)
            c5.metric("Position Size", f"{s.get('position_size', 0)} units")
            c6.metric("Position Value", f"${s.get('position_value', 0):,.2f}")
            c7.metric("$ at Risk", f"${s.get('dollar_risk', 0):,.2f}")

            if s.get("confirmations"):
                st.markdown("**Confirmations:**")
                tags_html = "".join(
                    f"<span class='tag'>{tf.replace('_', ' ').upper()}: {bias}</span>"
                    for tf, bias in s["confirmations"].items()
                )
                st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown(f"**Stop-loss method:** {s.get('sl_method', '-')}")
            st.markdown(f"**Invalidation:** {s.get('invalidation', '-')}")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**✅ Why this signal:**")
            if s.get("reasons_supporting"):
                for r in s["reasons_supporting"][:8]:
                    st.markdown(f"<div class='reason-good'>• {r}</div>", unsafe_allow_html=True)
            else:
                st.caption("No supporting evidence met the threshold.")
        with colB:
            st.markdown("**⚠️ Why it might fail / against:**")
            if s.get("reasons_against"):
                for r in s["reasons_against"][:8]:
                    st.markdown(f"<div class='reason-bad'>• {r}</div>", unsafe_allow_html=True)
            else:
                st.caption("No significant opposing evidence.")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Signal-only research tool. Not financial advice. No trade is executed automatically. "
    "Scores reflect setup quality based on configurable rules, not calibrated win probabilities."
)

if auto_refresh:
    time.sleep(30)
    st.rerun()
