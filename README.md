# Binance Signal Research Dashboard (Streamlit)

A **signal-only** crypto research tool for Binance markets. It analyzes public
market data across multiple timeframes and shows you BUY / SELL / NO TRADE
setups with entry zone, stop-loss, take-profits, risk/reward, position
sizing, and a full explanation of the reasoning — for **manual execution
only**.

**This app never places, modifies, or cancels any order, and no Binance API
key is required.** It only reads Binance's public market-data endpoints.

## What it does

- Fetches OHLCV candles across 4 timeframes (configurable, default 4h / 1h / 15m / 5m)
- Detects market regime (trend strength via ADX + EMA structure, volatility via ATR percentile)
- Detects market structure (swing highs/lows, HH/HL vs LH/LL, break of structure, support/resistance zones)
- Computes trend, momentum, volatility, and volume indicators
- Combines everything into a configurable weighted 0–100 score
- Applies a NO-TRADE gate: low score, poor risk/reward, conflicting higher timeframes, or bad data all default to **NO TRADE**
- Calculates stop-loss (structure + ATR buffer) and three take-profit levels (1R / 2R / nearest realistic structure target)
- Sizes the position from your account equity and risk % — never a fixed arbitrary %
- **Actually enforces**, not just displays: max open setups (keeps only the top-N by score), correlated exposure (excludes the weaker of two correlated setups), and a daily loss limit (blocks all new setups once hit)
- Shows the reasoning **for** and **against** every signal, and which risk control excluded a setup if it was excluded

## Fixes applied after external code review (2nd pass)

An independent review of the first build caught several real logic bugs. All were verified against the code and fixed:

1. **RR was always exactly 1.0, so almost every signal was rejected.** The old code measured risk/reward against TP1, which was itself defined as exactly `1R` from entry — so `reward/risk` was mathematically guaranteed to equal 1.0 regardless of the setup, and would fail against the 1.5 default minimum every time. RR is now measured against TP3 (the nearest realistic structure-derived target, not the furthest one as before), which is an independent, market-derived distance.
2. **The confirmation-timeframe EMA-stack check was a silent no-op.** `ema_bullish_stack`/`ema_bearish_stack` are computed inside `regime.classify_regime()`, but the scoring code was reading those keys off a raw indicator dataframe row where they never existed — so that check always evaluated to nothing. Fixed to pass the actual regime dict through.
3. **Higher-timeframe conflict was detected but never blocked a trade** — it only appended a note. It's now a hard gate: if the higher timeframe and the confirmation timeframe disagree, the result is forced to NO TRADE (the 5m entry timeframe is still allowed to diverge without blocking, matching the spec's "1M shouldn't dominate" principle).
4. **Max open setups, correlation, and daily loss limit were warnings only.** All three now actually change which setups are marked eligible: correlated pairs keep only the higher-scored setup, max-open-setups keeps only the top-N by score, and a breached daily loss limit blocks every new setup outright.
5. **TP3 picked the furthest opposing structure zone**, which could be an unrealistically distant target. It now picks the *nearest* opposing zone beyond entry.

## Fixes applied after a second round of review (3rd pass)

1. **No directional-edge gate.** A score of "68/100" could come from Bull=68 vs Bear=64 (barely a lean) or Bull=68 vs Bear=20 (a decisive lean) — the headline score alone couldn't tell those apart. There's now a configurable `min_directional_edge` (default 10 points) that blocks trades where the winning side doesn't beat the losing side by a meaningful margin. Not claimed to be an optimal value — like all thresholds here, it should be validated, not trusted by default.
2. **"Max open setups" was a misleading name.** There is no persistent trade ledger — the app has no memory of trades you've actually taken. It's been renamed everywhere to "max eligible signals per scan," and the UI now says so explicitly.
3. **Live and backtest logic were at risk of drifting apart.** The decision logic (regime → structure → scoring → gates → risk) has been extracted into a single function, `signals.generate_signal_from_frames()`, called by both the live app and the backtester. There is now exactly one implementation of "what counts as a signal," not two that could quietly disagree.
4. **Built the backtesting engine** (see below) — this is the biggest addition, because until now there was literally zero evidence this scoring model does anything useful.

## The backtester (`backtest.py` + the "Backtest" page)

Run `streamlit run app.py` and use the sidebar page navigation to switch to **Backtest**, or drive it directly:

```python
from config import Config
from backtest import run_backtest, compute_metrics
import pandas as pd

cfg = Config()
trades = run_backtest("BTCUSDT", cfg, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-01"))
metrics = compute_metrics(trades)
print(metrics["expectancy_r"], metrics["win_rate_pct"], metrics["profit_factor"])
```

What it does:
- Fetches real historical Binance candles (paginated) for all 4 configured timeframes.
- Walks forward through the entry timeframe candle-by-candle. At each point, every timeframe is sliced to **only candles closed by that timestamp** (`data_fetch.slice_as_of`) — no future data is ever visible to a decision.
- Calls `signals.generate_signal_from_frames()` — the exact same function the live app uses — so backtest results reflect what the live strategy would actually have done, not a separate reimplementation that could drift from it.
- Simulates each triggered trade forward against real future candles to determine whether SL, TP1/TP2/TP3, or signal expiry was hit first, applying configurable trading fees and slippage.
- Reports win rate, average R, expectancy, profit factor, max drawdown, a Sharpe-like ratio, TP hit rates, and performance broken down by regime, symbol, and score bucket — the last one directly tests whether a higher score has actually meant a better outcome historically.

Known simplifications in this first version (fixable, but not yet done):
- **No partial scale-out at TP1/TP2** — a trade is modeled as fully closing at either the stop-loss or TP3 (or expiry). TP1/TP2 hits are recorded for hit-rate stats but don't currently split the position.
- **No cross-symbol portfolio simulation** — correlation limits and max-eligible-signals-per-scan are live-dashboard concepts; the backtester tests each symbol independently.
- **Same-candle SL/TP ties resolve conservatively** — if a single candle's range contains both the stop-loss and a take-profit, the stop-loss is assumed to trigger first (worst case), since OHLC data alone can't tell you the true intra-candle order.
- This backtest was validated end-to-end against synthetic price data during development (to confirm the walk-forward/no-look-ahead mechanics are correct), not yet run against real Binance history — do that first, on more than one symbol and period, before drawing any conclusion about whether this strategy has an edge.

## Fixes applied after a fourth round of review: the entry-model correctness pass

This pass deliberately changed **nothing** about strategy logic, scoring weights, indicators, thresholds, BOS/S/R logic, or regime logic — only the backtester's execution model, per an explicit request to fix backtester correctness before touching strategy performance.

**1. Files changed:** `backtest.py` (rewritten), `pages/1_Backtest.py` (updated for new return signature and funnel metrics), `tests/test_backtest_entry_model.py` (new).

**2. Execution model implemented:** `SIGNAL GENERATED → WAITING FOR ENTRY → ENTRY TRIGGERED → OPEN → EXIT`. Previously the backtester used `entry_price = sig["current_price"]` — an instant fill at the signal candle's close — while the live dashboard displays an `entry_zone` (current_price ± 0.15×ATR) that a real user would wait for. Those were two different strategies being conflated. Now: a signal only becomes a trade once a **later** candle's `[low, high]` range actually overlaps the entry zone; if that never happens before the signal's expiry, it's recorded as `EXPIRED_NO_ENTRY`, not a trade.

**3. Fill-price rule:** deterministic "filled at the boundary first crossed" — if the triggering candle's open is already inside the zone, fill = open; if price approached from below, fill = zone_low; if from above, fill = zone_high. This deliberately avoids assuming an optimistic mid-zone fill.

**4. Gap rule:** if a candle gaps across the zone entirely (e.g. open below the zone, close above it), it still counts as touched as long as its `[low, high]` range overlapped the zone, and fill uses the same boundary rule above — no special mid-zone assumption on gaps.

**5. Same-candle rule:** if the triggering candle's own range *also* reaches the stop-loss or a take-profit after the fill, the stop-loss is assumed to hit first (unchanged conservative convention from earlier passes) — true intra-candle order can't be recovered from OHLC data alone.

**6. Signal expiry semantics:** expiry only blocks *entry* (a zone touch after expiry doesn't count). Once filled, expiry no longer applies — the position is only closed by stop-loss, TP3, or the max-hold-candles limit (renamed `MAX_HOLD_EXIT` to distinguish it from `EXPIRED_NO_ENTRY`, which is now a pre-entry-only outcome).

**7. TP1/TP2 status:** unchanged from before, now stated more explicitly — milestones only, tracked for hit-rate reporting, never a partial close. The full simulated position remains open until SL, TP3, or max-hold. The backtest page and README both now say this directly rather than leaving it implicit.

**8. Tests added:** `tests/test_backtest_entry_model.py`, 10 unit tests, all passing — BUY/SELL zone-touch triggers, zone-never-touched expiry, price-runs-away (no fake entry), entry-then-SL, entry-then-TP3, zone-touched-after-expiry (correctly rejected), cost model direction-correctness, same-candle SL/TP conservative resolution, and a structural no-look-ahead check on the trigger scan. Run with `python tests/test_backtest_entry_model.py`.

**9. Remaining known limitations (unchanged by this pass, on purpose):** no partial TP1/TP2 scale-out; no portfolio-level correlation/multi-symbol simulation (each symbol backtested independently — this is the next planned phase); one pending signal (waiting-or-open) per symbol at a time; the underlying strategy/scoring logic itself has not been touched or re-evaluated in this pass.

**10. Before vs. after — what the backtester is actually measuring:**
- *Before:* "If I could magically get filled at the exact price the moment this signal appeared, what would happen?" — a strategy nobody can actually execute, since it ignores the fact that the live dashboard tells you to wait for a zone.
- *After:* "If I actually waited for price to touch the advertised entry zone — and walked away when it never came before expiry — what would happen?" — the strategy the live dashboard actually implies.

This was validated end-to-end against synthetic data (not yet real Binance history): the same synthetic dataset that previously produced 6 trades assuming instant fills now correctly produces only 2 filled trades and 1 `EXPIRED_NO_ENTRY` at the same 3 signal-generation points, with realistic entry delays (10–15 minutes) and fill prices that differ from the original signal price — exactly the behavior this fix was meant to produce. It has **not** been run against real market data yet.

I am not calling this backtester "realistic" or "production-ready." It is more correct than the previous version specifically about *when and whether* a signal becomes a trade. Portfolio-level effects, partial scale-outs, and walk-forward validation are still open, and the underlying strategy/scoring logic is exactly as unvalidated as it was before this pass — this pass only makes sure that whatever the strategy's true performance turns out to be, we're now measuring it against the entry behavior the dashboard actually implies.

## Fixes applied after a fifth round of review: TP-before-entry ordering + RR/fill accounting

Two more real issues were found by re-reading the actual entry-lifecycle code, not just trusting that "10 tests passing" meant it was airtight. Both fixed, strategy/scoring logic untouched again.

**1. TP-before-entry ordering ambiguity.** The trigger candle (where the entry zone is touched) was being fully simulated for SL *and* TP hits using that same candle's entire high/low range. But entry happens intrabar on that candle — if the same candle's range also reached a take-profit, there was no way to know whether the TP was reached *before* the zone was ever touched (which would mean the "entry" was fake — the win happened before the trade existed) or after (a real win). The old code always credited it as a win. Fixed: on the trigger candle specifically, only the stop-loss is checked (conservative in both possible orderings — either the trade never should have existed, or it's a real loss, so "assume the loss" is safe either way); take-profit hits and milestones are only evaluated starting from the *next* candle, where entry has unambiguously already happened.

**2. RR/position-sizing basis mismatch made explicit.** Stop-loss, TP1/TP2/TP3, the live risk/reward gate, and position sizing are all computed from `signal_price` (the price at signal generation) — that's also what a live user would actually see and act on. The backtester's actual fill can differ slightly from that (bounded by the entry-zone width, ±0.15×ATR), so the R multiple realized in backtest was silently computed against a different reference price than the one the live RR gate checked. This wasn't wrong, exactly, but it was an unstated inconsistency. Fixed: every `Trade` now records both `rr_planned` (signal-price basis, matches the live gate) and `rr_realized` (actual-fill-price basis), plus `fill_slippage` (how far the fill drifted from the signal price) — visible in the trade log rather than silently absorbed. Position sizing is still not recalculated post-fill; that remains a known, now-explicitly-documented limitation.

**Tests:** 6 new tests added to `tests/test_backtest_entry_model.py` (16 total, all passing): TP-in-trigger-candle correctly NOT counted as a win (the exact bug reported), the general "TP before entry" case, zone-touch-then-later-TP as a valid win, zone-touch-then-later-SL as a valid loss, SL-in-trigger-candle still honored (regression check), and realized RR numerically differing from planned RR when fill differs from signal price.

**Not addressed in this pass (still open, on purpose):** portfolio-level/multi-symbol simulation, partial TP1/TP2 scale-out, position-size recalculation after fill, and — most importantly — the underlying strategy/scoring logic remains exactly as unvalidated as before. This pass only makes the backtester more honest about *when* a trade counts as a win; it says nothing new about whether the strategy is good. The actual next step is still to run this against real Binance history across multiple symbols and periods and look at what `expectancy_r` says.

## What's still simplified / not yet built

Still worth adding once you've actually run the backtester and looked at the results:

- **Walk-forward validation** — train/validate/lock/test-on-unseen-data discipline, so you don't just curve-fit thresholds to one historical window.
- Momentum/volume scoring is still fairly simple (binary MACD sign, a 10-candle OBV comparison) rather than incorporating slope, divergence, or breakout-specific volume patterns.
- Several computed indicators (VWAP, +DI/-DI, Stochastic RSI, Bollinger Bands) aren't currently used anywhere in scoring — they add the appearance of sophistication without contributing to the decision. Worth either wiring them in deliberately or removing them.
- Support/resistance zones are clustered swing points, not fully role-validated (a zone below price is assumed support, above is assumed resistance) — no separate handling for role-reversal after a break.
- The entry timeframe's bias is still just the MACD histogram's sign — this should eventually be a real entry-trigger check (pullback confirmation, liquidity sweep + rejection, breakout retest) rather than one oscillator's sign.
- Break-of-structure detection is a simple "close beyond last swing" check — no distinction between genuine displacement, a liquidity sweep, or noise.
- VWAP is cumulative over the fetched window, not reset per session/day.
- Spot market data only (`api.binance.com`); no Futures endpoint, so there's no long/short distinction beyond BUY/SELL naming.
- Data validation checks nulls/duplicates/staleness but not OHLC internal consistency or interval-gap continuity.
- No position/notional/lot-size constraints, no persistent signal journal (PostgreSQL), no full signal lifecycle state machine, no ML probability layer, no live WebSocket streaming (still polls REST), no Telegram/Discord/email alerts.

Say the word and any of these can be tackled next — but the honest next step is to actually run the backtester on real data across a few symbols and periods and see what it says, before adding anything else.

## Setup

```bash
cd binance_signal_system
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

No `.env` file or API key is needed — this app only calls Binance's public
`/api/v3/klines` endpoint.

## Using it

1. In the sidebar, set your watchlist symbols, timeframes, thresholds, and risk settings.
2. Click **Generate Signals**.
3. The ranking table shows every symbol's score, direction, and regime.
4. Scroll down for full detail cards: entry zone, stop-loss, TP1–TP3, position size, and the reasoning for/against.
5. Optionally enable auto-refresh (polls every 30s).

## Important disclaimers

- **No signal is a guarantee of profit.** Scores reflect configurable rule-based setup quality, not a calibrated win probability.
- This tool has **not been backtested** in this build — treat every signal as hypothetical until you've validated the logic against historical data (the backtesting engine is the natural next phase).
- Always **paper trade** before risking real capital.
- You are solely responsible for any trades you place manually based on this tool's output.
- This is not financial advice.

## Tuning weights responsibly

The scoring weights in the sidebar are defaults, not proven-optimal values.
Before trusting a weighting scheme, validate it out-of-sample (a future
backtesting phase) rather than hand-tuning until backtest curves look
perfect — that's a classic overfitting trap.
