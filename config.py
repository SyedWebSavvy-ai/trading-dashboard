"""
Central configuration. Nothing here is 'the right answer' — these are
starting defaults you are expected to tune and validate via backtesting.
No magic numbers should live anywhere else in the codebase.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    # ---- Universe ----
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
    ])

    # ---- Timeframes ----
    htf: str = "4h"          # higher timeframe -> regime / bias
    mtf_confirm: str = "1h"  # intermediate confirmation
    ltf_confirm: str = "15m" # setup confirmation
    entry_tf: str = "5m"     # entry timing
    candles_per_tf: int = 300

    # ---- Indicator periods ----
    ema_periods: List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_rsi_period: int = 14
    volume_sma_period: int = 20

    # ---- Regime thresholds ----
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 35.0

    # ---- Structure ----
    swing_lookback: int = 3  # candles either side to confirm a fractal swing

    # ---- Scoring weights (must sum to 1.0) ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 0.20,
        "structure": 0.20,
        "momentum": 0.15,
        "volume": 0.15,
        "support_resistance": 0.10,
        "volatility": 0.10,
        "mtf_confirmation": 0.10,
    })

    # ---- Signal thresholds ----
    min_signal_score: float = 65.0   # below this -> NO TRADE regardless
    min_rr: float = 1.5              # minimum acceptable risk:reward
    min_directional_edge: float = 10.0  # min |bull_score - bear_score| required to take a side

    score_bands = [
        (0, 50, "NO TRADE"),
        (50, 65, "WEAK / WATCH"),
        (65, 75, "MODERATE"),
        (75, 85, "HIGH CONFIDENCE"),
        (85, 101, "VERY HIGH CONFIDENCE"),
    ]

    # ---- Risk management ----
    account_equity: float = 1000.0
    risk_per_trade_pct: float = 1.0   # % of equity risked per trade
    atr_sl_multiplier: float = 1.5    # ATR buffer added to structural stop
    max_daily_loss_pct: float = 3.0
    max_open_setups: int = 3
    correlation_warning_threshold: float = 0.75

    # ---- Backtesting realism ----
    taker_fee_pct: float = 0.04        # % per side (Binance spot taker default ~0.04%)
    slippage_pct: float = 0.05         # % assumed adverse slippage per fill
    max_hold_candles: int = 200        # max LTF candles to hold a simulated trade before forcing exit

    # ---- Signal lifecycle ----
    signal_expiry_minutes: int = 90


DEFAULT_CONFIG = Config()
