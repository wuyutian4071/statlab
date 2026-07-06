"""Signal research: cointegration, half-life, Kalman hedge ratios, and z-score signals."""

from __future__ import annotations

from statlab.signals.cointegration import (
    EngleGrangerResult,
    JohansenResult,
    engle_granger,
    johansen,
)
from statlab.signals.discovery import PairCandidate, discover_pairs
from statlab.signals.half_life import half_life
from statlab.signals.kalman import KalmanResult, kalman_hedge
from statlab.signals.zscore import SignalParams, generate_positions, rolling_zscore

__all__ = [
    "EngleGrangerResult",
    "JohansenResult",
    "KalmanResult",
    "PairCandidate",
    "SignalParams",
    "discover_pairs",
    "engle_granger",
    "generate_positions",
    "half_life",
    "johansen",
    "kalman_hedge",
    "rolling_zscore",
]
