"""statlab: a statistical-arbitrage research platform and event-driven backtester.

The package is organised around a strict research workflow:

``data``       point-in-time data ingestion, validation, storage, and simulation.
``signals``    cointegration tests, half-life estimation, Kalman hedge ratios, z-scores.
``backtest``   an event-driven engine with realistic costs and portfolio accounting.
``validation`` walk-forward analysis and multiple-testing-aware performance metrics.
``report``     HTML tear-sheet generation.

Design philosophy: sloppy backtesting is treated as a bug. Every component is built to
avoid lookahead bias, model transaction costs honestly, and be fully reproducible from a
seed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
