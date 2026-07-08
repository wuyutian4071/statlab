"""Event-driven backtesting: costs, execution, portfolio accounting, and the engine."""

from __future__ import annotations

from statlab.backtest.costs import CostModel
from statlab.backtest.engine import BacktestEngine, BacktestResult, sharpe_ratio
from statlab.backtest.events import Fill, MarketEvent, Order
from statlab.backtest.execution import ExecutionSimulator
from statlab.backtest.pairs import PairsStrategy
from statlab.backtest.portfolio import Portfolio
from statlab.backtest.strategy import BuyAndHoldStrategy, Strategy

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BuyAndHoldStrategy",
    "CostModel",
    "ExecutionSimulator",
    "Fill",
    "MarketEvent",
    "Order",
    "PairsStrategy",
    "Portfolio",
    "Strategy",
    "sharpe_ratio",
]
