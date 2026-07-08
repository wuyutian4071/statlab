"""The event-driven backtest engine.

Loop, per trading day ``t``:

1. **Execute** orders decided on the previous bar at ``t``'s open (next-bar fills).
2. **Mark** the portfolio to market at ``t``'s close and record equity.
3. **Decide**: the strategy observes information through ``t`` and returns orders to be
   executed on the *next* bar.

This ordering is what enforces the one-bar execution lag: a decision at ``t`` can only ever
transact at ``t+1``. Combined with the point-in-time universe, the backtest is structurally
free of lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statlab.backtest.events import Fill, Order
from statlab.backtest.execution import ExecutionSimulator
from statlab.backtest.portfolio import Portfolio
from statlab.backtest.strategy import Strategy
from statlab.data.universe import PointInTimeUniverse, Timestampable


@dataclass
class BacktestResult:
    """Outputs of a backtest run."""

    equity_curve: pd.Series
    fills: list[Fill]
    initial_cash: float
    final_positions: dict[str, float]
    total_costs: float

    def returns(self) -> pd.Series:
        """Daily simple returns of the equity curve."""
        return self.equity_curve.pct_change().dropna()

    @property
    def total_return(self) -> float:
        """Total return over the whole backtest."""
        if len(self.equity_curve) < 2:
            return 0.0
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)


class BacktestEngine:
    """Drives a :class:`Strategy` over a :class:`PointInTimeUniverse`."""

    def __init__(
        self,
        universe: PointInTimeUniverse,
        strategy: Strategy,
        portfolio: Portfolio,
        *,
        execution: ExecutionSimulator | None = None,
    ) -> None:
        self.universe = universe
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution or ExecutionSimulator()

    def run(self, start: Timestampable, end: Timestampable) -> BacktestResult:
        """Run the backtest over ``[start, end]`` and return the result."""
        days = self.universe.trading_days(start, end)
        pending: list[Order] = []
        all_fills: list[Fill] = []

        for t in days:
            if pending:
                fills = self.execution.execute(pending, t, self.universe)
                self.portfolio.apply_fills(fills)
                all_fills.extend(fills)

            self.portfolio.mark(t, self.universe.close_row(t))
            pending = self.strategy.on_bar(t, self.universe)

        return BacktestResult(
            equity_curve=self.portfolio.equity_curve,
            fills=all_fills,
            initial_cash=self.portfolio.initial_cash,
            final_positions=dict(self.portfolio.positions),
            total_costs=self.portfolio.costs,
        )


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of a return series (0 if undefined)."""
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)
