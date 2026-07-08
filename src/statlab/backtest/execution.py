"""Execution simulator: turns orders into fills at the next bar's open.

An order decided using information through day ``t-1`` is executed here at day ``t``'s
(adjusted) open price. Filling at the *next* bar rather than the same close is the second
half of the anti-lookahead discipline (the first being the point-in-time universe): a
strategy can never trade on a price it used to make the decision.
"""

from __future__ import annotations

import pandas as pd

from statlab.backtest.costs import CostModel
from statlab.backtest.events import Fill, Order
from statlab.data.universe import PointInTimeUniverse


class ExecutionSimulator:
    """Fills market orders at the next bar's open, applying the cost model."""

    def __init__(
        self,
        cost_model: CostModel | None = None,
        *,
        vol_window: int = 20,
        adv_window: int = 20,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.vol_window = vol_window
        self.adv_window = adv_window

    def execute(
        self, orders: list[Order], date: pd.Timestamp, universe: PointInTimeUniverse
    ) -> list[Fill]:
        """Execute ``orders`` at ``date``'s open. Orders with no available open are skipped.

        Volatility and ADV for the impact term are read as of ``date`` (causal). An order
        for a ticker with no open price that day simply does not fill — the strategy keeps
        whatever position it had, which is the honest outcome when a name is untradeable.
        """
        fills: list[Fill] = []
        for order in orders:
            if order.quantity == 0.0:
                continue
            price = universe.open_price(date, order.ticker)
            if price is None:
                continue
            vol = universe.volatility_as_of(date, order.ticker, self.vol_window)
            adv = universe.volume_as_of(date, order.ticker, self.adv_window)
            cost = self.cost_model.cost(order.quantity, price, vol, adv)
            fills.append(Fill(date, order.ticker, order.quantity, price, cost))
        return fills
