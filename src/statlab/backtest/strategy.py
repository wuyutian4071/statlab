"""Strategy interface and simple baseline strategies.

A strategy observes the market through the point-in-time universe (so it is causal by
construction) and returns orders. The real cointegration pairs strategy is wired up in M5;
this module provides the interface plus a buy-and-hold baseline used for testing the engine
and as a performance benchmark.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from statlab.backtest.events import Order
from statlab.data.universe import PointInTimeUniverse


@runtime_checkable
class Strategy(Protocol):
    """Anything that turns a bar (as of ``date``) into a list of orders."""

    def on_bar(self, date: pd.Timestamp, universe: PointInTimeUniverse) -> list[Order]:
        """Return orders to submit after observing information through ``date``."""
        ...


class BuyAndHoldStrategy:
    """Invest a fixed notional equally across ``tickers`` on the first bar, then hold.

    Orders are emitted only once (on the first bar the strategy sees). They fill at the
    next bar's open, after which the strategy is inert — the canonical benchmark.
    """

    def __init__(self, tickers: list[str], notional: float) -> None:
        if not tickers:
            raise ValueError("tickers must be non-empty")
        if notional <= 0:
            raise ValueError("notional must be positive")
        self.tickers = tickers
        self.notional = notional
        self._invested = False

    def on_bar(self, date: pd.Timestamp, universe: PointInTimeUniverse) -> list[Order]:
        if self._invested:
            return []
        per_name = self.notional / len(self.tickers)
        orders: list[Order] = []
        for ticker in self.tickers:
            price = universe.price_as_of(date, ticker)
            if price is None or price <= 0:
                return []  # wait until every leg is priceable before committing
            orders.append(Order(ticker, per_name / price))
        self._invested = True
        return orders
