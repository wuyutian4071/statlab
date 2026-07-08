"""Event types for the event-driven backtester.

The engine implements the canonical loop

    MarketEvent -> Strategy -> Order -> ExecutionSimulator -> Fill -> Portfolio

These small immutable records are the messages passed along that chain. Keeping them as
frozen dataclasses (rather than dicts) makes the data flow explicit and type-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    """A new bar has arrived; the strategy may now act on information up to ``date``."""

    date: pd.Timestamp


@dataclass(frozen=True)
class Order:
    """A market order to trade ``quantity`` (signed) shares of ``ticker``.

    Positive ``quantity`` buys, negative sells (including opening/adding to a short).
    Only market orders are modelled in this milestone; they fill at the next bar's open.
    """

    ticker: str
    quantity: float


@dataclass(frozen=True)
class Fill:
    """The result of executing an order: ``quantity`` shares at ``price`` plus ``cost``.

    ``price`` is the (adjusted) fill price before costs; ``cost`` is the total transaction
    cost (commission + half-spread + market impact), always non-negative.
    """

    date: pd.Timestamp
    ticker: str
    quantity: float
    price: float
    cost: float
