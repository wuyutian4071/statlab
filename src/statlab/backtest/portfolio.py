r"""Portfolio accounting with a checkable invariant.

Accounting convention (deliberately split so the invariant is non-trivial):

* ``cash`` tracks only the *trade* cashflow — buying ``q`` shares at ``p`` moves cash by
  :math:`-qp`, selling by :math:`+qp`. Transaction costs are **not** taken out of cash.
* ``costs`` accumulates all transaction costs separately.
* Positions are marked to market at the current prices.

Equity is then

.. math::

    \text{equity}(\text{prices}) = \text{cash} + \sum_i q_i\, p_i - \text{costs}.

The invariant this enables (verified in tests): **equity changes only through market moves
on held positions and through transaction costs — trading at the fill price is otherwise
equity-neutral.** Concretely, filling a trade drops equity by exactly the fill's cost, and
between trades equity moves by exactly :math:`\sum_i q_i \Delta p_i`. Any bookkeeping bug
breaks one of these identities.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from statlab.backtest.events import Fill


class Portfolio:
    """Cash, positions, and cost ledger with mark-to-market equity."""

    def __init__(self, initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.costs = 0.0
        self.positions: dict[str, float] = {}
        self._equity_curve: list[tuple[pd.Timestamp, float]] = []

    def apply_fill(self, fill: Fill) -> None:
        """Update cash, position, and cost ledger for a single fill."""
        self.cash -= fill.quantity * fill.price  # buy(+q) reduces cash; sell(-q) adds
        self.costs += fill.cost
        new_qty = self.positions.get(fill.ticker, 0.0) + fill.quantity
        if new_qty == 0.0:
            self.positions.pop(fill.ticker, None)
        else:
            self.positions[fill.ticker] = new_qty

    def apply_fills(self, fills: list[Fill]) -> None:
        for fill in fills:
            self.apply_fill(fill)

    def position_value(self, prices: Mapping[str, float]) -> float:
        """Mark-to-market value of open positions at ``prices``.

        A position whose price is missing from ``prices`` is carried at zero contribution;
        in practice the engine always supplies a price for every held name.
        """
        return sum(qty * prices.get(tkr, 0.0) for tkr, qty in self.positions.items())

    def equity(self, prices: Mapping[str, float]) -> float:
        """Total equity: ``cash + position_value - cumulative costs``."""
        return self.cash + self.position_value(prices) - self.costs

    def mark(self, date: pd.Timestamp, prices: Mapping[str, float]) -> float:
        """Record and return the equity at ``date`` given marking ``prices``."""
        eq = self.equity(prices)
        self._equity_curve.append((date, eq))
        return eq

    @property
    def equity_curve(self) -> pd.Series:
        """The recorded equity time series (index = date)."""
        if not self._equity_curve:
            return pd.Series(dtype=float, name="equity")
        dates, values = zip(*self._equity_curve, strict=True)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name="equity")
