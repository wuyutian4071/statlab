r"""A pairs-trading :class:`~statlab.backtest.strategy.Strategy` built from M3's signals.

Wires the already-tested cointegration/Kalman/z-score primitives from :mod:`statlab.signals`
into the M4 engine. No new signal math lives here — this module's job is purely the
*integration*: turning a causal hedge ratio and z-score into correctly-signed, correctly-sized,
correctly-lagged orders.

Performance tradeoff, stated plainly rather than hidden: each bar re-runs the batch
:func:`~statlab.signals.kalman.kalman_hedge` filter over the *entire* history available so far,
rather than maintaining an incremental filter state between bars. This is :math:`O(n)` per bar
and so :math:`O(n^2)` over a full backtest — but it means the live strategy calls the exact same,
already-validated M3 function the exact same way discovery and research do, with zero risk of a
second, subtly-divergent re-implementation of the recursion. At this project's data scale
(hundreds to a few thousand daily bars) this is fast in absolute terms (measured, not assumed —
see the README's M5 section for a real timing). A true low-latency system would maintain
incremental filter state instead; this is a research backtester, and simplicity that's provably
consistent with the tested signal layer wins over a second implementation to keep in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statlab.backtest.events import Order
from statlab.data.universe import PointInTimeUniverse
from statlab.signals.kalman import kalman_hedge
from statlab.signals.zscore import SignalParams, generate_positions


class PairsStrategy:
    """Trade a single cointegrated pair ``(y, x)`` via a causal Kalman hedge ratio and a
    hysteresis z-score state machine.

    Sign convention: ``spread_t = log(y_t) - beta_t * log(x_t) - alpha_t`` (the Kalman filter's
    own innovation). State ``+1`` ("long the spread") is long ``y`` / short ``x``, opened when
    the spread is unusually low; state ``-1`` ("short the spread") is the reverse. Position
    sizing is dollar-neutral-ish: the ``y`` leg gets ``notional / 2`` and the ``x`` leg gets
    ``beta_t`` times that in the opposite economic direction, converted to shares at the current
    price — the standard convention that a regression beta on log prices is (to first order) a
    return-hedge ratio.

    The strategy has no access to the portfolio's actual fills (the ``Strategy`` protocol is
    intentionally one-directional — see ``backtest/strategy.py``), so it tracks its own belief
    about its current position the same way ``BuyAndHoldStrategy`` does, and assumes its orders
    fill exactly as submitted. If a leg is unpriceable at a transition, the strategy stays flat
    rather than risk that belief silently diverging from reality.

    ``delta`` default, and why it's smaller than :func:`kalman_hedge`'s own general-purpose
    default (``1e-4``): with ``Q`` growing every bar (``delta`` allows the state to drift), the
    filter's own uncertainty ``P`` — and so the innovation variance ``S`` the z-score divides
    by — tends to grow over a long series too, which *dampens* z-score sensitivity late in a
    long backtest even while the underlying spread keeps mean-reverting normally. A tighter
    ``delta`` (``1e-6``) keeps ``P`` (and so the z-score's sensitivity) stable across hundreds to
    thousands of bars. This was found empirically while building the CLI demo — worth stating
    plainly since it's the kind of thing that silently produces a "the strategy never trades"
    backtest with no error, not a crash.
    """

    def __init__(
        self,
        y: str,
        x: str,
        notional: float,
        *,
        params: SignalParams | None = None,
        delta: float = 1e-6,
        obs_var: float = 1e-3,
        beta0: float = 0.0,
        alpha0: float = 0.0,
        p0: float = 1.0,
        min_history: int = 60,
    ) -> None:
        if notional <= 0:
            raise ValueError("notional must be positive")
        if min_history < 20:
            raise ValueError("min_history must be at least 20 to fit a meaningful filter")

        self.y = y
        self.x = x
        self.notional = notional
        self.params = params or SignalParams()
        self.delta = delta
        self.obs_var = obs_var
        self.beta0 = beta0
        self.alpha0 = alpha0
        self.p0 = p0
        self.min_history = min_history

        self._state = 0
        self._qty_y = 0.0
        self._qty_x = 0.0

    def on_bar(self, date: pd.Timestamp, universe: PointInTimeUniverse) -> list[Order]:
        history = universe.as_of(date)
        if self.y not in history.columns or self.x not in history.columns:
            return []
        pair = history[[self.y, self.x]].dropna()
        if len(pair) < self.min_history:
            return []

        log_y = np.log(pair[self.y].to_numpy())
        log_x = np.log(pair[self.x].to_numpy())
        result = kalman_hedge(
            log_y,
            log_x,
            delta=self.delta,
            obs_var=self.obs_var,
            beta0=self.beta0,
            alpha0=self.alpha0,
            p0=self.p0,
        )
        positions = generate_positions(result.zscore(), self.params)
        new_state = int(positions[-1])
        beta_t = float(result.beta[-1])

        if new_state == self._state:
            return []

        orders: list[Order] = []

        if self._state != 0:
            if self._qty_y != 0.0:
                orders.append(Order(self.y, -self._qty_y))
            if self._qty_x != 0.0:
                orders.append(Order(self.x, -self._qty_x))
            self._qty_y = 0.0
            self._qty_x = 0.0

        if new_state != 0:
            price_y = universe.price_as_of(date, self.y)
            price_x = universe.price_as_of(date, self.x)
            if price_y is not None and price_y > 0 and price_x is not None and price_x > 0:
                y_dollar = new_state * self.notional / 2.0
                x_dollar = -new_state * beta_t * self.notional / 2.0
                qty_y = y_dollar / price_y
                qty_x = x_dollar / price_x
                orders.append(Order(self.y, qty_y))
                orders.append(Order(self.x, qty_x))
                self._qty_y = qty_y
                self._qty_x = qty_x
            else:
                new_state = 0  # couldn't price the new leg; stay flat rather than guess

        self._state = new_state
        return orders
