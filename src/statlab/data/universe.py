"""Point-in-time universe — the project's primary defence against lookahead bias.

The single most common way a backtest lies to you is by letting a decision at time *t*
peek at data that only became known after *t*: future prices, future universe membership
(survivorship bias), or restated fundamentals. :class:`PointInTimeUniverse` makes that
structurally impossible for price data.

The contract is deliberately narrow:

* Prices are stored privately. There is **no** public accessor that returns the full
  future panel — every read goes through an ``as_of``/``window`` method that clips to
  ``date <= t``.
* Universe membership is time-aware: a ticker is only visible between its first and last
  available observation (or an explicitly supplied listing interval), so a strategy can
  neither trade a name before it listed nor keep trading a delisted one.

Availability model: a daily bar dated ``d`` is considered *known* as of any query time
``t >= d`` (the close of day ``d`` is observable at the end of day ``d``). Execution lag
— acting on that information only at the next bar — is the *strategy's* responsibility and
is enforced by the backtester (M4), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from statlab.data.schema import ADJ_CLOSE, CLOSE, DATE, OPEN, TICKER, VOLUME, to_price_panel

Timestampable = str | pd.Timestamp


def _ts(t: Timestampable) -> pd.Timestamp:
    return t if isinstance(t, pd.Timestamp) else pd.Timestamp(t)


@dataclass(frozen=True)
class Membership:
    """Listing intervals per ticker: ``ticker -> (first_date, last_date)`` inclusive.

    A ticker is a member of the universe on date ``t`` iff ``first_date <= t <=
    last_date``. This is what makes the universe survivorship-bias-aware: delisted names
    remain in history and are simply not members after their last date.
    """

    intervals: dict[str, tuple[pd.Timestamp, pd.Timestamp]]

    def is_member(self, ticker: str, t: pd.Timestamp) -> bool:
        span = self.intervals.get(ticker)
        if span is None:
            return False
        first, last = span
        return first <= t <= last

    def members_as_of(self, t: pd.Timestamp) -> list[str]:
        return sorted(tkr for tkr in self.intervals if self.is_member(tkr, t))

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> Membership:
        """Derive membership from a wide price panel's first/last non-NaN observation."""
        intervals: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for ticker in panel.columns:
            valid = panel[ticker].dropna()
            if valid.empty:
                continue
            intervals[str(ticker)] = (
                pd.Timestamp(valid.index[0]),
                pd.Timestamp(valid.index[-1]),
            )
        return cls(intervals)


class PointInTimeUniverse:
    """A price panel that can only ever be observed *as of* a given date.

    Construct from long-form bars via :meth:`from_bars`, or directly from a wide price
    panel. All read methods clip to ``date <= t`` and to the tickers that are members at
    ``t``.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        *,
        volume: pd.DataFrame | None = None,
        opens: pd.DataFrame | None = None,
        membership: Membership | None = None,
    ) -> None:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must be indexed by a DatetimeIndex")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted ascending")
        if prices.index.has_duplicates:
            raise ValueError("prices index must not contain duplicate dates")

        # Store privately; the class exposes no getter for the raw future frame.
        # ``prices`` is the marking price (adjusted close); ``opens`` is the adjusted
        # open used for realistic next-bar fills; ``volume`` feeds the ADV cost term.
        self._prices = prices.sort_index(axis=1)
        self._volume = volume.sort_index(axis=1) if volume is not None else None
        self._opens = opens.sort_index(axis=1) if opens is not None else None
        self._membership = membership or Membership.from_panel(self._prices)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        *,
        price_field: str = ADJ_CLOSE,
        membership: Membership | None = None,
    ) -> PointInTimeUniverse:
        """Build a universe from canonical long-form bars.

        The open used for fills is *adjusted* by the same factor as the close
        (``open * adj_close / close``) so fills and marks live on the same, split-adjusted
        price scale. On the synthetic data (no corporate actions) this factor is 1.
        """
        prices = to_price_panel(bars, field=price_field)
        volume = to_price_panel(bars, field=VOLUME)

        adj_factor = bars[ADJ_CLOSE] / bars[CLOSE]
        adjusted = bars.assign(_adj_open=bars[OPEN] * adj_factor)
        opens = (
            adjusted.pivot_table(index=DATE, columns=TICKER, values="_adj_open", aggfunc="last")
            .sort_index()
            .sort_index(axis=1)
        )
        return cls(prices, volume=volume, opens=opens, membership=membership)

    # ------------------------------------------------------------------ introspection
    @property
    def tickers(self) -> list[str]:
        """All tickers ever present (not point-in-time; for setup/reporting only)."""
        return [str(c) for c in self._prices.columns]

    def trading_days(self, start: Timestampable, end: Timestampable) -> pd.DatetimeIndex:
        """Dates in ``[start, end]`` present in the panel — the backtest clock source."""
        lo, hi = _ts(start), _ts(end)
        mask = (self._prices.index >= lo) & (self._prices.index <= hi)
        return pd.DatetimeIndex(self._prices.index[mask])

    def members_as_of(self, t: Timestampable) -> list[str]:
        """Tickers that are universe members as of ``t`` (survivorship-aware)."""
        return self._membership.members_as_of(_ts(t))

    # ------------------------------------------------------------------ point-in-time reads
    def as_of(self, t: Timestampable, *, members_only: bool = True) -> pd.DataFrame:
        """Return the price panel observable as of ``t`` — rows with ``date <= t``.

        With ``members_only`` (default), columns are restricted to tickers that are
        members at ``t``, so names not yet listed are invisible. The result is a copy;
        callers cannot mutate internal state.
        """
        ts = _ts(t)
        frame = self._prices.loc[self._prices.index <= ts]
        if members_only:
            members = self.members_as_of(ts)
            frame = frame.loc[:, [c for c in frame.columns if c in members]]
        return frame.copy()

    def window(self, t: Timestampable, size: int, *, members_only: bool = True) -> pd.DataFrame:
        """Return the last ``size`` observations up to and including ``t``."""
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        return self.as_of(t, members_only=members_only).tail(size)

    def asof_date(self, t: Timestampable) -> pd.Timestamp | None:
        """The latest available trading date ``<= t`` (``None`` if none exists)."""
        ts = _ts(t)
        idx = self._prices.index[self._prices.index <= ts]
        return pd.Timestamp(idx[-1]) if len(idx) else None

    def price_as_of(self, t: Timestampable, ticker: str) -> float | None:
        """Latest known price for ``ticker`` as of ``t`` (``None`` if unavailable)."""
        ts = _ts(t)
        if not self._membership.is_member(ticker, ts) or ticker not in self._prices.columns:
            return None
        series = self._prices[ticker].loc[self._prices.index <= ts].dropna()
        return float(series.iloc[-1]) if not series.empty else None

    def volume_as_of(self, t: Timestampable, ticker: str, window: int = 20) -> float | None:
        """Average daily volume over the last ``window`` days as of ``t`` (for ADV)."""
        if self._volume is None or ticker not in self._volume.columns:
            return None
        ts = _ts(t)
        series = self._volume[ticker].loc[self._volume.index <= ts].dropna().tail(window)
        return float(series.mean()) if not series.empty else None

    def volatility_as_of(self, t: Timestampable, ticker: str, window: int = 20) -> float | None:
        """Trailing daily log-return volatility as of ``t`` (feeds the impact cost term).

        Causal: uses only returns dated ``<= t``. Returns ``None`` if there is not enough
        history to compute at least two returns.
        """
        if ticker not in self._prices.columns:
            return None
        ts = _ts(t)
        series = self._prices[ticker].loc[self._prices.index <= ts].dropna()
        rets = np.log(series).diff().dropna().tail(window)
        if len(rets) < 2:
            return None
        return float(rets.std(ddof=0))

    def open_price(self, date: Timestampable, ticker: str) -> float | None:
        """The (adjusted) open on the exact trading day ``date`` — the next-bar fill price.

        Returns ``None`` if there is no bar for that ticker on that date. Unlike the
        ``as_of`` reads this asks for one specific day, which is exactly what execution
        needs: an order decided at ``t-1`` fills at ``t``'s open.
        """
        if self._opens is None or ticker not in self._opens.columns:
            return None
        ts = _ts(date)
        if ts not in self._opens.index:
            return None
        value = self._opens.loc[ts, ticker]
        return None if pd.isna(value) else float(cast(float, value))

    def close_row(self, t: Timestampable) -> dict[str, float]:
        """Marking prices (adjusted close) for all members with data as of ``t``.

        Returns a ``{ticker: price}`` mapping from the latest available row ``<= t`` — the
        prices the portfolio is marked to at the close of ``t``.
        """
        frame = self.as_of(t)
        if frame.empty:
            return {}
        last = frame.iloc[-1].dropna()
        return {str(k): float(v) for k, v in last.items()}
