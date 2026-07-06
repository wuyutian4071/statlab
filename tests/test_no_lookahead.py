"""Proof-by-test that the point-in-time universe cannot leak future information.

This module is the project's headline correctness guarantee. Lookahead bias — a decision
at time *t* using data only knowable after *t* — is the single most common way a backtest
inflates its results. Rather than trusting the implementation by inspection, we assert the
defining invariants directly and, where possible, *exhaustively* over every date.

Two kinds of guarantee are proven:

1. **Clipping** — every read method returns only rows dated ``<= t``. Checked for every
   trading day in the panel, for both the price view and derived reads.
2. **Append-invariance (the decisive one)** — the value returned for a past date *t* must
   be a pure function of data available at *t*. We verify this by building a universe on a
   *prefix* of history and asserting it returns exactly the same thing as a universe built
   on the full history, for every *t* in the prefix. If future rows could influence a past
   read, this equality would break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statlab.data import PointInTimeUniverse, SyntheticSource
from statlab.data.schema import TICKER, to_price_panel


def _full_bars() -> pd.DataFrame:
    return SyntheticSource(n=300, n_pairs=2, n_noise=2, seed=123).fetch()


# --------------------------------------------------------------------- clipping


class TestClipping:
    def test_as_of_never_returns_future_rows(self, universe: PointInTimeUniverse) -> None:
        days = universe.trading_days("2000-01-01", "2100-01-01")
        for t in days:
            frame = universe.as_of(t)
            assert frame.empty or frame.index.max() <= t, f"leak at {t}"

    def test_window_never_returns_future_rows(self, universe: PointInTimeUniverse) -> None:
        days = universe.trading_days("2000-01-01", "2100-01-01")
        for t in days[::5]:
            w = universe.window(t, 20)
            assert w.empty or w.index.max() <= t

    def test_price_as_of_ignores_a_future_spike(self) -> None:
        # Plant an unmistakable spike on the LAST date; no earlier read may ever see it.
        bars = _full_bars()
        tkr = sorted(bars[TICKER].unique())[0]
        last_date = bars["date"].max()
        spike = 1e9
        bars.loc[(bars[TICKER] == tkr) & (bars["date"] == last_date), "adj_close"] = spike
        u = PointInTimeUniverse.from_bars(bars)

        days = u.trading_days("2000-01-01", "2100-01-01")
        for t in days[:-1]:  # every day before the spike
            px = u.price_as_of(t, tkr)
            assert px is None or px < spike / 2, f"future spike leaked into {t}"
        # And it *is* visible once we are at/after the spike date.
        assert u.price_as_of(last_date, tkr) == spike


# --------------------------------------------------------------- append-invariance


class TestAppendInvariance:
    """A past read must not change when future data is appended."""

    def _split(self, bars: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
        days = np.sort(bars["date"].unique())
        split = pd.Timestamp(days[len(days) // 2])
        prefix = bars[bars["date"] <= split].copy()
        return split, prefix

    def test_as_of_is_pure_function_of_the_past(self) -> None:
        full_bars = _full_bars()
        split, prefix_bars = self._split(full_bars)

        full = PointInTimeUniverse.from_bars(full_bars)
        prefix = PointInTimeUniverse.from_bars(prefix_bars)

        for t in prefix.trading_days("2000-01-01", split):
            pd.testing.assert_frame_equal(prefix.as_of(t), full.as_of(t), check_freq=False)

    def test_members_as_of_unaffected_by_future(self) -> None:
        full_bars = _full_bars()
        split, prefix_bars = self._split(full_bars)
        full = PointInTimeUniverse.from_bars(full_bars)
        prefix = PointInTimeUniverse.from_bars(prefix_bars)

        for t in prefix.trading_days("2000-01-01", split):
            assert prefix.members_as_of(t) == full.members_as_of(t)

    def test_price_as_of_unaffected_by_future(self) -> None:
        full_bars = _full_bars()
        split, prefix_bars = self._split(full_bars)
        full = PointInTimeUniverse.from_bars(full_bars)
        prefix = PointInTimeUniverse.from_bars(prefix_bars)

        tickers = sorted(full_bars[TICKER].unique())
        for t in prefix.trading_days("2000-01-01", split)[::7]:
            for tkr in tickers:
                assert prefix.price_as_of(t, tkr) == full.price_as_of(t, tkr)


# ---------------------------------------------------- membership / survivorship


class TestSurvivorship:
    def test_unlisted_ticker_is_invisible_before_listing(self) -> None:
        # Build a panel where BBB only starts trading a third of the way in.
        idx = pd.bdate_range("2021-01-04", periods=30, name="date")
        panel = pd.DataFrame({"AAA": np.arange(30.0), "BBB": np.arange(30.0)}, index=idx)
        panel.loc[idx[:10], "BBB"] = np.nan  # BBB not yet listed
        u = PointInTimeUniverse(panel)

        for t in idx[:10]:
            assert "BBB" not in u.as_of(t).columns
            assert u.price_as_of(t, "BBB") is None
        for t in idx[10:]:
            assert "BBB" in u.as_of(t).columns

    def test_no_read_reveals_a_column_before_it_has_data(self) -> None:
        bars = _full_bars()
        panel = to_price_panel(bars)
        u = PointInTimeUniverse.from_bars(bars)
        first_day = panel.index[0]
        # Sanity: at the very first date, only tickers with a value that day are members.
        members = set(u.members_as_of(first_day))
        have_data = {c for c in panel.columns if not np.isnan(panel.loc[first_day, c])}
        assert members == have_data


# --------------------------------------------------- API surface has no back door


class TestNoBackDoor:
    def test_public_dir_exposes_no_raw_future_frame(self) -> None:
        u = PointInTimeUniverse(
            pd.DataFrame(
                {"AAA": [1.0, 2.0]},
                index=pd.bdate_range("2021-01-04", periods=2, name="date"),
            )
        )
        # No *public* attribute should hand back a DataFrame (which would be the full,
        # unclipped future panel). All price access must go through as_of/window.
        public_attrs = [name for name in dir(u) if not name.startswith("_")]
        for name in public_attrs:
            attr = getattr(u, name)
            assert not isinstance(attr, pd.DataFrame), f"{name} exposes a raw frame"
