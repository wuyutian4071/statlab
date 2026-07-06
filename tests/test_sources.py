"""Tests for bar data sources (offline synthetic path only; yfinance is network-gated)."""

from __future__ import annotations

import pandas as pd

from statlab.data import SyntheticSource
from statlab.data.schema import (
    BAR_COLUMNS,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    PRICE_COLUMNS,
    VOLUME,
    validate_schema,
)
from statlab.data.sources import BarSource


class TestSyntheticSource:
    def test_is_a_bar_source(self) -> None:
        assert isinstance(SyntheticSource(n=50), BarSource)

    def test_fetch_returns_canonical_schema(self) -> None:
        bars = SyntheticSource(n=100, n_pairs=1, n_noise=1).fetch()
        assert list(bars.columns) == list(BAR_COLUMNS)
        validate_schema(bars)

    def test_row_count(self) -> None:
        src = SyntheticSource(n=120, n_pairs=2, n_noise=3)
        bars = src.fetch()
        n_tickers = 2 * 2 + 3
        assert len(bars) == 120 * n_tickers
        assert bars["ticker"].nunique() == n_tickers

    def test_ohlc_consistency(self) -> None:
        bars = SyntheticSource(n=200, seed=1).fetch()
        assert (bars[HIGH] >= bars[[OPEN, CLOSE]].max(axis=1) - 1e-9).all()
        assert (bars[LOW] <= bars[[OPEN, CLOSE]].min(axis=1) + 1e-9).all()

    def test_prices_positive_and_volume_nonnegative(self) -> None:
        bars = SyntheticSource(n=200, seed=2).fetch()
        for col in PRICE_COLUMNS:
            assert (bars[col] > 0).all()
        assert (bars[VOLUME] >= 0).all()

    def test_truth_matches_requested_pairs(self) -> None:
        src = SyntheticSource(n=50, n_pairs=3, n_noise=1)
        assert len(src.truth) == 3

    def test_reproducible(self) -> None:
        a = SyntheticSource(n=100, seed=9).fetch()
        b = SyntheticSource(n=100, seed=9).fetch()
        pd.testing.assert_frame_equal(a, b)

    def test_fetch_returns_a_copy(self) -> None:
        src = SyntheticSource(n=50)
        first = src.fetch()
        first.iloc[0, 2] = -999.0
        # Mutating the returned frame must not corrupt the source's internal bars.
        assert src.fetch().iloc[0, 2] != -999.0
