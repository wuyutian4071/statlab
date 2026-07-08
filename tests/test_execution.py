"""Tests for the execution simulator (next-bar-open fills with costs)."""

from __future__ import annotations

import pandas as pd
import pytest

from statlab.backtest import CostModel, ExecutionSimulator, Order
from statlab.data import PointInTimeUniverse
from statlab.data.schema import BAR_COLUMNS


def _two_day_universe() -> PointInTimeUniverse:
    """A tiny two-ticker, three-day universe with known opens/closes/volume."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    rows = []
    specs = {"AAA": (100.0, 101.0), "BBB": (200.0, 202.0)}  # (open, close)
    for tkr, (op, cl) in specs.items():
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": tkr,
                    "open": op + i,
                    "high": cl + i + 1,
                    "low": op + i - 1,
                    "close": cl + i,
                    "adj_close": cl + i,
                    "volume": 1_000_000.0,
                }
            )
    bars = pd.DataFrame(rows)[list(BAR_COLUMNS)]
    return PointInTimeUniverse.from_bars(bars)


class TestExecution:
    def test_fills_at_the_open_of_the_given_date(self) -> None:
        u = _two_day_universe()
        sim = ExecutionSimulator(CostModel(commission_min=0.0, half_spread_bps=0.0, impact_eta=0.0))
        fills = sim.execute([Order("AAA", 10)], pd.Timestamp("2020-01-03"), u)
        assert len(fills) == 1
        # open of AAA on 2020-01-03 is 100 + 1 = 101 (adj factor is 1 here)
        assert fills[0].price == pytest.approx(101.0)
        assert fills[0].quantity == 10

    def test_cost_is_applied(self) -> None:
        u = _two_day_universe()
        sim = ExecutionSimulator(CostModel(commission_per_share=0.01, commission_min=1.0))
        fills = sim.execute([Order("AAA", 1000)], pd.Timestamp("2020-01-03"), u)
        assert fills[0].cost > 0.0

    def test_zero_quantity_order_is_skipped(self) -> None:
        u = _two_day_universe()
        sim = ExecutionSimulator()
        assert sim.execute([Order("AAA", 0)], pd.Timestamp("2020-01-03"), u) == []

    def test_unknown_ticker_does_not_fill(self) -> None:
        u = _two_day_universe()
        sim = ExecutionSimulator()
        assert sim.execute([Order("ZZZ", 10)], pd.Timestamp("2020-01-03"), u) == []

    def test_non_trading_date_does_not_fill(self) -> None:
        u = _two_day_universe()
        sim = ExecutionSimulator()
        # 2020-01-04 is a Saturday: no bar, so no fill.
        assert sim.execute([Order("AAA", 10)], pd.Timestamp("2020-01-04"), u) == []
