"""End-to-end tests for the backtest engine: correctness, invariant, and execution lag."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statlab.backtest import (
    BacktestEngine,
    BuyAndHoldStrategy,
    CostModel,
    ExecutionSimulator,
    Order,
    Portfolio,
    sharpe_ratio,
)
from statlab.backtest.engine import BacktestResult
from statlab.backtest.strategy import Strategy
from statlab.data import PointInTimeUniverse, SyntheticSource


def _universe() -> PointInTimeUniverse:
    return PointInTimeUniverse.from_bars(
        SyntheticSource(n=400, n_pairs=1, n_noise=1, seed=99).fetch()
    )


def _span(u: PointInTimeUniverse) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = u.trading_days("2000-01-01", "2100-01-01")
    return days[0], days[-1]


class _RandomStrategy:
    """Emits random small trades in the universe's members each bar (for stress-testing)."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def on_bar(self, date: pd.Timestamp, universe: PointInTimeUniverse) -> list[Order]:
        members = universe.members_as_of(date)
        orders = []
        for tkr in members:
            if self.rng.random() < 0.3:
                orders.append(Order(tkr, float(self.rng.integers(-50, 50))))
        return orders


class _FirstBarStrategy:
    """Buys once on the very first bar; used to observe the execution lag."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._done = False

    def on_bar(self, date: pd.Timestamp, universe: PointInTimeUniverse) -> list[Order]:
        if self._done:
            return []
        self._done = True
        return [Order(self.ticker, 10)]


class TestEngineBasics:
    def test_buy_and_hold_runs_and_tracks_market(self) -> None:
        u = _universe()
        start, end = _span(u)
        tickers = u.members_as_of(start)[:2]
        strat = BuyAndHoldStrategy(tickers, notional=50_000)
        engine = BacktestEngine(u, strat, Portfolio(100_000))
        result = engine.run(start, end)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(u.trading_days(start, end))
        assert result.fills  # it actually traded

    def test_is_a_strategy(self) -> None:
        assert isinstance(BuyAndHoldStrategy(["A"], 1.0), Strategy)
        assert isinstance(_RandomStrategy(0), Strategy)


class TestExecutionLag:
    def test_first_bar_decision_fills_on_second_bar(self) -> None:
        u = _universe()
        start, end = _span(u)
        days = u.trading_days(start, end)
        ticker = u.members_as_of(start)[0]
        engine = BacktestEngine(u, _FirstBarStrategy(ticker), Portfolio(100_000))
        result = engine.run(start, end)
        assert len(result.fills) == 1
        # Decided on days[0], executed at the OPEN of days[1]: a one-bar lag, no lookahead.
        assert result.fills[0].date == days[1]


class TestInvariantEndToEnd:
    def test_equity_reconciles_from_fills(self) -> None:
        u = _universe()
        start, end = _span(u)
        engine = BacktestEngine(
            u,
            _RandomStrategy(7),
            Portfolio(1_000_000),
            execution=ExecutionSimulator(CostModel()),
        )
        result = engine.run(start, end)

        final_prices = u.close_row(end)
        expected = result.initial_cash
        for f in result.fills:
            expected += f.quantity * (final_prices[f.ticker] - f.price)
        expected -= result.total_costs

        assert result.equity_curve.iloc[-1] == pytest.approx(expected)

    def test_costs_reduce_equity_versus_frictionless(self) -> None:
        u = _universe()
        start, end = _span(u)
        free = CostModel(commission_per_share=0, commission_min=0, half_spread_bps=0, impact_eta=0)

        def run(cost_model: CostModel) -> float:
            engine = BacktestEngine(
                u,
                _RandomStrategy(3),
                Portfolio(1_000_000),
                execution=ExecutionSimulator(cost_model),
            )
            return float(engine.run(start, end).equity_curve.iloc[-1])

        # Same trades, only costs differ -> the costed run must end with less equity.
        assert run(CostModel()) < run(free)


class TestSharpe:
    def test_zero_for_constant_returns(self) -> None:
        assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0

    def test_positive_for_upward_drift(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.005, size=500))
        assert sharpe_ratio(rets) > 0.0

    def test_total_return_property(self) -> None:
        curve = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
        result = BacktestResult(curve, [], 100.0, {}, 0.0)
        assert result.total_return == pytest.approx(0.1)
