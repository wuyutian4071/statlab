"""Tests for portfolio accounting and the equity invariant.

The invariant under test: **equity changes only through market moves on held positions and
through transaction costs.** Trading at the fill price is otherwise equity-neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statlab.backtest import Fill, Portfolio


def _fill(ticker: str, qty: float, price: float, cost: float = 0.0) -> Fill:
    return Fill(pd.Timestamp("2020-01-02"), ticker, qty, price, cost)


class TestPortfolioBasics:
    def test_initial_equity_is_cash(self) -> None:
        p = Portfolio(100_000)
        assert p.equity({}) == 100_000

    def test_rejects_nonpositive_initial_cash(self) -> None:
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            Portfolio(0)

    def test_buy_moves_cash_and_position(self) -> None:
        p = Portfolio(100_000)
        p.apply_fill(_fill("AAA", 100, 50.0))
        assert p.cash == pytest.approx(100_000 - 100 * 50.0)
        assert p.positions["AAA"] == 100

    def test_closing_position_removes_it(self) -> None:
        p = Portfolio(100_000)
        p.apply_fill(_fill("AAA", 100, 50.0))
        p.apply_fill(_fill("AAA", -100, 55.0))
        assert "AAA" not in p.positions

    def test_short_position_is_supported(self) -> None:
        p = Portfolio(100_000)
        p.apply_fill(_fill("AAA", -100, 50.0))  # open a short
        assert p.positions["AAA"] == -100
        assert p.cash == pytest.approx(100_000 + 100 * 50.0)  # short sale adds cash


class TestEquityInvariant:
    def test_trading_at_fill_price_drops_equity_by_exactly_cost(self) -> None:
        p = Portfolio(100_000)
        price = 50.0
        before = p.equity({"AAA": price})
        p.apply_fill(_fill("AAA", 100, price, cost=7.5))
        after = p.equity({"AAA": price})  # marked at the same fill price
        assert before - after == pytest.approx(7.5)

    def test_equity_moves_only_by_mark_to_market_between_trades(self) -> None:
        p = Portfolio(100_000)
        p.apply_fill(_fill("AAA", 100, 50.0, cost=1.0))
        eq0 = p.equity({"AAA": 50.0})
        eq1 = p.equity({"AAA": 53.0})  # no trade, price +3
        assert eq1 - eq0 == pytest.approx(100 * 3.0)

    def test_short_pnl_sign(self) -> None:
        p = Portfolio(100_000)
        p.apply_fill(_fill("AAA", -100, 50.0))
        eq_up = p.equity({"AAA": 52.0})  # price up hurts a short
        eq_dn = p.equity({"AAA": 48.0})  # price down helps a short
        assert eq_dn > eq_up

    def test_full_reconciliation_over_a_trade_sequence(self, rng: np.random.Generator) -> None:
        # Independently reconstruct final equity from first principles and compare.
        p = Portfolio(1_000_000)
        fills = []
        for _ in range(200):
            tkr = rng.choice(["AAA", "BBB", "CCC"])
            qty = float(rng.integers(-100, 100))
            price = float(rng.uniform(20, 200))
            cost = float(rng.uniform(0, 5))
            f = _fill(str(tkr), qty, price, cost)
            fills.append(f)
            p.apply_fill(f)

        final_prices = {"AAA": 111.0, "BBB": 77.0, "CCC": 143.0}
        engine_equity = p.equity(final_prices)

        # equity = initial + sum over fills of qty*(final_price - fill_price) - total costs
        expected = 1_000_000.0
        for f in fills:
            expected += f.quantity * (final_prices[f.ticker] - f.price)
        expected -= sum(f.cost for f in fills)
        assert engine_equity == pytest.approx(expected)


class TestEquityCurve:
    def test_curve_records_marks(self) -> None:
        p = Portfolio(100_000)
        p.mark(pd.Timestamp("2020-01-02"), {})
        p.apply_fill(_fill("AAA", 100, 50.0, cost=1.0))
        p.mark(pd.Timestamp("2020-01-03"), {"AAA": 52.0})
        curve = p.equity_curve
        assert len(curve) == 2
        assert curve.iloc[0] == pytest.approx(100_000)
        assert curve.iloc[1] == pytest.approx(100_000 + 100 * 2.0 - 1.0)

    def test_empty_curve(self) -> None:
        assert Portfolio(100_000).equity_curve.empty
