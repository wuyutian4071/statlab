"""Tests for the transaction-cost model, component by component."""

from __future__ import annotations

import math

import pytest

from statlab.backtest import CostModel


class TestCostModel:
    def test_zero_quantity_is_free(self) -> None:
        assert CostModel().cost(0.0, 100.0) == 0.0

    def test_commission_minimum_applies_for_small_orders(self) -> None:
        model = CostModel(
            commission_per_share=0.01, commission_min=1.0, half_spread_bps=0.0, impact_eta=0.0
        )
        # 10 shares * 0.01 = 0.10 < 1.0 minimum -> charged 1.0
        assert model.cost(10, 100.0) == pytest.approx(1.0)

    def test_commission_per_share_dominates_for_large_orders(self) -> None:
        model = CostModel(
            commission_per_share=0.01, commission_min=1.0, half_spread_bps=0.0, impact_eta=0.0
        )
        # 1000 shares * 0.01 = 10.0 > 1.0 minimum
        assert model.cost(1000, 100.0) == pytest.approx(10.0)

    def test_half_spread_is_bps_of_notional(self) -> None:
        model = CostModel(
            commission_per_share=0.0, commission_min=0.0, half_spread_bps=5.0, impact_eta=0.0
        )
        # 5 bps of (100 price * 200 shares) = 5e-4 * 20000 = 10.0
        assert model.cost(200, 100.0) == pytest.approx(10.0)

    def test_market_impact_follows_square_root_law(self) -> None:
        model = CostModel(
            commission_per_share=0.0, commission_min=0.0, half_spread_bps=0.0, impact_eta=0.1
        )
        q, price, vol, adv = 1000.0, 50.0, 0.02, 1_000_000.0
        expected = 0.1 * vol * math.sqrt(q / adv) * price * q
        assert model.cost(q, price, vol, adv) == pytest.approx(expected)

    def test_impact_quadruples_when_quantity_quadruples_scaled(self) -> None:
        # Impact dollars scale like Q^{1.5}; doubling Q multiplies impact by 2^1.5.
        model = CostModel(
            commission_per_share=0.0, commission_min=0.0, half_spread_bps=0.0, impact_eta=0.1
        )
        c1 = model.cost(1000, 50.0, 0.02, 1e6)
        c2 = model.cost(2000, 50.0, 0.02, 1e6)
        assert c2 / c1 == pytest.approx(2.0**1.5, rel=1e-9)

    def test_impact_zero_without_vol_or_adv(self) -> None:
        model = CostModel(
            commission_per_share=0.0, commission_min=0.0, half_spread_bps=0.0, impact_eta=0.1
        )
        assert model.cost(1000, 50.0, None, None) == 0.0
        assert model.cost(1000, 50.0, 0.02, 0.0) == 0.0

    def test_cost_is_sign_symmetric(self) -> None:
        model = CostModel()
        assert model.cost(100, 50.0, 0.02, 1e6) == model.cost(-100, 50.0, 0.02, 1e6)

    def test_total_is_sum_of_components(self) -> None:
        model = CostModel(
            commission_per_share=0.01, commission_min=1.0, half_spread_bps=2.0, impact_eta=0.1
        )
        q, p, vol, adv = 1000.0, 100.0, 0.02, 1e6
        commission = max(1.0, 0.01 * q)
        half_spread = 2.0 * 1e-4 * p * q
        impact = 0.1 * vol * math.sqrt(q / adv) * p * q
        assert model.cost(q, p, vol, adv) == pytest.approx(commission + half_spread + impact)

    @pytest.mark.parametrize("bad", ["commission_per_share", "commission_min", "impact_eta"])
    def test_rejects_negative_params(self, bad: str) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostModel(**{bad: -1.0})
