"""Tests for walk-forward validation: window generation, the no-leakage guarantee (pair
selection must never see test-window dates), and the combined out-of-sample Sharpe.
"""

from __future__ import annotations

from itertools import pairwise
from typing import cast

import numpy as np
import pandas as pd
import pytest

from statlab.backtest import BacktestResult
from statlab.data import PointInTimeUniverse, SyntheticSource
from statlab.signals import PairCandidate
from statlab.signals import discover_pairs as _real_discover_pairs
from statlab.validation import walkforward
from statlab.validation.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    combined_oos_sharpe,
    run_walk_forward,
    walk_forward_windows,
)


def _universe(n: int = 1200, seed: int = 17) -> PointInTimeUniverse:
    return PointInTimeUniverse.from_bars(
        SyntheticSource(n=n, n_pairs=3, n_noise=3, seed=seed).fetch()
    )


class TestWalkForwardWindows:
    def test_rejects_nonpositive_lengths(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=100)
        with pytest.raises(ValueError, match="train_days and test_days"):
            walk_forward_windows(days, train_days=0, test_days=10)
        with pytest.raises(ValueError, match="train_days and test_days"):
            walk_forward_windows(days, train_days=10, test_days=0)

    def test_no_window_when_not_enough_days(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=50)
        assert walk_forward_windows(days, train_days=40, test_days=40) == []

    def test_test_windows_are_contiguous_and_non_overlapping(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=1000)
        windows = walk_forward_windows(days, train_days=200, test_days=100)
        assert len(windows) > 1
        for a, b in pairwise(windows):
            assert a.test_end < b.test_start
            # step_days defaults to test_days: no gap between consecutive test windows.
            assert days[cast(int, days.get_loc(a.test_end)) + 1] == b.test_start

    def test_train_immediately_precedes_test(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=1000)
        for w in walk_forward_windows(days, train_days=200, test_days=100):
            assert w.train_start < w.train_end < w.test_start < w.test_end
            assert days[cast(int, days.get_loc(w.train_end)) + 1] == w.test_start

    def test_custom_step_days_controls_train_overlap(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=1000)
        windows = walk_forward_windows(days, train_days=200, test_days=50, step_days=200)
        # With step == train_days, consecutive train windows don't overlap at all.
        for a, b in pairwise(windows):
            assert a.train_end < b.train_start


class TestNoLeakageAcrossTheWalkForwardBoundary:
    def test_discovery_never_sees_dates_outside_its_train_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decisive test: spy on discover_pairs and assert every price panel it's ever
        called with is confined to that window's [train_start, train_end] — mirroring the
        rigor of test_no_lookahead.py for this milestone's own causality guarantee.
        """
        seen_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def spy(
            prices: pd.DataFrame,
            *,
            min_correlation: float = 0.7,
            max_pvalue: float = 0.05,
            min_half_life: float = 1.0,
            max_half_life: float = 252.0,
        ) -> list[PairCandidate]:
            seen_ranges.append((prices.index.min(), prices.index.max()))
            return _real_discover_pairs(
                prices,
                min_correlation=min_correlation,
                max_pvalue=max_pvalue,
                min_half_life=min_half_life,
                max_half_life=max_half_life,
            )

        monkeypatch.setattr(walkforward, "discover_pairs", spy)

        u = _universe()
        days = u.trading_days("1900-01-01", "2100-01-01")
        windows = walk_forward_windows(days, train_days=200, test_days=100)
        run_walk_forward(u, windows, min_correlation=0.3, max_pvalue=0.1)

        assert len(seen_ranges) == len(windows)
        for (lo, hi), window in zip(seen_ranges, windows, strict=True):
            assert lo >= window.train_start
            assert hi <= window.train_end
            assert hi < window.test_start  # never touches the test window at all


class TestRunWalkForward:
    def test_every_result_pairs_with_its_window(self) -> None:
        u = _universe()
        days = u.trading_days("1900-01-01", "2100-01-01")
        windows = walk_forward_windows(days, train_days=200, test_days=100)
        results = run_walk_forward(u, windows, min_correlation=0.3, max_pvalue=0.1)
        assert [r.window for r in results] == windows

    def test_a_result_without_a_pair_has_no_backtest_result(self) -> None:
        u = _universe()
        days = u.trading_days("1900-01-01", "2100-01-01")
        windows = walk_forward_windows(days, train_days=200, test_days=100)
        # Impossibly strict thresholds -> no window should ever find a pair.
        results = run_walk_forward(u, windows, min_correlation=0.999, max_pvalue=1e-9)
        assert all(r.pair is None and r.result is None for r in results)

    def test_backtest_result_covers_only_the_test_window(self) -> None:
        u = _universe()
        days = u.trading_days("1900-01-01", "2100-01-01")
        windows = walk_forward_windows(days, train_days=200, test_days=100)
        results = run_walk_forward(u, windows, min_correlation=0.3, max_pvalue=0.1)
        for r in results:
            if r.result is None:
                continue
            assert r.result.equity_curve.index.min() >= r.window.test_start
            assert r.result.equity_curve.index.max() <= r.window.test_end


class TestCombinedOosSharpe:
    def test_matches_manual_concatenation(self) -> None:
        idx1 = pd.date_range("2020-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2020-02-01", periods=5, freq="D")
        curve1 = pd.Series([100.0, 101.0, 102.0, 101.5, 103.0], index=idx1, name="equity")
        curve2 = pd.Series([100.0, 99.0, 98.5, 100.0, 101.0], index=idx2, name="equity")

        r1 = BacktestResult(curve1, [], 100.0, {}, 0.0)
        r2 = BacktestResult(curve2, [], 100.0, {}, 0.0)
        window = WalkForwardWindow(
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-06-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-02-05"),
        )
        results = [
            WalkForwardResult(window=window, pair=None, result=r1),
            WalkForwardResult(window=window, pair=None, result=r2),
        ]

        from statlab.backtest import sharpe_ratio

        combined_returns = pd.concat([curve1.pct_change().dropna(), curve2.pct_change().dropna()])
        expected = sharpe_ratio(combined_returns)
        assert combined_oos_sharpe(results) == pytest.approx(expected)

    def test_zero_when_no_window_ever_traded(self) -> None:
        window = WalkForwardWindow(
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-06-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-02-05"),
        )
        results = [WalkForwardResult(window=window, pair=None, result=None)]
        assert combined_oos_sharpe(results) == 0.0


class TestDiscoveryPower:
    """Reuses M3's ground-truth-labelled synthetic panel to check walk-forward discovery
    tends to actually find the real cointegrated pairs and not just noise — the same power
    check discovery's own tests make, now through the walk-forward harness."""

    def test_selects_a_true_pair_more_often_than_chance(self, rng: np.random.Generator) -> None:
        from statlab.data.synthetic import simulate_correlated_ou_panel

        panel, truth = simulate_correlated_ou_panel(1500, rng, n_pairs=3, n_noise=5)
        truth_set = {frozenset(pair) for pair in truth}

        u = PointInTimeUniverse(panel)
        days = u.trading_days("1900-01-01", "2100-01-01")
        windows = walk_forward_windows(days, train_days=250, test_days=125)
        results = run_walk_forward(u, windows, min_correlation=0.3, max_pvalue=0.1)

        chosen_pairs = [r.pair for r in results if r.pair is not None]
        assert chosen_pairs  # the harness should find *something* across this many windows

        true_hits = sum(1 for p in chosen_pairs if frozenset((p.y, p.x)) in truth_set)
        # With 3 true pairs out of 8 possible names (28 possible pairs), a pair chosen at
        # random would rarely be a true one; discovery should do much better than that.
        assert true_hits / len(chosen_pairs) > 0.3
