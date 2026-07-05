"""Tests for the synthetic data generators.

These are *property* tests, not fragile golden-value snapshots: we assert the statistical
invariants the generators are supposed to have (stationary variance, mean reversion,
cointegration ground truth).

Two design notes that make the statistical assertions robust rather than flaky:

* We fix the ADF lag (``maxlag`` instead of ``autolag="AIC"``). Autolag re-fits an OLS for
  every candidate lag and is the dominant cost on long series; a fixed small lag is both
  fast and sufficient for these processes.
* For *non-stationarity* claims we do **not** assert ``p > 0.05`` on the levels alone — a
  true random walk still spuriously rejects the unit root ~5% of the time at the 5% level.
  Instead we assert the process is a unit-root process the proper way: the **differences**
  are strongly stationary while the **levels** fail to reject at the 1% level. That
  characterisation is what "has a unit root" actually means and is seed-robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from statlab.data import (
    OUParams,
    simulate_cointegrated_pair,
    simulate_correlated_ou_panel,
    simulate_ou,
    simulate_random_walk,
)

# A fixed, small ADF lag keeps the tests fast and deterministic in cost.
_ADF_MAXLAG = 5


def _adf_pvalue(x: np.ndarray, maxlag: int = _ADF_MAXLAG) -> float:
    """ADF unit-root test p-value with a fixed lag (small p ⇒ stationary)."""
    return float(adfuller(x, maxlag=maxlag, autolag=None)[1])


def _assert_has_unit_root(x: np.ndarray) -> None:
    """Assert ``x`` behaves like a unit-root process: I(1), stationary in differences.

    Robust to finite-sample false positives because it requires BOTH that the levels fail
    to reject the unit root at the 1% level AND that the first differences strongly reject
    it. A stationary series would reject in levels too; a random walk will not.
    """
    p_levels = _adf_pvalue(x)
    p_diff = _adf_pvalue(np.diff(x))
    assert p_levels > 0.01, f"levels rejected unit root at 1% (p={p_levels:.4f})"
    assert p_diff < 0.01, f"differences did not reject unit root (p={p_diff:.4f})"


class TestOUParams:
    def test_rejects_nonpositive_theta(self) -> None:
        with pytest.raises(ValueError, match="theta"):
            OUParams(theta=0.0, mu=0.0, sigma=1.0)

    def test_rejects_nonpositive_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            OUParams(theta=0.1, mu=0.0, sigma=-1.0)

    def test_half_life_formula(self) -> None:
        params = OUParams(theta=np.log(2.0), mu=0.0, sigma=1.0)
        assert params.half_life == pytest.approx(1.0)

    def test_stationary_std_formula(self) -> None:
        params = OUParams(theta=0.5, mu=0.0, sigma=2.0)
        # sigma / sqrt(2 theta) = 2 / sqrt(1) = 2
        assert params.stationary_std == pytest.approx(2.0)


class TestSimulateOU:
    def test_shape_and_initial_value(self, rng: np.random.Generator) -> None:
        params = OUParams(theta=0.1, mu=5.0, sigma=1.0)
        path = simulate_ou(50, params, rng, x0=3.0)
        assert path.shape == (50,)
        assert path[0] == 3.0

    def test_defaults_x0_to_mu(self, rng: np.random.Generator) -> None:
        params = OUParams(theta=0.1, mu=7.5, sigma=1.0)
        path = simulate_ou(10, params, rng)
        assert path[0] == pytest.approx(7.5)

    def test_n_one_returns_single_point(self, rng: np.random.Generator) -> None:
        params = OUParams(theta=0.1, mu=0.0, sigma=1.0)
        path = simulate_ou(1, params, rng, x0=2.0)
        assert path.shape == (1,)
        assert path[0] == 2.0

    def test_rejects_nonpositive_n(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            simulate_ou(0, OUParams(theta=0.1, mu=0.0, sigma=1.0), rng)

    def test_stationary_variance_matches_theory(self, rng: np.random.Generator) -> None:
        # Start at the mean and simulate a long path; the sample std should approach the
        # theoretical stationary std sigma / sqrt(2 theta).
        params = OUParams(theta=0.05, mu=0.0, sigma=0.5)
        path = simulate_ou(50_000, params, rng)
        sample_std = float(np.std(path[1000:]))  # drop a burn-in
        assert sample_std == pytest.approx(params.stationary_std, rel=0.05)

    def test_is_stationary_by_adf(self, rng: np.random.Generator) -> None:
        params = OUParams(theta=0.1, mu=0.0, sigma=1.0)
        path = simulate_ou(2000, params, rng)
        assert _adf_pvalue(path) < 0.01  # reject unit root: the OU path is stationary

    def test_dt_consistency_of_stationary_variance(self) -> None:
        # The exact transition means the stationary variance is invariant to dt.
        params = OUParams(theta=0.2, mu=1.0, sigma=1.0)
        coarse = simulate_ou(40_000, params, np.random.default_rng(1), dt=1.0)
        fine = simulate_ou(40_000, params, np.random.default_rng(2), dt=0.25)
        assert float(np.std(coarse[500:])) == pytest.approx(float(np.std(fine[500:])), rel=0.06)

    def test_reproducible_with_same_seed(self) -> None:
        params = OUParams(theta=0.1, mu=0.0, sigma=1.0)
        a = simulate_ou(100, params, np.random.default_rng(42))
        b = simulate_ou(100, params, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestSimulateRandomWalk:
    def test_shape_and_start(self, rng: np.random.Generator) -> None:
        walk = simulate_random_walk(100, rng, x0=10.0)
        assert walk.shape == (100,)
        assert walk[0] == pytest.approx(10.0)

    def test_has_unit_root(self, rng: np.random.Generator) -> None:
        walk = simulate_random_walk(2000, rng)
        _assert_has_unit_root(walk)

    def test_drift_moves_the_mean(self, rng: np.random.Generator) -> None:
        walk = simulate_random_walk(5000, rng, sigma=0.1, drift=0.05)
        assert walk[-1] > walk[0]


class TestCointegratedPair:
    def test_columns_and_index(self, rng: np.random.Generator) -> None:
        frame = simulate_cointegrated_pair(500, rng, names=("X", "Y"))
        assert list(frame.columns) == ["X", "Y"]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert (frame > 0).all().all()  # prices are strictly positive

    def test_spread_is_stationary(self, rng: np.random.Generator) -> None:
        beta = 1.3
        frame = simulate_cointegrated_pair(2000, rng, beta=beta)
        # The KNOWN cointegrating combination log(B) - beta log(A) must be stationary.
        spread = np.log(frame["B"].to_numpy()) - beta * np.log(frame["A"].to_numpy())
        assert _adf_pvalue(spread) < 0.05

    def test_wrong_hedge_ratio_has_unit_root(self, rng: np.random.Generator) -> None:
        # A grossly wrong hedge ratio leaves the common random-walk trend uncancelled,
        # so the residual should itself carry a unit root.
        frame = simulate_cointegrated_pair(2000, rng, beta=1.0)
        bad = np.log(frame["B"].to_numpy()) - 5.0 * np.log(frame["A"].to_numpy())
        _assert_has_unit_root(bad)


class TestCorrelatedPanel:
    def test_ground_truth_pairs_are_cointegrated(self, rng: np.random.Generator) -> None:
        panel, truth = simulate_correlated_ou_panel(2000, rng, n_pairs=2, n_noise=2)
        assert len(truth) == 2
        for a, b in truth:
            resid = _ols_residual(np.log(panel[a]), np.log(panel[b]))
            assert _adf_pvalue(resid) < 0.10

    def test_noise_columns_have_unit_roots(self, rng: np.random.Generator) -> None:
        panel, _ = simulate_correlated_ou_panel(2000, rng, n_pairs=1, n_noise=2)
        for col in ["N0", "N1"]:
            _assert_has_unit_root(panel[col].to_numpy())

    def test_panel_width(self, rng: np.random.Generator) -> None:
        panel, truth = simulate_correlated_ou_panel(300, rng, n_pairs=3, n_noise=4)
        assert panel.shape[1] == 3 * 2 + 4
        assert len(truth) == 3


def _ols_residual(y: pd.Series, x: pd.Series) -> np.ndarray:
    """Residual of an OLS regression y ~ const + x, used as a cointegration proxy."""
    x_mat = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(x_mat, y.to_numpy(), rcond=None)
    resid: np.ndarray = y.to_numpy() - x_mat @ coef
    return resid
