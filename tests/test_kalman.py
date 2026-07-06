"""Tests for the hand-rolled Kalman hedge-ratio filter.

Validated against closed-form references: in the constant-coefficient limit the filter is
recursive least squares and must converge to OLS; with a drifting coefficient it must
track the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from statlab.data import simulate_random_walk
from statlab.signals import kalman_hedge
from statlab.signals.kalman import KalmanResult


def _ols_beta_alpha(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1]), float(coef[0])


class TestKalmanStatic:
    def test_recovers_constant_beta(self, rng: np.random.Generator) -> None:
        x = simulate_random_walk(3000, rng, x0=5.0, sigma=0.05)
        y = 1.5 + 2.0 * x + rng.normal(0.0, 0.01, size=len(x))
        res = kalman_hedge(y, x, delta=1e-6, obs_var=1e-2, p0=10.0)
        assert res.beta[-1] == pytest.approx(2.0, abs=0.05)
        assert res.alpha[-1] == pytest.approx(1.5, abs=0.2)

    def test_matches_ols_in_static_limit(self, rng: np.random.Generator) -> None:
        # delta=0 makes the state constant -> the filter is RLS -> converges to OLS.
        x = simulate_random_walk(4000, rng, x0=3.0, sigma=0.05)
        y = -0.7 + 1.3 * x + rng.normal(0.0, 0.02, size=len(x))
        res = kalman_hedge(y, x, delta=0.0, obs_var=1e-2, p0=100.0)
        ols_beta, ols_alpha = _ols_beta_alpha(y, x)
        assert res.beta[-1] == pytest.approx(ols_beta, rel=0.02)
        assert res.alpha[-1] == pytest.approx(ols_alpha, abs=0.05)


class TestKalmanDynamic:
    def test_tracks_drifting_beta(self, rng: np.random.Generator) -> None:
        n = 3000
        x = simulate_random_walk(n, rng, x0=5.0, sigma=0.05)
        true_beta = np.linspace(1.0, 2.0, n)  # slow linear drift
        y = 0.5 + true_beta * x + rng.normal(0.0, 0.01, size=n)
        res = kalman_hedge(y, x, delta=1e-3, obs_var=1e-3, p0=10.0)

        warm = n // 5
        corr = np.corrcoef(res.beta[warm:], true_beta[warm:])[0, 1]
        assert corr > 0.9
        assert res.beta[-1] == pytest.approx(2.0, abs=0.15)


class TestKalmanOutputs:
    def test_shapes_and_zscore(self, rng: np.random.Generator) -> None:
        x = simulate_random_walk(500, rng, x0=5.0, sigma=0.05)
        y = 1.0 + x + rng.normal(0.0, 0.01, size=len(x))
        res = kalman_hedge(y, x)
        assert isinstance(res, KalmanResult)
        for arr in (res.beta, res.alpha, res.spread, res.innovation_std):
            assert arr.shape == (500,)
        z = res.zscore()
        assert z.shape == (500,)
        # After warm-up the standardised spread should be finite and roughly unit-scale.
        assert np.isfinite(z[50:]).all()
        assert abs(np.std(z[100:])) < 5.0

    def test_reproducible(self, rng: np.random.Generator) -> None:
        x = simulate_random_walk(300, rng, x0=5.0, sigma=0.05)
        y = 1.0 + 1.2 * x
        a = kalman_hedge(y, x, delta=1e-4)
        b = kalman_hedge(y, x, delta=1e-4)
        np.testing.assert_array_equal(a.beta, b.beta)


class TestKalmanValidation:
    def test_rejects_bad_delta(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            kalman_hedge(np.ones(10), np.ones(10), delta=1.0)

    def test_rejects_bad_obs_var(self) -> None:
        with pytest.raises(ValueError, match="obs_var must be positive"):
            kalman_hedge(np.ones(10), np.ones(10), obs_var=0.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            kalman_hedge(np.ones(10), np.ones(9))
