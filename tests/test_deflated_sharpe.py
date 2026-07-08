"""Tests for the Probabilistic and Deflated Sharpe Ratio.

This formula is easy to get subtly wrong from memory, so it's checked against two
independent anchors rather than trusted by inspection: an exact algebraic special case
(Lo (2002)'s classical Gaussian-returns Sharpe-ratio variance) and a Monte Carlo simulation
of the N-trial expected-maximum-under-the-null term, plus property tests (more trials
searched raises the bar; a genuinely skillful single track record scores high; an unskilled
one scores low regardless of N).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from statlab.validation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


class TestGaussianSpecialCase:
    """With skew=0 and kurtosis=3 (Gaussian returns), PSR's variance-shape term
    ``1 - skew*SR + (kurtosis-1)/4*SR^2`` must reduce to Lo (2002)'s classical asymptotic
    Sharpe-ratio variance shape ``1 + SR^2/2`` — an exact algebraic identity, checked here
    by computing Lo's z-score independently (not by calling this module) and comparing.
    """

    @pytest.mark.parametrize("sr_hat", [-1.5, -0.3, 0.0, 0.2, 0.75, 1.5, 3.0])
    def test_matches_los_classical_z_score(self, sr_hat: float) -> None:
        n = 252
        sr_benchmark = 0.0

        # Lo's formula, written out independently: Var(SR_hat) ~= (1 + SR_hat^2/2) / n.
        lo_variance = (1.0 + sr_hat**2 / 2.0) / n
        lo_z = (sr_hat - sr_benchmark) / math.sqrt(lo_variance)
        lo_psr = stats.norm.cdf(lo_z * math.sqrt((n - 1) / n))  # align the n vs n-1 convention

        psr = probabilistic_sharpe_ratio(sr_hat, sr_benchmark, n, skew=0.0, kurtosis=3.0)
        assert psr == pytest.approx(lo_psr, abs=1e-9)


class TestMonteCarloExpectedMax:
    """Simulate N independent pure-noise (true SR=0) return series many times, take the max
    observed Sharpe per replication, and confirm the mean of expected_max_sharpe's own
    predictions (one per replication, using that replication's empirical trial variance)
    tracks the mean of the actually-observed maxima — validating the extreme-value
    approximation itself, not just this module's transcription of it.
    """

    def test_mean_prediction_tracks_mean_observed_maximum(self) -> None:
        rng = np.random.default_rng(2024)
        n_obs = 100
        n_trials = 20
        n_reps = 1500

        actual_maxes = np.empty(n_reps)
        predicted = np.empty(n_reps)
        for i in range(n_reps):
            trial_sharpes = [
                rng.standard_normal(n_obs).mean() / rng.standard_normal(n_obs).std(ddof=1)
                for _ in range(n_trials)
            ]
            actual_maxes[i] = max(trial_sharpes)
            predicted[i] = expected_max_sharpe(trial_sharpes)

        mean_actual = float(np.mean(actual_maxes))
        mean_predicted = float(np.mean(predicted))
        assert mean_predicted == pytest.approx(mean_actual, rel=0.15)


class TestExpectedMaxSharpeMonotonicity:
    def test_increases_with_n_at_fixed_variance(self) -> None:
        # A +-1 repeated pattern has population variance = 1 regardless of length, isolating
        # the N-trials effect from a changing spread.
        ns = [4, 10, 20, 50, 100, 200]
        values = []
        for n in ns:
            trials = ([1.0, -1.0] * (n // 2 + 1))[:n]
            values.append(expected_max_sharpe(trials))
        assert values == sorted(values)
        assert values[-1] > values[0]

    def test_zero_for_a_single_trial(self) -> None:
        assert expected_max_sharpe([1.23]) == 0.0

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            expected_max_sharpe([])

    def test_zero_when_all_trials_are_identical(self) -> None:
        assert expected_max_sharpe([0.5, 0.5, 0.5]) == 0.0


class TestDeflatedSharpeProperties:
    def test_decreases_as_more_trials_are_searched(self) -> None:
        # Fixed best trial (1.5) plus a roughly-fixed-variance tail of "other" trials: as N
        # grows, the luck-adjusted bar (sr_0) rises and DSR must fall (or saturate at 0).
        ns = [3, 5, 11, 21, 51]
        dsrs = []
        for n in ns:
            trials = [1.5, *([1.0, -1.0] * (n // 2 + 1))[: n - 1]]
            dsrs.append(deflated_sharpe_ratio(trials, n_default=252).dsr)
        assert dsrs == sorted(dsrs, reverse=True)
        assert dsrs[0] > dsrs[-1]

    def test_single_skillful_trial_scores_high(self) -> None:
        result = deflated_sharpe_ratio([3.0], n_default=252)
        assert result.sr_0 == 0.0  # no multiple-comparisons penalty possible at N=1
        assert result.dsr > 0.99

    def test_unskilled_result_scores_low_regardless_of_n(self) -> None:
        for n_trials in (1, 5, 20, 100):
            trials = [0.01] * n_trials
            result = deflated_sharpe_ratio(trials, n_default=252)
            assert result.dsr < 0.6

    def test_rejects_empty_trials(self) -> None:
        with pytest.raises(ValueError, match="at least one trial"):
            deflated_sharpe_ratio([])

    def test_best_returns_supplies_its_own_skew_kurtosis_and_n(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.standard_normal(500) * 0.01 + 0.001
        result_with_returns = deflated_sharpe_ratio([0.5, 0.2, 0.1], best_returns=returns)
        result_without = deflated_sharpe_ratio([0.5, 0.2, 0.1], n_default=len(returns))
        # Near-Gaussian synthetic returns should give a very similar (not necessarily
        # identical, since sample skew/kurtosis won't be exactly 0/3) result to the default.
        assert result_with_returns.dsr == pytest.approx(result_without.dsr, abs=0.05)


class TestProbabilisticSharpeRatio:
    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            probabilistic_sharpe_ratio(1.0, 0.0, 1)

    def test_equals_half_when_observed_matches_benchmark(self) -> None:
        # SR_hat == SR* -> z == 0 -> Phi(0) == 0.5, regardless of skew/kurtosis/n.
        assert probabilistic_sharpe_ratio(0.5, 0.5, 100) == pytest.approx(0.5)

    def test_increases_with_observed_sharpe(self) -> None:
        low = probabilistic_sharpe_ratio(0.1, 0.0, 252)
        high = probabilistic_sharpe_ratio(1.0, 0.0, 252)
        assert high > low
