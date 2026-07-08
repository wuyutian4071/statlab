r"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado).

Two problems compound when reporting the Sharpe ratio of the best-performing configuration
found in a search (a sensitivity grid, a set of discovered pairs, ...):

1. **Non-normality.** The Sharpe ratio estimator's own sampling variance depends on the
   skewness and kurtosis of the underlying returns, not just their mean/variance — a plain
   Sharpe ratio implicitly assumes Gaussian returns and understates uncertainty for anything
   else. The Probabilistic Sharpe Ratio (PSR) corrects for this (Bailey & Lopez de Prado,
   "The Sharpe Ratio Efficient Frontier", 2012; the variance term is Mertens (2002) /
   Christie (2005)'s extension of Lo (2002)'s classical Gaussian-case formula).
2. **Selection bias.** Reporting the *best* of N trials as if it were the only trial run
   overstates significance — some trials look good by pure luck, and the more you try, the
   likelier one looks good even with zero true skill. The Deflated Sharpe Ratio (DSR)
   raises the bar the reported Sharpe must clear to account for how many trials were
   searched (Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014).

Formulas this easy to get subtly wrong from memory shouldn't be trusted by inspection —
``tests/test_deflated_sharpe.py`` checks this implementation against two independent
anchors: Lo (2002)'s classical closed-form Sharpe-ratio variance for Gaussian returns (an
exact algebraic special case this must reduce to), and a Monte Carlo simulation of the
N-trial expected-maximum-under-the-null term (accurate for the N >= ~10-20 the asymptotic
extreme-value approximation is meant for; smaller N is a known-limitation regime).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

#: Euler-Mascheroni constant, used in the expected-maximum-of-N-Gaussians approximation.
EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    n: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    r"""PSR: the probability the *true* Sharpe ratio exceeds ``sr_benchmark``, given an
    observed per-period Sharpe ``sr_hat`` from ``n`` observations with sample ``skew`` and
    ``kurtosis`` (non-excess; ``3.0`` for Gaussian returns, the default).

    ``PSR = Phi[ (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - skew*SR_hat + (kurtosis-1)/4*SR_hat^2) ]``
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    variance_term = 1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2
    if variance_term <= 0.0:
        # Degenerate (pathologically skewed/kurtotic) input; treat as maximal uncertainty
        # rather than raising or dividing by a non-positive number.
        return 0.5
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    r"""The expected maximum Sharpe ratio achievable by pure luck across
    ``len(trial_sharpes)`` independent trials under the null of zero true skill, from the
    extreme-value approximation for the max of N Gaussians:

    ``SR_0 ~= sqrt(V[SR_n]) * [ (1-gamma_e)*Phi^-1(1 - 1/N) + gamma_e*Phi^-1(1 - 1/(N*e)) ]``

    ``V[SR_n]`` is the empirical variance across the trial Sharpe ratios themselves —
    typically the output of a sensitivity grid or a set of discovered-pair backtests. This
    asymptotic approximation is accurate for moderate-to-large N (roughly N >= 10-20); it
    systematically understates the true expected maximum for very small N, a known
    limitation of the approximation itself, not a calibration knob.
    """
    n_trials = len(trial_sharpes)
    if n_trials < 1:
        raise ValueError("need at least one trial")
    if n_trials == 1:
        return 0.0  # no multiple-comparisons correction possible (or needed) with one trial
    variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    if variance <= 0.0:
        return 0.0
    std = math.sqrt(variance)
    term1 = (1.0 - EULER_MASCHERONI) * float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    term2 = EULER_MASCHERONI * float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return std * (term1 + term2)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """The deflated Sharpe ratio plus the intermediate numbers that produced it."""

    dsr: float
    sr_0: float
    best_sharpe: float
    n_trials: int


def deflated_sharpe_ratio(
    trial_sharpes: Sequence[float],
    best_returns: np.ndarray | None = None,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_default: int = 252,
) -> DeflatedSharpeResult:
    """The Deflated Sharpe Ratio: the PSR of the best-of-N observed Sharpe ratio, evaluated
    against the expected maximum achievable by luck alone across those N trials.

    ``trial_sharpes`` should be per-period (not annualized) Sharpe ratios from N candidate
    configurations (e.g. a :func:`~statlab.validation.sensitivity.sensitivity_grid`'s
    output); the largest is taken as the reported result. ``best_returns``, if given,
    supplies the actual per-period return series of the best trial so its own sample
    skew/kurtosis and observation count are used instead of the ``skew``/``kurtosis``
    defaults (Gaussian) and ``n_default``.
    """
    if not trial_sharpes:
        raise ValueError("need at least one trial")

    best_sharpe = max(trial_sharpes)
    sr_0 = expected_max_sharpe(trial_sharpes)

    n = n_default
    if best_returns is not None and len(best_returns) >= 2:
        n = len(best_returns)
        skew = float(stats.skew(best_returns))
        kurtosis = float(stats.kurtosis(best_returns, fisher=False))

    dsr = probabilistic_sharpe_ratio(best_sharpe, sr_0, n, skew=skew, kurtosis=kurtosis)
    return DeflatedSharpeResult(
        dsr=dsr, sr_0=sr_0, best_sharpe=best_sharpe, n_trials=len(trial_sharpes)
    )
