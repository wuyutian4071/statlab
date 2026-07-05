"""Synthetic market-data generation.

Everything in the test suite and the offline demo runs on data produced here, so the
project never *requires* a network download. The generators are deliberately simple but
statistically meaningful:

* :func:`simulate_ou` draws an Ornstein-Uhlenbeck (OU) mean-reverting process.
* :func:`simulate_random_walk` draws an (arithmetic) Brownian motion / random walk.
* :func:`simulate_cointegrated_pair` builds *two* price series that are genuinely
  cointegrated: they share a common stochastic trend and their spread is a stationary OU
  process. This is the canonical object a pairs-trading strategy is supposed to find.
* :func:`simulate_correlated_ou_panel` builds a wide panel mixing cointegrated pairs with
  independent random walks, i.e. a universe with a *known* ground truth of which pairs are
  tradable.

Reproducibility rule for the whole project: **callers pass in a ``numpy.random.Generator``
seeded upstream.** No function here ever touches global RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "OUParams",
    "simulate_cointegrated_pair",
    "simulate_correlated_ou_panel",
    "simulate_ou",
    "simulate_random_walk",
]

# A single fixed "start of history" so synthetic frames get a realistic business-day index.
_DEFAULT_START = "2015-01-02"


@dataclass(frozen=True)
class OUParams:
    r"""Parameters of an Ornstein-Uhlenbeck process.

    The OU process solves the stochastic differential equation

    .. math::

        dX_t = \theta (\mu - X_t)\, dt + \sigma\, dW_t,

    where :math:`\theta > 0` is the speed of mean reversion, :math:`\mu` the long-run
    mean, and :math:`\sigma > 0` the instantaneous volatility. Its exact discrete-time
    transition over a step of length ``dt`` is Gaussian:

    .. math::

        X_{t+dt} \mid X_t \sim \mathcal{N}\!\left(
            \mu + (X_t - \mu) e^{-\theta\, dt},\;
            \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\, dt}\right)
        \right).

    We simulate with this *exact* transition (not an Euler approximation) so the sampled
    path has the correct stationary variance :math:`\sigma^2 / (2\theta)` and
    autocorrelation regardless of ``dt``.

    The theoretical half-life of a shock is :math:`\ln(2)/\theta` in time units.
    """

    theta: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def half_life(self) -> float:
        """Theoretical half-life of a shock, ``ln(2) / theta`` (in time units of ``dt``)."""
        return float(np.log(2.0) / self.theta)

    @property
    def stationary_std(self) -> float:
        """Standard deviation of the stationary distribution, ``sigma / sqrt(2 theta)``."""
        return float(self.sigma / np.sqrt(2.0 * self.theta))


def simulate_ou(
    n: int,
    params: OUParams,
    rng: np.random.Generator,
    *,
    x0: float | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Simulate an OU path of length ``n`` using the exact Gaussian transition.

    Parameters
    ----------
    n:
        Number of samples to return (including the initial point).
    params:
        Mean-reversion parameters.
    rng:
        A seeded ``numpy.random.Generator``.
    x0:
        Initial value. Defaults to the long-run mean ``params.mu``. Pass a draw from the
        stationary distribution if you want the path to be stationary from ``t=0``.
    dt:
        Time-step size. The exact transition makes results ``dt``-consistent.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` float64 array.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    start = params.mu if x0 is None else x0
    path = np.empty(n, dtype=np.float64)
    path[0] = start
    if n == 1:
        return path

    decay = np.exp(-params.theta * dt)
    # Conditional standard deviation of the exact transition.
    cond_std = params.sigma * np.sqrt((1.0 - decay**2) / (2.0 * params.theta))
    shocks = rng.standard_normal(n - 1) * cond_std

    for t in range(1, n):
        path[t] = params.mu + (path[t - 1] - params.mu) * decay + shocks[t - 1]
    return path


def simulate_random_walk(
    n: int,
    rng: np.random.Generator,
    *,
    sigma: float = 1.0,
    x0: float = 0.0,
    drift: float = 0.0,
) -> np.ndarray:
    r"""Simulate an arithmetic random walk ``X_t = X_{t-1} + drift + sigma * Z_t``.

    This is a unit-root (non-stationary) process — the natural *negative control* for
    cointegration tests. Two independent random walks are, with probability one, **not**
    cointegrated.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    increments = rng.standard_normal(n) * sigma + drift
    increments[0] = 0.0
    return x0 + np.cumsum(increments)


def _business_day_index(n: int, start: str = _DEFAULT_START) -> pd.DatetimeIndex:
    """Return ``n`` consecutive business days starting at ``start``."""
    return pd.bdate_range(start=start, periods=n, name="date")


def simulate_cointegrated_pair(
    n: int,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
    alpha: float = 0.0,
    common_sigma: float = 1.0,
    spread_params: OUParams | None = None,
    p0: float = 100.0,
    start: str = _DEFAULT_START,
    names: tuple[str, str] = ("A", "B"),
) -> pd.DataFrame:
    r"""Simulate two **cointegrated** price series with a known hedge ratio.

    Construction
    ------------
    Let :math:`f_t` be a common stochastic trend (a random walk), and :math:`s_t` a
    stationary OU spread. The two *log* prices are

    .. math::

        \log P^{A}_t &= f_t, \\
        \log P^{B}_t &= \alpha + \beta f_t + s_t.

    Then the linear combination :math:`\log P^{B}_t - \beta \log P^{A}_t = \alpha + s_t`
    is stationary, so the pair is cointegrated with cointegrating vector
    :math:`(\,-\beta,\ 1\,)` and the spread mean-reverts at the OU speed. The half-life of
    the tradable spread is exactly ``spread_params.half_life``.

    We build the series in log space and exponentiate so prices are strictly positive, the
    way real equity prices behave.

    Parameters
    ----------
    beta:
        The true hedge ratio (recovered later by cointegration / Kalman filtering).
    alpha:
        Intercept of the cointegrating relation.
    common_sigma:
        Volatility of the shared random-walk trend.
    spread_params:
        OU parameters of the stationary spread. Defaults to a moderately fast reverter
        (``theta=0.05`` → half-life ≈ 13.9 days, ``sigma=0.02``).
    p0:
        Approximate initial price level of asset A.
    names:
        Column names for the two assets.

    Returns
    -------
    pandas.DataFrame
        Columns ``names`` indexed by business day; values are positive prices.
    """
    if spread_params is None:
        spread_params = OUParams(theta=0.05, mu=0.0, sigma=0.02)

    log_p0 = float(np.log(p0))
    common_trend = simulate_random_walk(n, rng, sigma=common_sigma * 0.01, x0=log_p0)
    spread = simulate_ou(n, spread_params, rng, x0=spread_params.mu)

    log_a = common_trend
    log_b = alpha + beta * common_trend + spread

    frame = pd.DataFrame(
        {names[0]: np.exp(log_a), names[1]: np.exp(log_b)},
        index=_business_day_index(n, start),
    )
    return frame


def simulate_correlated_ou_panel(
    n: int,
    rng: np.random.Generator,
    *,
    n_pairs: int = 3,
    n_noise: int = 4,
    start: str = _DEFAULT_START,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Build a wide price panel with a *known* set of cointegrated pairs.

    The panel contains ``n_pairs`` genuinely cointegrated pairs (columns ``P0a/P0b``,
    ``P1a/P1b`` …) plus ``n_noise`` independent random walks (``N0`` …) that share no
    long-run relationship. This is the ground truth a discovery routine should recover: it
    lets tests assert both power (finds the real pairs) and specificity (rejects noise).

    Returns
    -------
    (panel, truth):
        ``panel`` is a wide price DataFrame; ``truth`` is the list of column-name tuples
        that are actually cointegrated.
    """
    if n_pairs < 0 or n_noise < 0:
        raise ValueError("n_pairs and n_noise must be non-negative")

    columns: dict[str, np.ndarray] = {}
    truth: list[tuple[str, str]] = []

    for i in range(n_pairs):
        # Vary hedge ratio and reversion speed across pairs for a realistic mix.
        beta = float(rng.uniform(0.5, 1.8))
        theta = float(rng.uniform(0.03, 0.12))
        pair = simulate_cointegrated_pair(
            n,
            rng,
            beta=beta,
            alpha=float(rng.uniform(-0.3, 0.3)),
            spread_params=OUParams(theta=theta, mu=0.0, sigma=0.02),
            p0=float(rng.uniform(40.0, 200.0)),
            start=start,
            names=(f"P{i}a", f"P{i}b"),
        )
        columns[f"P{i}a"] = pair[f"P{i}a"].to_numpy()
        columns[f"P{i}b"] = pair[f"P{i}b"].to_numpy()
        truth.append((f"P{i}a", f"P{i}b"))

    for j in range(n_noise):
        p0 = float(rng.uniform(20.0, 300.0))
        log_walk = simulate_random_walk(n, rng, sigma=0.015, x0=float(np.log(p0)))
        columns[f"N{j}"] = np.exp(log_walk)

    panel = pd.DataFrame(columns, index=_business_day_index(n, start))
    return panel, truth
