r"""Cointegration testing for pairs trading.

Two non-stationary price series are *cointegrated* if some linear combination of them is
stationary — that combination is the tradable, mean-reverting spread. This module provides
the two workhorse tests:

* :func:`engle_granger` — the two-step Engle-Granger procedure. Step 1 estimates the
  cointegrating vector by OLS; step 2 tests the regression residual for a unit root. We
  compute the hedge ratio ourselves (so it is inspectable) but take the p-value from
  statsmodels' :func:`~statsmodels.tsa.stattools.coint`, which uses the correct MacKinnon
  critical values that account for the residual being *estimated* rather than observed
  (a plain ADF on estimated residuals would be biased toward finding cointegration).
* :func:`johansen` — Johansen's system test, which handles >2 series and estimates the
  cointegration *rank* via the trace statistic.

Everything operates on price *levels* (typically log prices); differencing them away would
destroy the long-run relationship these tests are meant to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class EngleGrangerResult:
    """Outcome of an Engle-Granger test of ``y ~ alpha + beta * x``.

    Attributes
    ----------
    beta, alpha:
        The estimated cointegrating vector: ``spread = y - (alpha + beta * x)``.
    stat:
        The Engle-Granger test statistic (a cointegrating ADF t-statistic).
    pvalue:
        MacKinnon p-value; small ⇒ the spread is stationary ⇒ the pair is cointegrated.
    resid:
        The estimated spread (regression residual).
    """

    beta: float
    alpha: float
    stat: float
    pvalue: float
    resid: np.ndarray

    def is_cointegrated(self, level: float = 0.05) -> bool:
        return self.pvalue < level


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of ``y = alpha + beta * x``; return ``(beta, alpha, residuals)``."""
    design = np.column_stack([np.ones_like(x), x])  # [1, x] -> [alpha, beta]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - design @ coef
    return beta, alpha, resid


def engle_granger(y: pd.Series | np.ndarray, x: pd.Series | np.ndarray) -> EngleGrangerResult:
    """Run the two-step Engle-Granger cointegration test of ``y`` on ``x``.

    Parameters
    ----------
    y, x:
        Price *level* series of equal length (log prices recommended). ``y`` is the
        dependent leg; ``beta`` is the number of units of ``x`` that hedge one unit of
        ``y``.

    Notes
    -----
    The order of ``(y, x)`` matters for a finite sample — Engle-Granger is not symmetric.
    For discovery we simply fix an order; a more elaborate scheme would test both and take
    the more significant direction.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape != x_arr.shape or y_arr.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if len(y_arr) < 20:
        raise ValueError("need at least 20 observations for a meaningful test")

    beta, alpha, resid = _ols_hedge(y_arr, x_arr)
    # statsmodels.coint gives the correctly-tabulated EG statistic and p-value.
    stat, pvalue, _crit = coint(y_arr, x_arr, trend="c", autolag="aic")
    return EngleGrangerResult(beta, alpha, float(stat), float(pvalue), resid)


@dataclass(frozen=True)
class JohansenResult:
    """Outcome of a Johansen trace test on a set of series.

    Attributes
    ----------
    rank:
        Estimated number of cointegrating relations at the chosen significance level.
    trace_stats:
        Trace statistic for each null hypothesis ``r <= i``.
    crit_values:
        Critical values at the chosen level for each null.
    eigenvectors:
        Columns are the estimated cointegrating vectors (first = strongest relation).
    """

    rank: int
    trace_stats: np.ndarray
    crit_values: np.ndarray
    eigenvectors: np.ndarray

    @property
    def is_cointegrated(self) -> bool:
        return self.rank >= 1

    def hedge_ratios(self) -> np.ndarray:
        """The leading cointegrating vector, normalised so the first component is 1."""
        vec = self.eigenvectors[:, 0]
        normalised: np.ndarray = vec / vec[0]
        return normalised


def johansen(
    prices: pd.DataFrame, *, level: float = 0.05, det_order: int = 0, k_ar_diff: int = 1
) -> JohansenResult:
    """Johansen trace test for the cointegration rank of a system of price series.

    Parameters
    ----------
    prices:
        Wide frame of price *levels* (columns = series). Must have no missing values.
    level:
        Significance level for the trace test; one of ``{0.10, 0.05, 0.01}``.
    det_order:
        Deterministic trend order passed through to statsmodels (0 = constant).
    k_ar_diff:
        Number of lagged differences in the VECM.
    """
    if prices.isna().any().any():
        raise ValueError("johansen requires a complete (no-NaN) price panel")
    if prices.shape[1] < 2:
        raise ValueError("johansen needs at least two series")

    col = {0.10: 0, 0.05: 1, 0.01: 2}
    if level not in col:
        raise ValueError(f"level must be one of {sorted(col)}, got {level}")
    idx = col[level]

    res = coint_johansen(prices.to_numpy(), det_order, k_ar_diff)
    trace = res.lr1
    crit = res.cvt[:, idx]
    rank = int(np.sum(trace > crit))
    return JohansenResult(rank, trace, crit, res.evec)
