r"""Half-life of mean reversion.

For a spread that follows a discrete Ornstein-Uhlenbeck / AR(1) process,

.. math::

    \Delta s_t = c + \lambda\, s_{t-1} + \varepsilon_t,

mean reversion requires :math:`-1 < \lambda < 0`. The expected time for a shock to decay
by half is the **half-life**

.. math::

    H = -\frac{\ln 2}{\lambda}.

We estimate :math:`\lambda` by OLS of :math:`\Delta s_t` on :math:`s_{t-1}`. The half-life
is the single most useful number for sizing a pairs trade: it tells you roughly how long
you expect to hold before the spread reverts, which drives both the trading horizon and
whether transaction costs can be earned back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def half_life(spread: pd.Series | np.ndarray) -> float:
    r"""Estimate the mean-reversion half-life of ``spread`` (in observations).

    Returns ``inf`` when the series shows no mean reversion (estimated
    :math:`\lambda \ge 0`), which is the correct signal that the "spread" is really a
    random walk and not tradable.

    Raises
    ------
    ValueError
        If fewer than 3 observations are supplied.
    """
    s = np.asarray(spread, dtype=float)
    if s.ndim != 1:
        raise ValueError("spread must be 1-D")
    if len(s) < 3:
        raise ValueError("need at least 3 observations to estimate a half-life")

    delta = np.diff(s)
    lagged = s[:-1]
    design = np.column_stack([np.ones_like(lagged), lagged])  # [const, s_{t-1}]
    coef, *_ = np.linalg.lstsq(design, delta, rcond=None)
    lam = float(coef[1])

    if lam >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / lam)
