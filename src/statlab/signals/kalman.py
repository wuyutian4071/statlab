r"""A hand-rolled Kalman filter for a *dynamic* hedge ratio.

Static OLS gives one hedge ratio for the whole sample — but real relationships drift, and
using a full-sample OLS beta in a backtest is itself a subtle lookahead: at time ``t`` you
would not have known the beta fitted on data through the end of the sample. A Kalman filter
estimates the hedge ratio *causally* — using only information up to ``t`` — and lets it
evolve over time.

Model
-----
Hidden state (what we want to track) is the regression coefficient vector

.. math::

    \mathbf{x}_t = \begin{bmatrix} \beta_t \\ \alpha_t \end{bmatrix},

which we assume follows a random walk (a flexible "the relationship drifts slowly" prior):

.. math::

    \mathbf{x}_t = \mathbf{x}_{t-1} + \mathbf{w}_t, \qquad
    \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}).

The observation is the dependent leg's price, linear in the state with the *other* leg's
price as the (time-varying) design row :math:`\mathbf{H}_t = [\,x_t,\ 1\,]`:

.. math::

    y_t = \mathbf{H}_t \mathbf{x}_t + v_t, \qquad v_t \sim \mathcal{N}(0, R).

Recursion (predict → update)
----------------------------
Predict (random-walk transition ``F = I``):

.. math::

    \hat{\mathbf{x}}_{t|t-1} = \hat{\mathbf{x}}_{t-1}, \qquad
    \mathbf{P}_{t|t-1} = \mathbf{P}_{t-1} + \mathbf{Q}.

Update, given the *innovation* (one-step forecast error) :math:`e_t`:

.. math::

    e_t &= y_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1}, \\
    S_t &= \mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^\top + R, \\
    \mathbf{K}_t &= \mathbf{P}_{t|t-1}\mathbf{H}_t^\top S_t^{-1}, \\
    \hat{\mathbf{x}}_t &= \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t e_t, \qquad
    \mathbf{P}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)\,\mathbf{P}_{t|t-1}.

The innovation :math:`e_t` **is** the mean-reverting spread, and :math:`e_t/\sqrt{S_t}` is a
naturally normalised z-score — both fall straight out of the filter with no extra
estimation and, crucially, are causal.

Parameterisation (following Chan): the state-drift covariance is set as
:math:`\mathbf{Q} = \tfrac{\delta}{1-\delta}\mathbf{I}` for a small ``delta``. ``delta`` → 0
makes the state (nearly) constant and the filter reduces to recursive least squares, which
converges to full-sample OLS; larger ``delta`` lets the hedge ratio adapt faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanResult:
    """Filtered dynamic-hedge-ratio series (all length ``n``, aligned to the inputs).

    Attributes
    ----------
    beta, alpha:
        Filtered state at each step (causal — uses only data up to that step).
    spread:
        The innovation ``e_t = y_t - (alpha_t + beta_t x_t)`` — the tradable spread.
    innovation_std:
        ``sqrt(S_t)``; ``spread / innovation_std`` is a causal z-score.
    """

    beta: np.ndarray
    alpha: np.ndarray
    spread: np.ndarray
    innovation_std: np.ndarray

    def zscore(self) -> np.ndarray:
        """The causal, filter-native standardised spread ``e_t / sqrt(S_t)``."""
        z: np.ndarray = self.spread / self.innovation_std
        return z


def kalman_hedge(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    *,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    beta0: float = 0.0,
    alpha0: float = 0.0,
    p0: float = 1.0,
) -> KalmanResult:
    """Filter a dynamic hedge ratio for ``y`` regressed on ``x``.

    Parameters
    ----------
    y, x:
        Price *level* series of equal length (log prices recommended). ``beta`` hedges one
        unit of ``y`` with ``beta`` units of ``x``.
    delta:
        State-drift parameter in ``[0, 1)``. ``0`` ⇒ constant coefficients (RLS/OLS);
        larger ⇒ faster adaptation. Typical range ``1e-5`` to ``1e-3``.
    obs_var:
        Observation noise variance ``R``.
    beta0, alpha0, p0:
        Initial state mean and (isotropic) covariance ``P0 = p0 * I``. A larger ``p0``
        encodes a more diffuse prior, letting early data move the estimate quickly.

    Returns
    -------
    KalmanResult
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape != x_arr.shape or y_arr.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if not (0.0 <= delta < 1.0):
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    if obs_var <= 0.0:
        raise ValueError(f"obs_var must be positive, got {obs_var}")

    n = len(y_arr)
    q_scalar = delta / (1.0 - delta) if delta > 0.0 else 0.0
    q_mat = q_scalar * np.eye(2)

    state = np.array([beta0, alpha0], dtype=float)  # [beta, alpha]
    cov = p0 * np.eye(2)

    beta = np.empty(n)
    alpha = np.empty(n)
    spread = np.empty(n)
    innov_std = np.empty(n)

    for t in range(n):
        h = np.array([x_arr[t], 1.0])  # design row [x_t, 1] -> beta*x + alpha

        # Predict (F = I).
        cov = cov + q_mat

        # Innovation.
        y_hat = float(h @ state)
        e = float(y_arr[t] - y_hat)
        s = float(h @ cov @ h + obs_var)

        # Update.
        gain = (cov @ h) / s
        state = state + gain * e
        cov = cov - np.outer(gain, h) @ cov

        beta[t] = state[0]
        alpha[t] = state[1]
        spread[t] = e
        innov_std[t] = np.sqrt(s)

    return KalmanResult(beta, alpha, spread, innov_std)
