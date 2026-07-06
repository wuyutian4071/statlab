r"""Z-score signals for a mean-reverting spread.

The trading logic is a threshold state machine on the spread's z-score. Two things matter
for research honesty:

* **Causality.** The rolling z-score at time ``t`` uses only the trailing ``window``
  observations up to and including ``t`` — never future data. (Execution lag — acting on
  the signal at the *next* bar — is enforced later by the backtester, not here.)
* **Hysteresis.** Entry and exit use *different* thresholds (``|z| >= entry`` to open,
  ``|z| <= exit`` to close) so the position does not chatter around a single boundary. A
  wider ``stop`` closes a position whose spread keeps diverging instead of reverting.

Sign convention: a *long-spread* position (``+1``) is opened when the spread is unusually
**low** (``z <= -entry``), betting it reverts up; a *short-spread* position (``-1``) when
the spread is unusually **high** (``z >= +entry``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series | np.ndarray, window: int) -> pd.Series:
    """Causal rolling z-score: ``(s_t - mean) / std`` over the trailing ``window``.

    The first ``window - 1`` values are ``NaN`` (insufficient history) and a zero-variance
    window yields ``NaN`` rather than a divide-by-zero.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    s = pd.Series(np.asarray(spread, dtype=float))
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    z = (s - mean) / std.replace(0.0, np.nan)
    return z


@dataclass(frozen=True)
class SignalParams:
    """Thresholds for the z-score state machine (on the *absolute* z-score)."""

    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.exit < self.entry < self.stop):
            raise ValueError("require 0 <= exit < entry < stop")


def generate_positions(
    zscore: pd.Series | np.ndarray, params: SignalParams | None = None
) -> np.ndarray:
    """Turn a z-score series into target positions in ``{-1, 0, +1}``.

    The returned position at index ``t`` is the target *after* observing ``z_t`` — a
    stateful walk that opens, holds, and closes according to :class:`SignalParams`. ``NaN``
    z-scores (warm-up) are treated as "no information": the position is held flat until a
    valid z-score arrives.

    Returns
    -------
    numpy.ndarray
        Integer array of the same length as ``zscore``.
    """
    p = params or SignalParams()
    z = np.asarray(zscore, dtype=float)
    pos = np.zeros(len(z), dtype=int)

    state = 0
    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = state
            continue

        if state == 0:
            if zt <= -p.entry:
                state = 1  # spread cheap -> long the spread
            elif zt >= p.entry:
                state = -1  # spread rich -> short the spread
        elif state == 1:
            # Close if reverted back to the exit band, or stopped out on further divergence.
            if zt >= -p.exit or zt <= -p.stop:
                state = 0
        else:  # state == -1
            if zt <= p.exit or zt >= p.stop:
                state = 0

        pos[t] = state

    return pos
